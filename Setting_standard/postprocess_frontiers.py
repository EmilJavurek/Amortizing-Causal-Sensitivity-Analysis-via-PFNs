from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)


REQUIRED_FRONTIER_COLUMNS = ["query_id", "bound_type", "gamma_star", "theta_star"]
FM_FRONTIER_COLUMNS = ["query_id", "bound_type", "gamma_star", "theta_star"]
VALID_BOUND_TYPES = {"upper", "lower"}
VALID_DIAGNOSTICS_LEVELS = {"summary", "curve", "row"}

STATIC_DIR_SPECS = [
    ("datasets", "synthetic_continous_dataset_", ".csv"),
    ("metadata", "dgp_", ".json"),
    ("models", "outcome_bnn_", ".pt"),
    ("models", "propensity_bnn_", ".pt"),
    ("scalers", "scaler_", ".json"),
]
QUERY_PREFIX = "queries_"
FRONTIER_PREFIX = "frontier_points_"


def resolve_code_relative_path(path: str) -> str:
    """
    Resolve artifact paths relative to the repository's code directory.

    Example:
    - `Setting_standard/training_data` -> `<repo>/code/Setting_standard/training_data`
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(CODE_ROOT, path))


def require_runtime_imports():
    global np, pd, tqdm
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    return np, pd, tqdm


def extract_dataset_id(path: str, prefix: str) -> int:
    filename = os.path.basename(path)
    match = re.match(rf"^{re.escape(prefix)}(\d+)", filename)
    if match is None:
        raise ValueError(f"Could not extract dataset id from {filename} with prefix {prefix}")
    return int(match.group(1))


def sorted_paths_with_ids(directory: str, prefix: str, suffix: str) -> Dict[int, str]:
    paths = glob.glob(os.path.join(directory, f"{prefix}*{suffix}"))
    mapping = {}
    for path in paths:
        dataset_id = extract_dataset_id(path, prefix)
        mapping[dataset_id] = path
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


def dataset_static_spec() -> tuple:
    for dirname, prefix, suffix in STATIC_DIR_SPECS:
        if dirname == "datasets":
            return prefix, suffix
    assert False, "STATIC_DIR_SPECS must include datasets"


def parse_dataset_ids(dataset_ids: Optional[str]) -> Optional[List[int]]:
    if dataset_ids is None or dataset_ids.strip() == "":
        return None
    tokens = [token for token in re.split(r"[\s,]+", dataset_ids.strip()) if token]
    return sorted({int(token) for token in tokens})


def prepare_output_root(output_root_dir: str, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return

    if os.path.exists(output_root_dir):
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root_dir}")
        if output_root_dir in {"/", ""}:
            raise ValueError(f"Refusing to remove unsafe output root: {output_root_dir}")
        shutil.rmtree(output_root_dir)

    os.makedirs(output_root_dir, exist_ok=False)


def link_or_copy_file(src_path: str, dst_path: str, copy_mode: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src_path, dst_path)
    elif copy_mode == "symlink":
        os.symlink(os.path.abspath(src_path), dst_path)
    elif copy_mode == "hardlink":
        os.link(src_path, dst_path)
    else:
        raise ValueError(f"Unknown copy mode: {copy_mode}")


def materialize_static_artifacts(
    base_root_dir: str,
    frontier_run_dir: str,
    output_root_dir: str,
    dataset_ids: Sequence[int],
    copy_mode: str,
    dry_run: bool,
) -> Dict[str, Dict[int, str]]:
    copied: Dict[str, Dict[int, str]] = {}

    for dirname, prefix, suffix in STATIC_DIR_SPECS:
        src_dir = os.path.join(base_root_dir, dirname)
        if not os.path.isdir(src_dir):
            if dirname == "datasets":
                raise FileNotFoundError(f"Required datasets directory missing: {src_dir}")
            continue

        src_map = sorted_paths_with_ids(src_dir, prefix, suffix)
        copied_key = f"{dirname}:{prefix}"
        copied[copied_key] = {}
        dst_dir = os.path.join(output_root_dir, dirname)
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)

        for dataset_id in dataset_ids:
            if dataset_id not in src_map:
                if dirname == "datasets":
                    raise FileNotFoundError(f"Missing dataset artifact for dataset_id={dataset_id} under {src_dir}")
                continue
            src_path = src_map[dataset_id]
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            copied[copied_key][dataset_id] = dst_path
            if dry_run:
                print(f"Would {copy_mode} {dirname}: {src_path} -> {dst_path}")
            else:
                link_or_copy_file(src_path, dst_path, copy_mode)

    queries_src_dir = os.path.join(frontier_run_dir, "queries")
    queries_src_map = sorted_paths_with_ids(queries_src_dir, QUERY_PREFIX, ".csv")
    queries_dst_dir = os.path.join(output_root_dir, "queries")
    copied["queries"] = {}
    if not dry_run:
        os.makedirs(queries_dst_dir, exist_ok=True)
    for dataset_id in dataset_ids:
        if dataset_id not in queries_src_map:
            raise FileNotFoundError(f"Missing queries artifact for dataset_id={dataset_id} under {queries_src_dir}")
        src_path = queries_src_map[dataset_id]
        dst_path = os.path.join(queries_dst_dir, os.path.basename(src_path))
        copied["queries"][dataset_id] = dst_path
        if dry_run:
            print(f"Would {copy_mode} queries: {src_path} -> {dst_path}")
        else:
            link_or_copy_file(src_path, dst_path, copy_mode)

    return copied


def count_monotonicity_violations(bound_type: str, theta_values: np.ndarray) -> int:
    if len(theta_values) <= 1:
        return 0
    diffs = np.diff(theta_values)
    if bound_type == "upper":
        return int((diffs < 0).sum())
    if bound_type == "lower":
        return int((diffs > 0).sum())
    raise ValueError(f"Unknown bound_type: {bound_type}")


def repair_curve(bound_type: str, theta_values: np.ndarray, repair_method: str) -> np.ndarray:
    if repair_method == "none":
        return theta_values.copy()
    if repair_method == "isotonic":
        raise NotImplementedError("repair-method=isotonic is not implemented yet")
    if repair_method != "cumulative":
        raise ValueError(f"Unknown repair method: {repair_method}")

    if bound_type == "upper":
        return np.maximum.accumulate(theta_values)
    if bound_type == "lower":
        return np.minimum.accumulate(theta_values)
    raise ValueError(f"Unknown bound_type: {bound_type}")


def validate_frontier_dataframe(dataframe: pd.DataFrame, frontier_path: str) -> None:
    missing_columns = [column for column in REQUIRED_FRONTIER_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{frontier_path} is missing required columns: {missing_columns}")

    unknown_bounds = sorted(set(dataframe["bound_type"].astype(str).unique()) - VALID_BOUND_TYPES)
    if unknown_bounds:
        raise ValueError(f"{frontier_path} has invalid bound_type values: {unknown_bounds}")

    for column in ["gamma_star", "theta_star"]:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad_count = int((~np.isfinite(values)).sum())
            raise ValueError(f"{frontier_path} has {bad_count} nonfinite values in {column}")


def compute_monotonicity_violation_mask(dataframe: pd.DataFrame, theta_column: str) -> pd.Series:
    diffs = dataframe.groupby(["query_id", "bound_type"], sort=False)[theta_column].diff()
    upper_mask = dataframe["bound_type"] == "upper"
    lower_mask = dataframe["bound_type"] == "lower"
    return (upper_mask & diffs.lt(0)) | (lower_mask & diffs.gt(0))


def numeric_histogram(values: Sequence[int]) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    for value in values:
        key = int(value)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def postprocess_frontier_dataset(
    dataset_id: int,
    frontier_run_dir: str,
    output_root_dir: str,
    repair_method: str = "cumulative",
    diagnostics_level: str = "summary",
    dry_run: bool = False,
) -> Dict:
    frontier_path = os.path.join(frontier_run_dir, "frontiers", f"frontier_points_{dataset_id}.csv")
    output_frontier_path = os.path.join(output_root_dir, "frontiers", f"frontier_points_{dataset_id}.csv")
    diagnostics_path = os.path.join(
        output_root_dir,
        "postprocess_diagnostics",
        f"postprocess_diagnostics_{dataset_id}.csv",
    )
    curve_summary_path = os.path.join(
        output_root_dir,
        "postprocess_diagnostics",
        f"curve_summary_{dataset_id}.csv",
    )

    if dry_run:
        print(f"Would read frontier: {frontier_path}")
        print(f"Would write frontier: {output_frontier_path}")
        if diagnostics_level == "row":
            print(f"Would write diagnostics: {diagnostics_path}")
        if diagnostics_level in {"curve", "row"}:
            print(f"Would write curve summary: {curve_summary_path}")
        return {
            "dataset_id": dataset_id,
            "input_rows": 0,
            "output_rows": 0,
            "total_curves": 0,
            "total_repaired_points": 0,
            "monotonicity_violations_before": 0,
            "monotonicity_violations_after": 0,
            "repair_delta_abs_summary": summarize_numeric([]),
            "repair_delta_abs_count": 0,
            "point_counts_by_query_histogram": {},
            "point_counts_by_query_bound_histogram": {},
        }

    require_runtime_imports()
    if diagnostics_level not in VALID_DIAGNOSTICS_LEVELS:
        raise ValueError(f"Unknown diagnostics level: {diagnostics_level}")
    if repair_method == "isotonic":
        raise NotImplementedError("repair-method=isotonic is not implemented yet")

    dataframe = pd.read_csv(frontier_path)
    validate_frontier_dataframe(dataframe, frontier_path)

    work = dataframe.copy()
    work["query_id"] = pd.to_numeric(work["query_id"], errors="raise")
    work["bound_type"] = work["bound_type"].astype(str)
    work["gamma_star"] = pd.to_numeric(work["gamma_star"], errors="raise")
    work["theta_star"] = pd.to_numeric(work["theta_star"], errors="raise")
    work["original_row_index"] = np.arange(len(work), dtype=np.int64)

    sorted_df = work.sort_values(
        by=["query_id", "bound_type", "gamma_star", "original_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    if repair_method == "none":
        theta_repaired = sorted_df["theta_star"].copy()
    elif repair_method == "cumulative":
        upper_mask = sorted_df["bound_type"] == "upper"
        lower_mask = sorted_df["bound_type"] == "lower"

        theta_repaired = sorted_df["theta_star"].copy()
        theta_repaired.loc[upper_mask] = (
            sorted_df.loc[upper_mask]
            .groupby("query_id", sort=False)["theta_star"]
            .cummax()
        )
        theta_repaired.loc[lower_mask] = (
            sorted_df.loc[lower_mask]
            .groupby("query_id", sort=False)["theta_star"]
            .cummin()
        )
    else:
        raise ValueError(f"Unknown repair method: {repair_method}")

    repair_delta = theta_repaired - sorted_df["theta_star"]
    repair_delta_abs = repair_delta.abs()
    was_repaired = repair_delta_abs > 0

    output_df = sorted_df[["query_id", "bound_type", "gamma_star"]].copy()
    output_df["theta_star"] = theta_repaired
    output_df = output_df[FM_FRONTIER_COLUMNS]

    os.makedirs(os.path.dirname(output_frontier_path), exist_ok=True)
    output_df.to_csv(output_frontier_path, index=False)

    before_violation_mask = compute_monotonicity_violation_mask(sorted_df, "theta_star")
    repaired_for_diagnostics = None
    if diagnostics_level in {"curve", "row"}:
        repaired_for_diagnostics = sorted_df.copy()
        repaired_for_diagnostics["theta_star_repaired"] = theta_repaired
        repaired_for_diagnostics["repair_delta"] = repair_delta
        repaired_for_diagnostics["repair_delta_abs"] = repair_delta_abs
        repaired_for_diagnostics["was_repaired"] = was_repaired
        repaired_for_diagnostics["monotonicity_violation_before"] = before_violation_mask
        repaired_for_diagnostics["monotonicity_violation_after"] = compute_monotonicity_violation_mask(
            repaired_for_diagnostics,
            "theta_star_repaired",
        )
        repaired_for_diagnostics["sort_rank_by_gamma"] = repaired_for_diagnostics.groupby(
            ["query_id", "bound_type"],
            sort=False,
        ).cumcount()
        curve_summary = (
            repaired_for_diagnostics.groupby(["query_id", "bound_type"], sort=False)
            .agg(
                n_points=("theta_star", "size"),
                gamma_min=("gamma_star", "min"),
                gamma_max=("gamma_star", "max"),
                n_repaired=("was_repaired", "sum"),
                max_repair_delta_abs=("repair_delta_abs", "max"),
                mean_repair_delta_abs=("repair_delta_abs", "mean"),
                monotonicity_violations_before=("monotonicity_violation_before", "sum"),
                monotonicity_violations_after=("monotonicity_violation_after", "sum"),
            )
            .reset_index()
        )
        curve_summary.insert(0, "dataset_id", dataset_id)

        os.makedirs(os.path.dirname(curve_summary_path), exist_ok=True)
        curve_summary.to_csv(curve_summary_path, index=False)

    if diagnostics_level == "row":
        diagnostics = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "query_id": repaired_for_diagnostics["query_id"],
                "bound_type": repaired_for_diagnostics["bound_type"],
                "original_row_index": repaired_for_diagnostics["original_row_index"],
                "sort_rank_by_gamma": repaired_for_diagnostics["sort_rank_by_gamma"],
                "gamma_star_raw": repaired_for_diagnostics["gamma_star"],
                "theta_star_raw": repaired_for_diagnostics["theta_star"],
                "theta_star_repaired": repaired_for_diagnostics["theta_star_repaired"],
                "repair_delta": repaired_for_diagnostics["repair_delta"],
                "repair_delta_abs": repaired_for_diagnostics["repair_delta_abs"],
                "was_repaired": repaired_for_diagnostics["was_repaired"],
            }
        )
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
        diagnostics.to_csv(diagnostics_path, index=False)

    if diagnostics_level in {"curve", "row"}:
        monotonicity_violations_after = int(curve_summary["monotonicity_violations_after"].sum())
    elif repair_method == "cumulative":
        monotonicity_violations_after = 0
    else:
        monotonicity_violations_after = int(before_violation_mask.sum())

    point_counts_by_query = sorted_df.groupby("query_id", sort=False).size()
    point_counts_by_query_bound = sorted_df.groupby(["query_id", "bound_type"], sort=False).size()

    return {
        "dataset_id": dataset_id,
        "input_rows": int(len(dataframe)),
        "output_rows": int(len(output_df)),
        "total_curves": int(point_counts_by_query_bound.shape[0]),
        "total_repaired_points": int(was_repaired.sum()),
        "monotonicity_violations_before": int(before_violation_mask.sum()),
        "monotonicity_violations_after": monotonicity_violations_after,
        "repair_delta_abs_summary": summarize_numeric(repair_delta_abs.to_numpy(dtype=np.float64)),
        "repair_delta_abs_count": int(len(repair_delta_abs)),
        "point_counts_by_query_histogram": numeric_histogram(point_counts_by_query.to_numpy(dtype=np.int64)),
        "point_counts_by_query_bound_histogram": numeric_histogram(
            point_counts_by_query_bound.to_numpy(dtype=np.int64)
        ),
    }


def summarize_numeric(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def weighted_percentile(values: Sequence[float], weights: Sequence[int], percentile: float) -> Optional[float]:
    pairs = sorted(
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if value is not None and int(weight) > 0
    )
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    threshold = percentile / 100.0 * total_weight
    cumulative = 0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def merge_histograms(histograms: Sequence[Dict[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for histogram in histograms:
        for value, count in histogram.items():
            key = int(value)
            merged[key] = merged.get(key, 0) + int(count)
    return merged


def summarize_histogram(histogram: Dict[int, int]) -> Dict[str, Optional[float]]:
    if not histogram:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None}
    values = sorted((int(value), int(count)) for value, count in histogram.items())
    return {
        "min": float(values[0][0]),
        "p5": weighted_percentile([value for value, _ in values], [count for _, count in values], 5),
        "median": weighted_percentile([value for value, _ in values], [count for _, count in values], 50),
        "p95": weighted_percentile([value for value, _ in values], [count for _, count in values], 95),
        "max": float(values[-1][0]),
    }


def summarize_repair_delta_from_dataset_summaries(dataset_results: Sequence[Dict]) -> Dict[str, Optional[float]]:
    summaries = [result["repair_delta_abs_summary"] for result in dataset_results]
    weights = [int(result["repair_delta_abs_count"]) for result in dataset_results]
    nonempty = [
        (summary, weight)
        for summary, weight in zip(summaries, weights)
        if weight > 0 and summary["max"] is not None
    ]
    if not nonempty:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None}

    return {
        "min": float(min(summary["min"] for summary, _ in nonempty)),
        "p5": weighted_percentile([summary["p5"] for summary, _ in nonempty], [weight for _, weight in nonempty], 5),
        "median": weighted_percentile(
            [summary["median"] for summary, _ in nonempty],
            [weight for _, weight in nonempty],
            50,
        ),
        "p95": weighted_percentile(
            [summary["p95"] for summary, _ in nonempty],
            [weight for _, weight in nonempty],
            95,
        ),
        "max": float(max(summary["max"] for summary, _ in nonempty)),
    }


def build_collection_summary(
    dataset_results: Sequence[Dict],
    base_root_dir: str,
    frontier_run_dir: str,
    output_root_dir: str,
    repair_method: str,
    copy_mode: str,
    dataset_ids: Sequence[int],
    expected_min_rows_per_query: int,
    diagnostics_level: str,
) -> Dict:
    input_rows = int(sum(result["input_rows"] for result in dataset_results))
    output_rows = int(sum(result["output_rows"] for result in dataset_results))
    total_curves = int(sum(result["total_curves"] for result in dataset_results))
    total_repaired_points = int(sum(result["total_repaired_points"] for result in dataset_results))
    point_count_query_histogram = merge_histograms(
        [result["point_counts_by_query_histogram"] for result in dataset_results]
    )
    point_count_query_bound_histogram = merge_histograms(
        [result["point_counts_by_query_bound_histogram"] for result in dataset_results]
    )

    repair_delta_summary = summarize_repair_delta_from_dataset_summaries(dataset_results)
    return {
        "base_root_dir": base_root_dir,
        "frontier_run_dir": frontier_run_dir,
        "output_root_dir": output_root_dir,
        "repair_method": repair_method,
        "copy_mode": copy_mode,
        "dataset_ids": [int(dataset_id) for dataset_id in dataset_ids],
        "diagnostics_level": diagnostics_level,
        "expected_min_rows_per_query": int(expected_min_rows_per_query),
        "datasets_processed": int(len(dataset_results)),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rows_preserved_fraction": float(output_rows / input_rows) if input_rows else None,
        "total_curves": total_curves,
        "total_repaired_points": total_repaired_points,
        "fraction_repaired": float(total_repaired_points / output_rows) if output_rows else None,
        "repair_delta_abs_median": repair_delta_summary["median"],
        "repair_delta_abs_p95": repair_delta_summary["p95"],
        "repair_delta_abs_max": repair_delta_summary["max"],
        "monotonicity_violations_before": int(
            sum(result["monotonicity_violations_before"] for result in dataset_results)
        ),
        "monotonicity_violations_after": int(
            sum(result["monotonicity_violations_after"] for result in dataset_results)
        ),
        "point_count_per_dataset_query": summarize_histogram(point_count_query_histogram),
        "point_count_per_dataset_query_bound": summarize_histogram(point_count_query_bound_histogram),
    }


def write_summary_files(summary: Dict, output_root_dir: str, dry_run: bool) -> None:
    if dry_run:
        print("Would write postprocess_summary.json and postprocess_summary.md")
        print(json.dumps(summary, indent=2))
        return

    diagnostics_dir = os.path.join(output_root_dir, "postprocess_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    summary_json_path = os.path.join(diagnostics_dir, "postprocess_summary.json")
    summary_md_path = os.path.join(diagnostics_dir, "postprocess_summary.md")

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Frontier Postprocess Summary",
        "",
        f"- Base root dir: {summary['base_root_dir']}",
        f"- Frontier run dir: {summary['frontier_run_dir']}",
        f"- Output root dir: {summary['output_root_dir']}",
        f"- Repair method: {summary['repair_method']}",
        f"- Copy mode: {summary['copy_mode']}",
        f"- Dataset ids: {summary['dataset_ids']}",
        f"- Diagnostics level: {summary['diagnostics_level']}",
        f"- Expected min rows per query: {summary['expected_min_rows_per_query']}",
        f"- Datasets processed: {summary['datasets_processed']}",
        f"- Input rows: {summary['input_rows']}",
        f"- Output rows: {summary['output_rows']}",
        f"- Rows preserved fraction: {summary['rows_preserved_fraction']}",
        f"- Total curves: {summary['total_curves']}",
        f"- Total repaired points: {summary['total_repaired_points']}",
        f"- Fraction repaired: {summary['fraction_repaired']}",
        f"- Repair delta abs median: {summary['repair_delta_abs_median']}",
        f"- Repair delta abs p95: {summary['repair_delta_abs_p95']}",
        f"- Repair delta abs max: {summary['repair_delta_abs_max']}",
        f"- Monotonicity violations before: {summary['monotonicity_violations_before']}",
        f"- Monotonicity violations after: {summary['monotonicity_violations_after']}",
        f"- Point count per (dataset_id, query_id): {summary['point_count_per_dataset_query']}",
        f"- Point count per (dataset_id, query_id, bound_type): {summary['point_count_per_dataset_query_bound']}",
        "",
    ]
    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def discover_dataset_ids(
    base_root_dir: str,
    frontier_run_dir: str,
    requested_dataset_ids: Optional[Sequence[int]],
) -> List[int]:
    dataset_prefix, dataset_suffix = dataset_static_spec()
    dataset_map = sorted_paths_with_ids(
        os.path.join(base_root_dir, "datasets"),
        dataset_prefix,
        dataset_suffix,
    )
    queries_map = sorted_paths_with_ids(os.path.join(frontier_run_dir, "queries"), QUERY_PREFIX, ".csv")
    frontier_map = sorted_paths_with_ids(os.path.join(frontier_run_dir, "frontiers"), FRONTIER_PREFIX, ".csv")

    available_ids = set(dataset_map.keys()) & set(queries_map.keys()) & set(frontier_map.keys())
    if not available_ids:
        raise ValueError(
            "No aligned dataset ids found across base datasets, frontier-run queries, and frontier-run frontiers"
        )

    if requested_dataset_ids is None:
        return sorted(available_ids)

    requested = set(requested_dataset_ids)
    missing = sorted(requested - available_ids)
    if missing:
        raise ValueError(f"Requested dataset ids are not aligned across inputs: {missing}")
    return sorted(requested)


def validate_system_c_compatibility(
    output_root_dir: str,
    expected_min_rows_per_query: int,
    expected_dataset_ids: Set[int],
) -> Dict:
    require_runtime_imports()
    dataset_prefix, dataset_suffix = dataset_static_spec()
    dataset_map = sorted_paths_with_ids(
        os.path.join(output_root_dir, "datasets"),
        dataset_prefix,
        dataset_suffix,
    )
    queries_map = sorted_paths_with_ids(os.path.join(output_root_dir, "queries"), QUERY_PREFIX, ".csv")
    frontier_map = sorted_paths_with_ids(os.path.join(output_root_dir, "frontiers"), FRONTIER_PREFIX, ".csv")

    discovered_ids = set(dataset_map.keys())
    if discovered_ids != set(queries_map.keys()):
        raise ValueError("Output dataset and queries IDs do not match")
    if discovered_ids != set(frontier_map.keys()):
        raise ValueError("Output dataset and frontier IDs do not match")
    if discovered_ids != expected_dataset_ids:
        raise ValueError(
            f"Output artifact ids do not match processed ids: discovered={sorted(discovered_ids)}, "
            f"expected={sorted(expected_dataset_ids)}"
        )

    min_rows_per_query = None
    violations = []
    for dataset_id in sorted(discovered_ids):
        queries = pd.read_csv(queries_map[dataset_id], usecols=["query_id"])
        frontier = pd.read_csv(frontier_map[dataset_id], usecols=["query_id"])

        absent_query_ids = sorted(set(frontier["query_id"].unique()) - set(queries["query_id"].unique()))
        if absent_query_ids:
            raise ValueError(
                f"Dataset {dataset_id} has frontier query_ids absent from queries: "
                f"{absent_query_ids[:20]}"
            )

        counts = frontier.groupby("query_id").size()
        if len(counts) == 0:
            raise ValueError(f"Dataset {dataset_id} has no frontier rows")
        dataset_min = int(counts.min())
        min_rows_per_query = dataset_min if min_rows_per_query is None else min(min_rows_per_query, dataset_min)

        bad_counts = counts[counts < expected_min_rows_per_query]
        for query_id, row_count in bad_counts.items():
            violations.append(
                {
                    "dataset_id": int(dataset_id),
                    "query_id": int(query_id),
                    "row_count": int(row_count),
                }
            )

    if violations:
        preview = violations[:20]
        raise ValueError(
            f"System C compatibility validation failed: {len(violations)} query groups have fewer than "
            f"{expected_min_rows_per_query} rows. First violations: {preview}"
        )

    return {
        "datasets": int(len(discovered_ids)),
        "min_rows_per_query": int(min_rows_per_query) if min_rows_per_query is not None else None,
        "violations": violations,
    }


def postprocess_frontier_collection(
    base_root_dir: str,
    frontier_run_dir: str,
    output_root_dir: str,
    repair_method: str = "cumulative",
    copy_mode: str = "copy",
    overwrite: bool = False,
    dataset_ids: Optional[str] = None,
    dry_run: bool = False,
    expected_min_rows_per_query: int = 10,
    diagnostics_level: str = "summary",
) -> List[Dict]:
    total_start = time.perf_counter()
    resolved_base_root_dir = resolve_code_relative_path(base_root_dir)
    resolved_frontier_run_dir = resolve_code_relative_path(frontier_run_dir)
    resolved_output_root_dir = resolve_code_relative_path(output_root_dir)

    if repair_method == "isotonic":
        raise NotImplementedError("repair-method=isotonic is not implemented yet")
    if diagnostics_level not in VALID_DIAGNOSTICS_LEVELS:
        raise ValueError(f"Unknown diagnostics level: {diagnostics_level}")
    if os.path.abspath(resolved_output_root_dir) in {
        os.path.abspath(resolved_base_root_dir),
        os.path.abspath(resolved_frontier_run_dir),
    }:
        raise ValueError("Output root must be distinct from base-root-dir and frontier-run-dir")
    if not dry_run:
        require_runtime_imports()

    requested_ids = parse_dataset_ids(dataset_ids)
    selected_dataset_ids = discover_dataset_ids(
        base_root_dir=resolved_base_root_dir,
        frontier_run_dir=resolved_frontier_run_dir,
        requested_dataset_ids=requested_ids,
    )

    print(f"Found {len(selected_dataset_ids)} aligned dataset ids to postprocess")
    print(f"Base root: {resolved_base_root_dir}")
    print(f"Frontier run: {resolved_frontier_run_dir}")
    print(f"Output root: {resolved_output_root_dir}")
    print(f"Diagnostics level: {diagnostics_level}")
    if dry_run:
        print(f"Dry run dataset ids: {selected_dataset_ids[:20]}{' ...' if len(selected_dataset_ids) > 20 else ''}")

    materialize_start = time.perf_counter()
    prepare_output_root(output_root_dir=resolved_output_root_dir, overwrite=overwrite, dry_run=dry_run)
    materialize_static_artifacts(
        base_root_dir=resolved_base_root_dir,
        frontier_run_dir=resolved_frontier_run_dir,
        output_root_dir=resolved_output_root_dir,
        dataset_ids=selected_dataset_ids,
        copy_mode=copy_mode,
        dry_run=dry_run,
    )
    print(f"Timing materialize static artifacts: {time.perf_counter() - materialize_start:.2f}s")

    postprocess_start = time.perf_counter()
    dataset_results = []
    progress_iter = selected_dataset_ids if dry_run else tqdm(selected_dataset_ids, desc="Postprocessing frontiers")
    for dataset_id in progress_iter:
        result = postprocess_frontier_dataset(
            dataset_id=dataset_id,
            frontier_run_dir=resolved_frontier_run_dir,
            output_root_dir=resolved_output_root_dir,
            repair_method=repair_method,
            diagnostics_level=diagnostics_level,
            dry_run=dry_run,
        )
        dataset_results.append(result)
    print(f"Timing postprocess frontiers: {time.perf_counter() - postprocess_start:.2f}s")

    summary = build_collection_summary(
        dataset_results=dataset_results,
        base_root_dir=resolved_base_root_dir,
        frontier_run_dir=resolved_frontier_run_dir,
        output_root_dir=resolved_output_root_dir,
        repair_method=repair_method,
        copy_mode=copy_mode,
        dataset_ids=selected_dataset_ids,
        expected_min_rows_per_query=expected_min_rows_per_query,
        diagnostics_level=diagnostics_level,
    )
    write_summary_files(summary=summary, output_root_dir=resolved_output_root_dir, dry_run=dry_run)

    validation_start = time.perf_counter()
    if not dry_run:
        compatibility = validate_system_c_compatibility(
            output_root_dir=resolved_output_root_dir,
            expected_min_rows_per_query=expected_min_rows_per_query,
            expected_dataset_ids=set(selected_dataset_ids),
        )
        print(
            "System C compatibility validation passed: "
            f"{compatibility['datasets']} datasets, min rows per query={compatibility['min_rows_per_query']}"
        )
    print(f"Timing validation: {time.perf_counter() - validation_start:.2f}s")

    print(f"Timing total: {time.perf_counter() - total_start:.2f}s")
    print(f"Frontier postprocessing complete. Output saved to {resolved_output_root_dir}")
    return dataset_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess frontier collections into a System-C-compatible PI-FM data root."
    )
    parser.add_argument(
        "--base-root-dir",
        type=str,
        required=True,
        help="Existing base data root containing at least datasets/.",
    )
    parser.add_argument(
        "--frontier-run-dir",
        type=str,
        required=True,
        help="Existing frontier generation run containing queries/ and frontiers/.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        required=True,
        help="New System-C-compatible output root.",
    )
    parser.add_argument(
        "--repair-method",
        type=str,
        choices=["none", "cumulative", "isotonic"],
        default="cumulative",
        help="Theta repair method. isotonic is reserved and currently raises NotImplementedError.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize static artifacts. Frontiers are always newly written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and replace output root if it already exists.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default=None,
        help="Optional comma/space separated dataset id subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered files and intended outputs without writing.",
    )
    parser.add_argument(
        "--expected-min-rows-per-query",
        type=int,
        default=10,
        help="Minimum total frontier rows required per query_id for System C sampling.",
    )
    parser.add_argument(
        "--diagnostics-level",
        type=str,
        choices=["summary", "curve", "row"],
        default="summary",
        help="Diagnostics detail level: summary only, curve summaries, or full row diagnostics.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "base_root_dir": args.base_root_dir,
        "frontier_run_dir": args.frontier_run_dir,
        "output_root_dir": args.output_root_dir,
        "repair_method": args.repair_method,
        "copy_mode": args.copy_mode,
        "overwrite": args.overwrite,
        "dataset_ids": args.dataset_ids,
        "dry_run": args.dry_run,
        "expected_min_rows_per_query": args.expected_min_rows_per_query,
        "diagnostics_level": args.diagnostics_level,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    postprocess_frontier_collection(**config)


if __name__ == "__main__":
    main()
