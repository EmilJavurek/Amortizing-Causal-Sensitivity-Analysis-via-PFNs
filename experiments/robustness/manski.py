"""Manski envelope diagnostics, dense conditional extrema, and corrected containment."""

from __future__ import annotations
import sys
from pathlib import Path
from experiments.shared.paths import CODE, EXPERIMENTS
import argparse
import json
import time
import numpy as np
import pandas as pd
from experiments.shared.runtime import canonical_json_sha256, sha, utc_now, write
import traceback
import uuid
import torch
from setting_standard.causaldataset import discover_causal_artifacts
from experiments.shared.runtime import (
    canonical_json_sha256,
    provenance,
    hardware_snapshot,
    sha,
    utc_now,
    write,
)
from experiments.shared.runtime import (
    canonical_json_sha256,
    provenance,
    sha,
    utc_now,
    write,
)


ANALYSIS_TOLERANCE = 1e-7
ANALYSIS_BOOTSTRAP_REPETITIONS = 2_000
ANALYSIS_BOOTSTRAP_SEED = 20_260_805
WEIGHT_BINS = [-np.inf, 0.01, 0.05, 0.10, 0.25, 0.50, np.inf]
WEIGHT_LABELS = [
    "<=0.01",
    "(0.01,0.05]",
    "(0.05,0.10]",
    "(0.10,0.25]",
    "(0.25,0.50]",
    ">0.50",
]
SCALE_BINS = [-np.inf, 0.01, 0.025, 0.05, 0.10, 0.25, np.inf]
SCALE_LABELS = [
    "<=0.01",
    "(0.01,0.025]",
    "(0.025,0.05]",
    "(0.05,0.10]",
    "(0.10,0.25]",
    ">0.25",
]
ORACLE_WIDTH_BINS = [-np.inf, 0.10, 0.25, 0.50, 1.0, np.inf]
ORACLE_WIDTH_LABELS = ["<=0.10", "(0.10,0.25]", "(0.25,0.50]", "(0.50,1]", ">1"]

def analysis_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2a-run-dir", type=Path, required=True)
    parser.add_argument("--e2b-run-dir", type=Path, required=True)
    parser.add_argument("--e2a-dense-run-dir", type=Path, required=True)
    parser.add_argument("--e2b-dense-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def analysis_prediction_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "predictions_and_references.parquet",
        run_dir / "predictions.parquet",
    ]
    found = [path for path in candidates if path.is_file()]
    assert len(found) == 1, (run_dir, found)
    return found[0]

def add_diagnostics(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    required = {
        "dataset_id",
        "source_row_index",
        "a_query",
        "gamma",
        "pi_query",
        "counterfactual_weight",
        "q0",
        "envelope_lower",
        "envelope_upper",
        "theta_lower",
        "theta_upper",
        "oracle_width",
        "lower_mean",
        "upper_mean",
        "lower_q05",
        "upper_q95",
    }
    assert required <= set(frame.columns)
    result = frame.copy()
    result["cohort"] = cohort
    result["oracle_contained"] = (
        result["envelope_lower"] <= result["theta_lower"] + ANALYSIS_TOLERANCE
    ) & (result["theta_upper"] <= result["envelope_upper"] + ANALYSIS_TOLERANCE)
    assert result["oracle_contained"].all()
    result["lower_slack"] = result["q0"] - result["envelope_lower"]
    result["upper_slack"] = result["envelope_upper"] - result["q0"]
    assert (result[["lower_slack", "upper_slack"]] >= -ANALYSIS_TOLERANCE).all().all()
    result["lower_signed_prediction_error_q0"] = result["lower_mean"] - result["q0"]
    result["upper_signed_prediction_error_q0"] = result["upper_mean"] - result["q0"]
    result["lower_abs_prediction_error_q0"] = result[
        "lower_signed_prediction_error_q0"
    ].abs()
    result["upper_abs_prediction_error_q0"] = result[
        "upper_signed_prediction_error_q0"
    ].abs()
    result["lower_envelope_violation"] = np.maximum(
        result["envelope_lower"] - result["lower_mean"], 0.0
    )
    result["upper_envelope_violation"] = np.maximum(
        result["upper_mean"] - result["envelope_upper"], 0.0
    )
    result["lower_mean_contained"] = result["lower_envelope_violation"] <= ANALYSIS_TOLERANCE
    result["upper_mean_contained"] = result["upper_envelope_violation"] <= ANALYSIS_TOLERANCE
    result["both_means_contained"] = (
        result["lower_mean_contained"] & result["upper_mean_contained"]
    )
    result["outer_quantiles_contained"] = (
        result["lower_q05"] >= result["envelope_lower"] - ANALYSIS_TOLERANCE
    ) & (result["upper_q95"] <= result["envelope_upper"] + ANALYSIS_TOLERANCE)
    result["gamma_region"] = np.where(
        np.isclose(result["gamma"], 1.0), "Gamma=1", "Gamma>1"
    )
    result["weight_bin"] = pd.cut(
        result["counterfactual_weight"], WEIGHT_BINS, labels=WEIGHT_LABELS
    ).astype(str)
    result["oracle_width_bin"] = pd.cut(
        result["oracle_width"],
        ORACLE_WIDTH_BINS,
        labels=ORACLE_WIDTH_LABELS,
    ).astype(str)
    for side in ("lower", "upper"):
        result[f"{side}_slack_bin"] = pd.cut(
            result[f"{side}_slack"], SCALE_BINS, labels=SCALE_LABELS
        ).astype(str)
        result[f"{side}_error_bin"] = pd.cut(
            result[f"{side}_abs_prediction_error_q0"],
            SCALE_BINS,
            labels=SCALE_LABELS,
        ).astype(str)
    assert np.isfinite(
        result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all()
    return result

def long_side_table(frame: pd.DataFrame) -> pd.DataFrame:
    common = [
        "cohort",
        "dataset_id",
        "source_row_index",
        "a_query",
        "gamma",
        "gamma_region",
        "counterfactual_weight",
        "weight_bin",
        "oracle_width",
        "oracle_width_bin",
    ]
    frames: list[pd.DataFrame] = []
    for side in ("lower", "upper"):
        local = frame[common].copy()
        local["side"] = side
        local["slack"] = frame[f"{side}_slack"]
        local["slack_bin"] = frame[f"{side}_slack_bin"]
        local["signed_prediction_error_q0"] = frame[
            f"{side}_signed_prediction_error_q0"
        ]
        local["abs_prediction_error_q0"] = frame[f"{side}_abs_prediction_error_q0"]
        local["error_bin"] = frame[f"{side}_error_bin"]
        local["envelope_violation"] = frame[f"{side}_envelope_violation"]
        local["contained"] = frame[f"{side}_mean_contained"]
        frames.append(local)
    result = pd.concat(frames, ignore_index=True)
    assert len(result) == 2 * len(frame)
    return result

def grouped_side_metrics(long: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    strata = {
        "gamma": "gamma",
        "gamma_region": "gamma_region",
        "counterfactual_weight": "weight_bin",
        "slack": "slack_bin",
        "oracle_width": "oracle_width_bin",
        "prediction_error_magnitude": "error_bin",
    }
    for stratum_type, column in strata.items():
        for (cohort, side, value), group in long.groupby(
            ["cohort", "side", column], sort=False
        ):
            records.append(
                {
                    "cohort": cohort,
                    "side": side,
                    "stratum_type": stratum_type,
                    "stratum": str(value),
                    "n_dgps": int(group["dataset_id"].nunique()),
                    "n_rows": len(group),
                    "containment_rate": float(group["contained"].mean()),
                    "mean_violation": float(group["envelope_violation"].mean()),
                    "p95_violation": float(group["envelope_violation"].quantile(0.95)),
                    "maximum_violation": float(group["envelope_violation"].max()),
                    "mean_slack": float(group["slack"].mean()),
                    "mean_abs_prediction_error_q0": float(
                        group["abs_prediction_error_q0"].mean()
                    ),
                }
            )
    return pd.DataFrame(records)

def per_dgp_sufficient(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for dataset_id, group in frame.groupby("dataset_id", sort=True):
        record = {
            "dataset_id": str(dataset_id),
            "n": len(group),
            "both_means_contained": group["both_means_contained"].sum(),
            "outer_quantiles_contained": group["outer_quantiles_contained"].sum(),
        }
        for side in ("lower", "upper"):
            record[f"{side}_contained"] = group[f"{side}_mean_contained"].sum()
            record[f"{side}_violation"] = group[f"{side}_envelope_violation"].sum()
        records.append(record)
    return pd.DataFrame(records)

def bootstrap(
    frame: pd.DataFrame, cohort: str, rng: np.random.Generator
) -> pd.DataFrame:
    sufficient = per_dgp_sufficient(frame)
    count = len(sufficient)
    indices = rng.integers(0, count, size=(ANALYSIS_BOOTSTRAP_REPETITIONS, count))
    denominator = sufficient["n"].to_numpy()[indices].sum(axis=1)
    columns = [
        "both_means_contained",
        "outer_quantiles_contained",
        "lower_contained",
        "upper_contained",
        "lower_violation",
        "upper_violation",
    ]
    records: list[dict] = []
    for column in columns:
        values = sufficient[column].to_numpy()[indices].sum(axis=1) / denominator
        records.append(
            {
                "cohort": cohort,
                "metric": (
                    f"mean_{column}"
                    if column.endswith("violation")
                    else f"{column}_rate"
                ),
                "point_estimate": float(
                    sufficient[column].sum() / sufficient["n"].sum()
                ),
                "repetitions": ANALYSIS_BOOTSTRAP_REPETITIONS,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(records)

def bootstrap_gamma_headlines(
    frame: pd.DataFrame,
    cohort: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict] = []
    for gamma in (1.0, 5.0):
        local = frame[np.isclose(frame["gamma"], gamma)]
        sufficient = per_dgp_sufficient(local)
        count = len(sufficient)
        indices = rng.integers(
            0,
            count,
            size=(ANALYSIS_BOOTSTRAP_REPETITIONS, count),
        )
        denominator = sufficient["n"].to_numpy()[indices].sum(axis=1)
        for column in ("both_means_contained", "outer_quantiles_contained"):
            values = sufficient[column].to_numpy()[indices].sum(axis=1) / denominator
            records.append(
                {
                    "cohort": cohort,
                    "gamma": gamma,
                    "metric": f"{column}_rate",
                    "n_dgps": count,
                    "n_rows": len(local),
                    "point_estimate": float(local[column].mean()),
                    "repetitions": ANALYSIS_BOOTSTRAP_REPETITIONS,
                    "lower_95": float(np.quantile(values, 0.025)),
                    "median": float(np.quantile(values, 0.5)),
                    "upper_95": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(records)

def bootstrap_gamma_width_fraction(
    frame: pd.DataFrame,
    cohort: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Whole-DGP bootstrap interval for the median predicted-to-envelope width ratio at Gamma=5."""
    local = frame[np.isclose(frame["gamma"], 5.0)].copy()
    assert (local["envelope_width"] > 0).all()
    local["predicted_to_envelope_width_fraction"] = (
        local["upper_mean"] - local["lower_mean"]
    ) / local["envelope_width"]
    groups = [
        group["predicted_to_envelope_width_fraction"].to_numpy(dtype=np.float64)
        for _, group in local.groupby("dataset_id", sort=True)
    ]
    assert groups and len({len(values) for values in groups}) == 1
    values_by_dgp = np.stack(groups)
    count = len(values_by_dgp)
    indices = rng.integers(
        0,
        count,
        size=(ANALYSIS_BOOTSTRAP_REPETITIONS, count),
    )
    bootstrap_values = np.empty(ANALYSIS_BOOTSTRAP_REPETITIONS, dtype=np.float64)
    chunk_size = 100
    for start in range(0, ANALYSIS_BOOTSTRAP_REPETITIONS, chunk_size):
        stop = min(start + chunk_size, ANALYSIS_BOOTSTRAP_REPETITIONS)
        bootstrap_values[start:stop] = np.median(
            values_by_dgp[indices[start:stop]].reshape(stop - start, -1),
            axis=1,
        )
    point = float(np.median(values_by_dgp))
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "gamma": 5.0,
                "metric": "median_predicted_to_envelope_width_fraction",
                "n_dgps": count,
                "n_rows": len(local),
                "point_estimate": point,
                "repetitions": ANALYSIS_BOOTSTRAP_REPETITIONS,
                "lower_95": float(np.quantile(bootstrap_values, 0.025)),
                "median": float(np.quantile(bootstrap_values, 0.5)),
                "upper_95": float(np.quantile(bootstrap_values, 0.975)),
                "resampling_unit": "whole DGP",
            }
        ]
    )

def add_dense_gap_diagnostics(dense: pd.DataFrame) -> pd.DataFrame:
    """Recover old latent extrema and compare them with dense candidates."""
    result = dense.copy()
    weight = result["counterfactual_weight"].to_numpy(dtype=np.float64)
    assert np.isfinite(weight).all() and (weight > 0).all()
    pi = result["pi_query"].to_numpy(dtype=np.float64)
    q0 = result["q0"].to_numpy(dtype=np.float64)
    result["old_minimum"] = (
        result["envelope_lower"].to_numpy(dtype=np.float64) - pi * q0
    ) / weight
    result["old_maximum"] = (
        result["envelope_upper"].to_numpy(dtype=np.float64) - pi * q0
    ) / weight
    reconstructed_lower = pi * q0 + weight * result["old_minimum"]
    reconstructed_upper = pi * q0 + weight * result["old_maximum"]
    assert np.allclose(
        reconstructed_lower,
        result["envelope_lower"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        reconstructed_upper,
        result["envelope_upper"],
        atol=1e-10,
        rtol=1e-10,
    )
    result["lower_extrema_gap"] = result["dense_minimum"] - result["old_minimum"]
    result["upper_extrema_gap"] = result["dense_maximum"] - result["old_maximum"]
    for side, dense_column in (
        ("lower", "dense_minimum"),
        ("upper", "dense_maximum"),
    ):
        result[f"{side}_absolute_extrema_gap"] = result[
            f"{side}_extrema_gap"
        ].abs()
        result[f"{side}_relative_extrema_gap"] = result[
            f"{side}_absolute_extrema_gap"
        ] / np.maximum(np.abs(result[dense_column]), 1e-8)
        assert np.allclose(
            result[f"{side}_abs_envelope_gap"],
            weight * result[f"{side}_absolute_extrema_gap"],
            atol=1e-8,
            rtol=1e-8,
        )
    return result

def dense_summary(run_dir: Path, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    dense = pd.read_parquet(run_dir / "dense_extrema.parquet")
    assert len(dense) == 32 * 16
    dense["cohort"] = cohort
    dense = add_dense_gap_diagnostics(dense)
    dense["corrected_envelope_lower"] = np.minimum(
        dense["envelope_lower"], dense["dense_envelope_lower"]
    )
    dense["corrected_envelope_upper"] = np.maximum(
        dense["envelope_upper"], dense["dense_envelope_upper"]
    )
    dense["lower_extension"] = (
        dense["envelope_lower"] - dense["corrected_envelope_lower"]
    )
    dense["upper_extension"] = (
        dense["corrected_envelope_upper"] - dense["envelope_upper"]
    )
    records: list[dict] = []
    for side in ("lower", "upper"):
        absolute_extrema_gap = dense[f"{side}_absolute_extrema_gap"]
        relative_extrema_gap = dense[f"{side}_relative_extrema_gap"]
        absolute_envelope_gap = dense[f"{side}_abs_envelope_gap"]
        relative_envelope_gap = dense[f"{side}_relative_envelope_gap"]
        extension = dense[f"{side}_extension"]
        records.append(
            {
                "cohort": cohort,
                "side": side,
                "n_dgps": int(dense["dataset_id"].nunique()),
                "n_queries": len(dense),
                "mean_absolute_extrema_gap": float(absolute_extrema_gap.mean()),
                "p95_absolute_extrema_gap": float(
                    absolute_extrema_gap.quantile(0.95)
                ),
                "maximum_absolute_extrema_gap": float(absolute_extrema_gap.max()),
                "mean_relative_extrema_gap": float(relative_extrema_gap.mean()),
                "p95_relative_extrema_gap": float(
                    relative_extrema_gap.quantile(0.95)
                ),
                "maximum_relative_extrema_gap": float(relative_extrema_gap.max()),
                "mean_absolute_envelope_gap": float(absolute_envelope_gap.mean()),
                "p95_absolute_envelope_gap": float(
                    absolute_envelope_gap.quantile(0.95)
                ),
                "maximum_absolute_envelope_gap": float(
                    absolute_envelope_gap.max()
                ),
                "mean_relative_envelope_gap": float(relative_envelope_gap.mean()),
                "p95_relative_envelope_gap": float(
                    relative_envelope_gap.quantile(0.95)
                ),
                "maximum_relative_envelope_gap": float(relative_envelope_gap.max()),
                "mean_envelope_extension": float(extension.mean()),
                "p95_envelope_extension": float(extension.quantile(0.95)),
                "maximum_envelope_extension": float(extension.max()),
                "queries_extended_over_1e_4": int((extension > 1e-4).sum()),
                "queries_extended_over_1e_3": int((extension > 1e-3).sum()),
            }
        )
    return dense, pd.DataFrame(records)

def analyze() -> None:
    args = analysis_args()
    for path in (
        args.e2a_run_dir,
        args.e2b_run_dir,
        args.e2a_dense_run_dir,
        args.e2b_dense_run_dir,
        args.output_dir,
    ):
        assert path.is_absolute(), path
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    started = time.time()
    input_runs = {
        "E2a_validation": args.e2a_run_dir,
        "E2b_fresh": args.e2b_run_dir,
    }
    diagnostics: list[pd.DataFrame] = []
    long: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    gamma_bootstraps: list[pd.DataFrame] = []
    width_fraction_bootstraps: list[pd.DataFrame] = []
    rng = np.random.default_rng(ANALYSIS_BOOTSTRAP_SEED)
    for cohort, run_dir in input_runs.items():
        frame = add_diagnostics(pd.read_parquet(analysis_prediction_path(run_dir)), cohort)
        assert frame["oracle_contained"].mean() == 1.0
        diagnostics.append(frame)
        long.append(long_side_table(frame))
        bootstraps.append(bootstrap(frame, cohort, rng))
        gamma_bootstraps.append(bootstrap_gamma_headlines(frame, cohort, rng))
        width_fraction_bootstraps.append(
            bootstrap_gamma_width_fraction(frame, cohort, rng)
        )
    diagnostic_frame = pd.concat(diagnostics, ignore_index=True)
    diagnostic_frame.to_parquet(
        args.output_dir / "envelope_diagnostics.parquet", index=False
    )
    long_frame = pd.concat(long, ignore_index=True)
    grouped_side_metrics(long_frame).to_csv(
        args.output_dir / "stratified_metrics.csv", index=False
    )
    pd.concat(bootstraps, ignore_index=True).to_csv(
        args.output_dir / "bootstrap_headlines.csv", index=False
    )
    pd.concat(gamma_bootstraps, ignore_index=True).to_csv(
        args.output_dir / "bootstrap_gamma_headlines.csv", index=False
    )
    pd.concat(width_fraction_bootstraps, ignore_index=True).to_csv(
        args.output_dir / "bootstrap_gamma_width_fraction.csv",
        index=False,
    )

    dense_a, summary_a = dense_summary(args.e2a_dense_run_dir, "E2a_validation")
    dense_b, summary_b = dense_summary(args.e2b_dense_run_dir, "E2b_fresh")
    dense_frame = pd.concat([dense_a, dense_b], ignore_index=True)
    dense_frame.to_parquet(
        args.output_dir / "dense_extrema_comparison.parquet", index=False
    )
    dense_metrics = pd.concat([summary_a, summary_b], ignore_index=True)
    dense_metrics.to_csv(args.output_dir / "dense_extrema_summary.csv", index=False)

    manifest = {
        "experiment": "manski_envelope",
        "status": "completed",
        "started_unix": started,
        "finished_at_utc": utc_now(),
        "finished_unix": time.time(),
        "resolved_config": {
            "oracle_containment_tolerance": ANALYSIS_TOLERANCE,
            "bootstrap_repetitions": ANALYSIS_BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": ANALYSIS_BOOTSTRAP_SEED,
            "bootstrap_unit": "whole DGP",
            "weight_bins": ["-inf", 0.01, 0.05, 0.10, 0.25, 0.50, "inf"],
            "slack_and_error_bins": [
                "-inf",
                0.01,
                0.025,
                0.05,
                0.10,
                0.25,
                "inf",
            ],
            "oracle_width_bins": ["-inf", 0.10, 0.25, 0.50, 1.0, "inf"],
            "dense_correction_rule": (
                "retain old envelope; corrected diagnostic union uses the smaller "
                "old/dense lower and larger old/dense upper"
            ),
            "manski_clipping_deployable": False,
        },
        "input_hashes": {},
        "row_counts": {
            "diagnostics": len(diagnostic_frame),
            "long_side": len(long_frame),
            "dense_queries": len(dense_frame),
        },
        "oracle_containment_rate": {
            cohort: float(
                diagnostic_frame.loc[
                    diagnostic_frame["cohort"].eq(cohort), "oracle_contained"
                ].mean()
            )
            for cohort in input_runs
        },
    }
    manifest["resolved_config_sha256"] = canonical_json_sha256(
        manifest["resolved_config"]
    )
    for cohort, run_dir in input_runs.items():
        manifest["input_hashes"][cohort] = {
            "manifest": sha(run_dir / "finalization_manifest.json"),
            "predictions": sha(analysis_prediction_path(run_dir)),
        }
    for cohort, run_dir in (
        ("E2a_dense", args.e2a_dense_run_dir),
        ("E2b_dense", args.e2b_dense_run_dir),
    ):
        manifest["input_hashes"][cohort] = {
            "manifest": sha(run_dir / "manifest.json"),
            "dense_extrema": sha(run_dir / "dense_extrema.parquet"),
        }
    manifest["elapsed_seconds"] = manifest["finished_unix"] - started
    outputs = sorted(args.output_dir.iterdir())
    manifest["output_hashes"] = {
        path.name: sha(path) for path in outputs if path.is_file()
    }
    write(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("Manski analysis completed")


CODE_DIR = CODE


MSM_RUN_DIR = CODE_DIR / "artifacts/priors/paper/analytical_msm"
GRID_POINTS = 65_537
GRID_LOW = -25.0
GRID_HIGH = 25.0
LOCAL_STARTS = 8
REFINEMENT_STEPS = 300
REFINEMENT_LR = 0.02
TAIL_POINTS = [-50.0, -25.0, 25.0, 50.0]
RETAINED_DGPS = 32

def dense_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("validation", "fresh"), required=True)
    parser.add_argument("--mode", choices=("smoke", "retained"), required=True)
    parser.add_argument("--e2-run-dir", type=Path, required=True)
    parser.add_argument("--fresh-pool-dir", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dgp-count",
        type=int,
        help="Retained-mode prefix size; defaults to the original 32-DGP audit.",
    )
    return parser.parse_args()

def dense_prediction_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "predictions_and_references.parquet",
        run_dir / "predictions.parquet",
    ]
    found = [path for path in candidates if path.is_file()]
    assert len(found) == 1, (run_dir, found)
    return found[0]

def update_candidates(
    candidate_values: torch.Tensor | None,
    candidate_latent: torch.Tensor | None,
    values: torch.Tensor,
    latent: torch.Tensor,
    largest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(LOCAL_STARTS, values.shape[1])
    local_values, indices = torch.topk(values, count, dim=1, largest=largest)
    local_latent = latent[indices]
    if candidate_values is not None:
        local_values = torch.cat([candidate_values, local_values], dim=1)
        local_latent = torch.cat([candidate_latent, local_latent], dim=1)
        local_values, indices = torch.topk(
            local_values, LOCAL_STARTS, dim=1, largest=largest
        )
        local_latent = torch.gather(local_latent, 1, indices)
    return local_values, local_latent

def dense_extrema(
    outcome_map,
    x_query: torch.Tensor,
    a_query: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    instances = len(x_query)
    grid = torch.linspace(GRID_LOW, GRID_HIGH, GRID_POINTS, device=device)
    best_max_values = None
    best_max_latent = None
    best_min_values = None
    best_min_latent = None
    chunk_size = 4_096
    with torch.no_grad():
        for start in range(0, GRID_POINTS, chunk_size):
            local_grid = grid[start : start + chunk_size]
            latent = local_grid[None, :].expand(instances, -1)
            values = outcome_map.evaluate(x_query, a_query, latent)
            best_max_values, best_max_latent = update_candidates(
                best_max_values,
                best_max_latent,
                values,
                local_grid,
                largest=True,
            )
            best_min_values, best_min_latent = update_candidates(
                best_min_values,
                best_min_latent,
                values,
                local_grid,
                largest=False,
            )
    assert best_max_latent is not None and best_min_latent is not None
    grid_max = best_max_values[:, 0]
    grid_min = best_min_values[:, 0]
    u_max = torch.nn.Parameter(best_max_latent.detach().clone())
    u_min = torch.nn.Parameter(best_min_latent.detach().clone())
    optimizer = torch.optim.Adam([u_max, u_min], lr=REFINEMENT_LR)
    for _ in range(REFINEMENT_STEPS):
        y_max = outcome_map.evaluate(x_query, a_query, u_max)
        y_min = outcome_map.evaluate(x_query, a_query, u_min)
        loss = (-y_max + y_min).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u_max.clamp_(-50.0, 50.0)
            u_min.clamp_(-50.0, 50.0)
    with torch.no_grad():
        refined_max = outcome_map.evaluate(x_query, a_query, u_max).max(dim=1).values
        refined_min = outcome_map.evaluate(x_query, a_query, u_min).min(dim=1).values
        tail_latent = torch.tensor(TAIL_POINTS, device=device)[None, :].expand(
            instances, -1
        )
        tail_values = outcome_map.evaluate(x_query, a_query, tail_latent)
        maximum = torch.maximum(
            torch.maximum(grid_max, refined_max), tail_values.max(dim=1).values
        )
        minimum = torch.minimum(
            torch.minimum(grid_min, refined_min), tail_values.min(dim=1).values
        )
    return (
        minimum.cpu().numpy(),
        maximum.cpu().numpy(),
        {
            "grid_points": GRID_POINTS,
            "grid_low": GRID_LOW,
            "grid_high": GRID_HIGH,
            "grid_chunk_size": chunk_size,
            "local_starts_per_direction": LOCAL_STARTS,
            "refinement_steps": REFINEMENT_STEPS,
            "refinement_learning_rate": REFINEMENT_LR,
            "refinement_clamp": [-50.0, 50.0],
            "tail_points": TAIL_POINTS,
            "maximum_abs_refined_latent": float(
                torch.maximum(u_max.abs(), u_min.abs()).max().detach().cpu()
            ),
        },
    )

def compute_dense() -> None:
    args = dense_args()
    for path in (args.e2_run_dir, args.output_dir):
        assert path.is_absolute(), path
    assert not args.output_dir.exists(), args.output_dir
    if args.cohort == "fresh":
        assert args.fresh_pool_dir is not None and args.fresh_pool_dir.is_absolute()
        artifact_root = args.fresh_pool_dir
    else:
        assert args.fresh_pool_dir is None
        artifact_root = MSM_RUN_DIR
    device = torch.device(args.device)
    assert torch.cuda.is_available() and device.type == "cuda"
    torch.cuda.set_device(device)
    from experiments.shared import msm_reference as legacy

    input_path = dense_prediction_path(args.e2_run_dir)
    prediction = pd.read_parquet(input_path)
    required = {
        "dataset_id",
        "global_position",
        "source_row_index",
        "a_query",
        "pi_query",
        "counterfactual_weight",
        "q0",
        "envelope_lower",
        "envelope_upper",
    }
    assert required <= set(prediction.columns)
    ordered = (
        prediction[["global_position", "dataset_id"]]
        .drop_duplicates()
        .sort_values("global_position")
    )
    if args.mode == "smoke":
        assert args.dgp_count in (None, 2), args.dgp_count
        count = 2
    else:
        count = RETAINED_DGPS if args.dgp_count is None else args.dgp_count
        assert 1 <= count <= len(ordered), (count, len(ordered))
    selected_ids = ordered["dataset_id"].astype(int).tolist()[:count]
    assert len(selected_ids) == count
    subset = prediction[prediction["dataset_id"].isin(selected_ids)].copy()
    base = (
        subset.sort_values("gamma")
        .drop_duplicates(["dataset_id", "source_row_index", "a_query"])
        .sort_values(["global_position", "source_row_index", "a_query"])
    )
    query_rows_by_dgp = base.groupby("dataset_id")["source_row_index"].nunique()
    assert len(query_rows_by_dgp) == count
    assert query_rows_by_dgp.nunique() == 1
    query_rows = int(query_rows_by_dgp.iloc[0])
    assert query_rows in (2, 8)
    assert len(base) == count * query_rows * 2
    artifacts_list = discover_causal_artifacts(str(artifact_root))
    artifacts = {int(item["dataset_id"]): item for item in artifacts_list}
    assert set(selected_ids) <= set(artifacts)

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "fragments").mkdir()
    (args.output_dir / "completion_state").mkdir()
    (args.output_dir / "failure_ledger.jsonl").touch()
    config = {
        "cohort": args.cohort,
        "mode": args.mode,
        "subset_selection": (
            f"first ordered {count} E2 DGPs"
            if args.mode == "retained"
            else "first ordered 2 E2 DGPs"
        ),
        "requested_dgp_count": count,
        "selected_dataset_ids": selected_ids,
        "selected_dataset_ids_sha256": canonical_json_sha256(selected_ids),
        "dense_grid_points": GRID_POINTS,
        "dense_grid_range": [GRID_LOW, GRID_HIGH],
        "local_starts_per_direction": LOCAL_STARTS,
        "refinement_steps": REFINEMENT_STEPS,
        "refinement_learning_rate": REFINEMENT_LR,
        "tail_points": TAIL_POINTS,
    }
    manifest = {
        "experiment_uuid": str(uuid.uuid4()),
        "experiment": "manski_dense_extrema",
        "status": "running",
        "started_at_utc": utc_now(),
        "started_unix": time.time(),
        "device": str(device),
        "resolved_config": {
            "values": config,
            "config_sha256": canonical_json_sha256(config),
        },
        "e2_run_dir": str(args.e2_run_dir),
        "e2_manifest_sha256": sha(
            args.e2_run_dir / "finalization_manifest.json"
        ),
        "e2_predictions_sha256": sha(input_path),
        "artifact_root": str(artifact_root),
        "git": provenance(),
        "hardware_start": hardware_snapshot(),
        "completed_dataset_ids": [],
    }
    write(args.output_dir / "manifest.json", manifest)
    x_columns = sorted(
        [column for column in base if column.startswith("x_")],
        key=lambda value: int(value.split("_")[1]),
    )
    assert len(x_columns) == 10
    try:
        frames: list[pd.DataFrame] = []
        for dataset_id in selected_ids:
            started = time.time()
            try:
                local = base[base["dataset_id"].eq(dataset_id)].copy()
                x_query = torch.tensor(
                    local[x_columns].to_numpy(),
                    dtype=torch.float32,
                    device=device,
                )
                a_query = torch.tensor(
                    local["a_query"].to_numpy(),
                    dtype=torch.float32,
                    device=device,
                )
                outcome_map = legacy.load_outcome_map(artifacts[dataset_id], device)
                minimum, maximum, diagnostics = dense_extrema(
                    outcome_map, x_query, a_query, device
                )
                pi = local["pi_query"].to_numpy(dtype=float)
                weight = local["counterfactual_weight"].to_numpy(dtype=float)
                q0 = local["q0"].to_numpy(dtype=float)
                old_lower = local["envelope_lower"].to_numpy(dtype=float)
                old_upper = local["envelope_upper"].to_numpy(dtype=float)
                dense_lower = pi * q0 + weight * minimum
                dense_upper = pi * q0 + weight * maximum
                local["dense_minimum"] = minimum
                local["dense_maximum"] = maximum
                local["dense_envelope_lower"] = dense_lower
                local["dense_envelope_upper"] = dense_upper
                local["lower_envelope_gap"] = dense_lower - old_lower
                local["upper_envelope_gap"] = dense_upper - old_upper
                local["lower_abs_envelope_gap"] = np.abs(local["lower_envelope_gap"])
                local["upper_abs_envelope_gap"] = np.abs(local["upper_envelope_gap"])
                local["lower_relative_envelope_gap"] = local[
                    "lower_abs_envelope_gap"
                ] / np.maximum(np.abs(dense_lower), 1e-8)
                local["upper_relative_envelope_gap"] = local[
                    "upper_abs_envelope_gap"
                ] / np.maximum(np.abs(dense_upper), 1e-8)
                assert np.isfinite(
                    local.select_dtypes(include=[np.number]).to_numpy()
                ).all()
                assert (dense_lower <= q0 + 1e-8).all()
                assert (q0 <= dense_upper + 1e-8).all()
                fragment = args.output_dir / "fragments" / f"dgp_{dataset_id}.parquet"
                local.to_parquet(fragment, index=False)
                write(
                    args.output_dir / "completion_state" / f"dgp_{dataset_id}.json",
                    {
                        "status": "completed",
                        "dataset_id": dataset_id,
                        "rows": len(local),
                        "elapsed_seconds": time.time() - started,
                        "fragment_sha256": sha(fragment),
                        "diagnostics": diagnostics,
                    },
                )
                frames.append(local)
                manifest["completed_dataset_ids"].append(dataset_id)
                write(args.output_dir / "manifest.json", manifest)
                print(
                    f"e5_dense_completed={len(frames)}/{count} dataset_id={dataset_id}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "dataset_id": dataset_id,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                    "at_utc": utc_now(),
                }
                with (args.output_dir / "failure_ledger.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                raise
        merged = pd.concat(frames, ignore_index=True)
        assert len(merged) == len(base)
        merged.to_parquet(args.output_dir / "dense_extrema.parquet", index=False)
        manifest["dense_extrema_sha256"] = sha(
            args.output_dir / "dense_extrema.parquet"
        )
        manifest["maximum_absolute_envelope_gap"] = float(
            merged[["lower_abs_envelope_gap", "upper_abs_envelope_gap"]]
            .to_numpy()
            .max()
        )
        manifest["hardware_finish"] = hardware_snapshot()
        manifest["finished_at_utc"] = utc_now()
        manifest["finished_unix"] = time.time()
        manifest["elapsed_seconds"] = (
            manifest["finished_unix"] - manifest["started_unix"]
        )
        manifest["status"] = "completed"
        write(args.output_dir / "manifest.json", manifest)
        print("Manski extrema completed")
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now()
        manifest["fatal_traceback"] = traceback.format_exc()
        write(args.output_dir / "manifest.json", manifest)
        raise

"""Full-cohort corrected Manski-envelope containment from retained E2/E5 artifacts."""


CONTAINMENT_BOOTSTRAP_REPETITIONS = 2_000
CONTAINMENT_BOOTSTRAP_SEED = 20_260_805
CONTAINMENT_TOLERANCE = 1e-7
COHORTS = {
    "E2a_validation": 1_000,
    "E2b_fresh": 512,
}
KEYS = ["dataset_id", "source_row_index", "a_query"]

def containment_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-analysis-dir", type=Path, required=True)
    parser.add_argument("--validation-dense-run-dir", type=Path, required=True)
    parser.add_argument("--fresh-dense-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def read_json(path: Path) -> dict:
    assert path.is_file(), path
    return json.loads(path.read_text(encoding="utf-8"))

def validate_dense_run(run_dir: Path, expected_count: int) -> tuple[pd.DataFrame, dict]:
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["status"] == "completed", manifest["status"]
    config = manifest["resolved_config"]["values"]
    assert config["mode"] == "retained"
    assert config["requested_dgp_count"] == expected_count
    completed_ids = [int(value) for value in manifest["completed_dataset_ids"]]
    assert len(completed_ids) == expected_count
    assert len(set(completed_ids)) == expected_count
    failure_ledger = run_dir / "failure_ledger.jsonl"
    assert failure_ledger.is_file() and not failure_ledger.read_text().strip()
    dense_path = run_dir / "dense_extrema.parquet"
    assert sha(dense_path) == manifest["dense_extrema_sha256"]
    dense = pd.read_parquet(dense_path)
    assert len(dense) == expected_count * 16
    assert dense["dataset_id"].nunique() == expected_count
    assert not dense.duplicated(KEYS).any()
    assert set(completed_ids) == set(dense["dataset_id"].astype(int))
    required = {
        *KEYS,
        "dense_envelope_lower",
        "dense_envelope_upper",
        "envelope_lower",
        "envelope_upper",
    }
    assert required <= set(dense.columns)
    numeric = dense[list(required - set(KEYS))].to_numpy(dtype=np.float64)
    assert np.isfinite(numeric).all()
    return dense, manifest

def attach_corrected_envelope(
    old: pd.DataFrame,
    dense: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    local = old[old["cohort"].eq(cohort)].copy()
    expected_count = COHORTS[cohort]
    assert local["dataset_id"].nunique() == expected_count
    assert not local.duplicated([*KEYS, "gamma"]).any()
    dense_local = dense[
        [
            *KEYS,
            "dense_envelope_lower",
            "dense_envelope_upper",
            "envelope_lower",
            "envelope_upper",
        ]
    ].rename(
        columns={
            "envelope_lower": "dense_source_old_envelope_lower",
            "envelope_upper": "dense_source_old_envelope_upper",
        }
    )
    assert not dense_local.duplicated(KEYS).any()
    result = local.merge(dense_local, on=KEYS, how="left", validate="many_to_one")
    assert result["dense_envelope_lower"].notna().all()
    assert result["dense_envelope_upper"].notna().all()
    assert np.allclose(
        result["envelope_lower"],
        result["dense_source_old_envelope_lower"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        result["envelope_upper"],
        result["dense_source_old_envelope_upper"],
        atol=1e-10,
        rtol=1e-10,
    )
    result["corrected_envelope_lower"] = np.minimum(
        result["envelope_lower"], result["dense_envelope_lower"]
    )
    result["corrected_envelope_upper"] = np.maximum(
        result["envelope_upper"], result["dense_envelope_upper"]
    )
    result["lower_extension"] = (
        result["envelope_lower"] - result["corrected_envelope_lower"]
    )
    result["upper_extension"] = (
        result["corrected_envelope_upper"] - result["envelope_upper"]
    )
    assert (result[["lower_extension", "upper_extension"]] >= -1e-12).all().all()
    for side in ("lower", "upper"):
        if side == "lower":
            result["corrected_lower_mean_contained"] = result["lower_mean"].ge(
                result["corrected_envelope_lower"] - CONTAINMENT_TOLERANCE
            )
        else:
            result["corrected_upper_mean_contained"] = result["upper_mean"].le(
                result["corrected_envelope_upper"] + CONTAINMENT_TOLERANCE
            )
    result["corrected_both_means_contained"] = (
        result["corrected_lower_mean_contained"]
        & result["corrected_upper_mean_contained"]
    )
    result["corrected_outer_quantiles_contained"] = result["lower_q05"].ge(
        result["corrected_envelope_lower"] - CONTAINMENT_TOLERANCE
    ) & result["upper_q95"].le(result["corrected_envelope_upper"] + CONTAINMENT_TOLERANCE)
    result["corrected_oracle_contained"] = result["theta_lower"].ge(
        result["corrected_envelope_lower"] - CONTAINMENT_TOLERANCE
    ) & result["theta_upper"].le(result["corrected_envelope_upper"] + CONTAINMENT_TOLERANCE)
    assert result["oracle_contained"].all()
    assert result["corrected_oracle_contained"].all()
    assert (
        result["corrected_lower_mean_contained"] >= result["lower_mean_contained"]
    ).all()
    assert (
        result["corrected_upper_mean_contained"] >= result["upper_mean_contained"]
    ).all()
    assert (
        result["corrected_both_means_contained"] >= result["both_means_contained"]
    ).all()
    return result

def clustered_comparison(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    ids = np.sort(frame["dataset_id"].unique())
    assert len(ids) == COHORTS[cohort]
    rng = np.random.default_rng(CONTAINMENT_BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ids), size=(CONTAINMENT_BOOTSTRAP_REPETITIONS, len(ids)))
    scopes = {
        "overall": np.ones(len(frame), dtype=bool),
        "Gamma=1": np.isclose(frame["gamma"], 1.0),
        "Gamma=5": np.isclose(frame["gamma"], 5.0),
    }
    metrics = {
        "both_means": ("both_means_contained", "corrected_both_means_contained"),
        "lower_mean": ("lower_mean_contained", "corrected_lower_mean_contained"),
        "upper_mean": ("upper_mean_contained", "corrected_upper_mean_contained"),
        "outer_quantiles": (
            "outer_quantiles_contained",
            "corrected_outer_quantiles_contained",
        ),
    }
    records: list[dict] = []
    for scope, mask in scopes.items():
        local = frame.loc[mask]
        assert local["dataset_id"].nunique() == len(ids)
        for metric, (old_column, corrected_column) in metrics.items():
            sufficient = (
                local.assign(_n=1.0)
                .groupby("dataset_id")[["_n", old_column, corrected_column]]
                .sum()
                .reindex(ids, fill_value=0.0)
                .to_numpy(dtype=np.float64)
            )
            totals = sufficient[draws].sum(axis=1)
            assert np.all(totals[:, 0] > 0)
            old_draws = totals[:, 1] / totals[:, 0]
            corrected_draws = totals[:, 2] / totals[:, 0]
            delta_draws = corrected_draws - old_draws
            old_point = sufficient[:, 1].sum() / sufficient[:, 0].sum()
            corrected_point = sufficient[:, 2].sum() / sufficient[:, 0].sum()
            assert corrected_point >= old_point - 1e-12
            records.append(
                {
                    "cohort": cohort,
                    "scope": scope,
                    "metric": metric,
                    "n_dgps": len(ids),
                    "n_rows": len(local),
                    "old_rate": old_point,
                    "corrected_rate": corrected_point,
                    "corrected_ci95_lower": np.quantile(corrected_draws, 0.025),
                    "corrected_bootstrap_median": np.quantile(corrected_draws, 0.5),
                    "corrected_ci95_upper": np.quantile(corrected_draws, 0.975),
                    "corrected_minus_old": corrected_point - old_point,
                    "delta_ci95_lower": np.quantile(delta_draws, 0.025),
                    "delta_bootstrap_median": np.quantile(delta_draws, 0.5),
                    "delta_ci95_upper": np.quantile(delta_draws, 0.975),
                    "bootstrap_repetitions": CONTAINMENT_BOOTSTRAP_REPETITIONS,
                    "bootstrap_seed": CONTAINMENT_BOOTSTRAP_SEED,
                    "bootstrap_unit": "whole DGP",
                }
            )
    return pd.DataFrame(records)

def extension_summary(dense: pd.DataFrame, cohort: str) -> pd.DataFrame:
    corrected_lower = np.minimum(
        dense["envelope_lower"], dense["dense_envelope_lower"]
    )
    corrected_upper = np.maximum(
        dense["envelope_upper"], dense["dense_envelope_upper"]
    )
    extensions = {
        "lower": dense["envelope_lower"] - corrected_lower,
        "upper": corrected_upper - dense["envelope_upper"],
    }
    records: list[dict] = []
    for side, values in extensions.items():
        assert (values >= -1e-12).all()
        records.append(
            {
                "cohort": cohort,
                "side": side,
                "n_dgps": dense["dataset_id"].nunique(),
                "n_base_queries": len(dense),
                "fraction_extended_gt_1e-4": (values > 1e-4).mean(),
                "fraction_extended_gt_1e-3": (values > 1e-3).mean(),
                "mean_extension": values.mean(),
                "p95_extension": values.quantile(0.95),
                "maximum_extension": values.max(),
            }
        )
    return pd.DataFrame(records)

def summary_markdown(results: pd.DataFrame, extensions: pd.DataFrame) -> str:
    primary = results[
        results["metric"].eq("both_means")
        & results["scope"].isin(["overall", "Gamma=1", "Gamma=5"])
    ].copy()
    lines = [
        "# E5 full-cohort corrected-envelope containment",
        "",
        "The corrected envelope is the union of the retained old envelope and the full-cohort ",
        "65,537-point dense search, eight-start 300-step local refinement, and tail audit.",
        "",
        "| Cohort | Scope | Old rate | Corrected rate [95% whole-DGP CI] | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.scope} | {row.old_rate:.5f} | "
            f"{row.corrected_rate:.5f} [{row.corrected_ci95_lower:.5f}, "
            f"{row.corrected_ci95_upper:.5f}] | {row.corrected_minus_old:+.5f} |"
        )
    lines.extend(
        [
            "",
            "Analytical-oracle containment is 1.0 in both cohorts before and after expansion.",
            "The correction is an oracle-dependent diagnostic, not a deployable model repair.",
            "",
            "## Full-cohort envelope extensions",
            "",
            "| Cohort | Side | Fraction extended >1e-3 | Mean extension | Max extension |",
            "|---|:---:|---:|---:|---:|",
        ]
    )
    for row in extensions.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.side} | {row.fraction_extended_gt_1e_3:.5f} | "
            f"{row.mean_extension:.6f} | {row.maximum_extension:.6f} |"
        )
    return "\n".join(lines) + "\n"

def analyze_containment() -> None:
    args = containment_args()
    for path in (
        args.old_analysis_dir,
        args.validation_dense_run_dir,
        args.fresh_dense_run_dir,
        args.output_dir,
    ):
        assert path.is_absolute(), path
    assert args.old_analysis_dir.is_dir()
    assert not args.output_dir.exists(), args.output_dir
    args.output_dir.mkdir(parents=True)
    started = time.time()
    old_manifest = read_json(args.old_analysis_dir / "manifest.json")
    assert old_manifest["status"] == "completed"
    old_path = args.old_analysis_dir / "envelope_diagnostics.parquet"
    assert sha(old_path) == old_manifest["output_hashes"][old_path.name]
    old = pd.read_parquet(old_path)
    assert len(old) == 508_032

    validation_dense, validation_manifest = validate_dense_run(
        args.validation_dense_run_dir, COHORTS["E2a_validation"]
    )
    fresh_dense, fresh_manifest = validate_dense_run(
        args.fresh_dense_run_dir, COHORTS["E2b_fresh"]
    )
    corrected_frames = [
        attach_corrected_envelope(old, validation_dense, "E2a_validation"),
        attach_corrected_envelope(old, fresh_dense, "E2b_fresh"),
    ]
    corrected = pd.concat(corrected_frames, ignore_index=True)
    assert len(corrected) == len(old)
    result = pd.concat(
        [
            clustered_comparison(frame, cohort)
            for frame, cohort in zip(corrected_frames, COHORTS, strict=True)
        ],
        ignore_index=True,
    )
    extensions = pd.concat(
        [
            extension_summary(validation_dense, "E2a_validation"),
            extension_summary(fresh_dense, "E2b_fresh"),
        ],
        ignore_index=True,
    )
    compact_columns = [
        "cohort",
        "dataset_id",
        "source_row_index",
        "a_query",
        "gamma",
        "envelope_lower",
        "envelope_upper",
        "corrected_envelope_lower",
        "corrected_envelope_upper",
        "lower_extension",
        "upper_extension",
        "both_means_contained",
        "corrected_both_means_contained",
        "lower_mean_contained",
        "corrected_lower_mean_contained",
        "upper_mean_contained",
        "corrected_upper_mean_contained",
        "oracle_contained",
        "corrected_oracle_contained",
    ]
    corrected[compact_columns].to_parquet(
        args.output_dir / "corrected_containment_rows.parquet", index=False
    )
    result.to_csv(args.output_dir / "corrected_containment_bootstrap.csv", index=False)
    extensions.to_csv(args.output_dir / "dense_extension_summary.csv", index=False)
    (args.output_dir / "SUMMARY.md").write_text(
        summary_markdown(result, extensions), encoding="utf-8"
    )
    finished = time.time()
    config = {
        "bootstrap_repetitions": CONTAINMENT_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": CONTAINMENT_BOOTSTRAP_SEED,
        "bootstrap_unit": "whole DGP",
        "correction_rule": "min old/dense lower and max old/dense upper",
        "containment_tolerance": CONTAINMENT_TOLERANCE,
    }
    manifest = {
        "experiment": "manski_corrected_containment",
        "experiment_uuid": str(uuid.uuid4()),
        "status": "completed",
        "started_at_utc": utc_now(),
        "finished_at_utc": utc_now(),
        "elapsed_seconds": finished - started,
        "resolved_config": config,
        "resolved_config_sha256": canonical_json_sha256(config),
        "git": provenance(),
        "inputs": {
            "old_analysis_manifest": sha(
                args.old_analysis_dir / "manifest.json"
            ),
            "old_envelope_diagnostics": sha(old_path),
            "validation_dense_manifest": sha(
                args.validation_dense_run_dir / "manifest.json"
            ),
            "validation_dense_extrema": validation_manifest["dense_extrema_sha256"],
            "fresh_dense_manifest": sha(
                args.fresh_dense_run_dir / "manifest.json"
            ),
            "fresh_dense_extrema": fresh_manifest["dense_extrema_sha256"],
        },
        "row_counts": {
            "diagnostics": len(corrected),
            "validation_dense": len(validation_dense),
            "fresh_dense": len(fresh_dense),
        },
        "oracle_containment": {
            cohort: float(
                frame["corrected_oracle_contained"].mean()
            )
            for frame, cohort in zip(corrected_frames, COHORTS, strict=True)
        },
    }
    manifest["outputs"] = {
        path.name: sha(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    write(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("Manski containment completed")

def main():
    stages = {'analyze': analyze, 'dense': compute_dense, 'containment': analyze_containment}
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('stage', choices=stages)
        parser.parse_args()
    stage = sys.argv.pop(1)
    assert stage in stages, f'Unknown Manski stage: {stage}'
    stages[stage]()

if __name__ == '__main__':
    main()
