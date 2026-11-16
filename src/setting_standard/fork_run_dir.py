"""
fork_run_dir.py — zero-disk-cost run forking via hardlinks.

Creates a new run_dir that contains hardlinks (or copies, when cross-device)
of all upstream artifacts from a source run_dir, then lets the caller run
a subsequent pipeline step in the new directory.

Fork table (which subdirs are upstream for each step):

    Step 2: datasets_raw/, models_raw/, metadata_raw/
    Step 3: all of step 2 + datasets/, models/, metadata/, scalers/
    Step 4: all of step 3 + queries/, frontiers_raw/, metadata_frontier/,
            diagnostics_frontier/ (if present), spline_params/ (if present)

ID-bearing dirs (files matched by *_{id}.*) support per-ID filtering when
dataset_ids is supplied. Run-level dirs are always fully linked. Applicable
upstream run-manifest attempts are copied, never hardlinked, and the destination
receives a new run identity with source lineage and subset information.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


from setting_standard.run_manifest import initialize_fork_run_manifest, validate_fork_provenance


# ============================================================================
# Constants — fork table
# ============================================================================

# ID-bearing subdirs: files inside contain the dataset id in the filename.
# These support per-ID filtering when dataset_ids is provided.
_ID_BEARING_SUBDIRS = {
    "datasets_raw",
    "models_raw",
    "metadata_raw",
    "datasets",
    "models",
    "metadata",
    "scalers",
    "queries",
    "frontiers_raw",
    "diagnostics_frontier",
    "spline_params",
}

# Which subdirs are upstream for each step (must be hardlinked before running).
_UPSTREAM_SUBDIRS: Dict[int, List[str]] = {
    2: [
        "datasets_raw",
        "models_raw",
        "metadata_raw",
    ],
    3: [
        "datasets_raw",
        "models_raw",
        "metadata_raw",
        "datasets",
        "models",
        "metadata",
        "scalers",
    ],
    4: [
        "datasets_raw",
        "models_raw",
        "metadata_raw",
        "datasets",
        "models",
        "metadata",
        "scalers",
        "queries",
        "frontiers_raw",
        "metadata_frontier",
        "diagnostics_frontier",   # optional — skipped if absent
        "spline_params",           # optional — skipped if absent
    ],
}

# Optional subdirs: silently skipped if absent in source (no assertion).
_OPTIONAL_SUBDIRS = {"diagnostics_frontier", "spline_params"}


# ============================================================================
# Internal helpers
# ============================================================================


def _extract_id_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"_(\d+)(?:\.[^.]*)+(?:\.zst)?$", filename)
    return None if match is None else int(match.group(1))


# Validate families separately: finding an outcome model never proves its propensity partner exists.
_ARTIFACT_FAMILIES = {
    "datasets_raw": ("synthetic_continous_dataset_{}.csv",),
    "models_raw": ("outcome_bnn_{}.pt", "propensity_bnn_{}.pt"),
    "metadata_raw": ("dgp_{}.json",),
    "datasets": ("synthetic_continous_dataset_{}.csv",),
    "models": ("outcome_bnn_{}.pt", "propensity_bnn_{}.pt"),
    "metadata": ("dgp_{}.json",),
    "scalers": ("scaler_{}.json",),
    "queries": ("queries_{}.csv",),
    "frontiers_raw": ("frontier_points_{}.csv",),
}


def _discover_family_ids(run_dir: str, subdir: str, pattern: str) -> set[int]:
    prefix, suffix = pattern.split("{}")
    regex = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"(?:\.zst)?$")
    return {
        int(match.group(1))
        for path in Path(run_dir, subdir).iterdir()
        if path.is_file() and (match := regex.fullmatch(path.name)) is not None
    }


def _validate_artifact_ids(run_dir: str, step: int, dataset_ids: Optional[List[int]]) -> List[int]:
    families = [
        (subdir, pattern, _discover_family_ids(run_dir, subdir, pattern))
        for subdir in _UPSTREAM_SUBDIRS[step]
        for pattern in _ARTIFACT_FAMILIES.get(subdir, ())
    ]
    source_ids = _discover_family_ids(run_dir, "metadata_raw", "dgp_{}.json")
    selected = source_ids if dataset_ids is None else set(dataset_ids)
    assert selected, "A fork must select at least one dataset ID"
    for subdir, pattern, available in families:
        assert selected <= available, (
            f"Incomplete fork input {subdir}/{pattern}: missing IDs {sorted(selected - available)}"
        )
        if dataset_ids is None:
            assert available == selected, (
                f"Fork input ID sets differ: metadata_raw vs {subdir}/{pattern}; "
                "select an explicit complete subset instead of silently dropping IDs"
            )
        for dataset_id in selected:
            path = Path(run_dir, subdir, pattern.format(dataset_id))
            assert path.is_file(), (
                f"Required fork input is compressed or missing: {path}. "
                "Restore its .zst file before forking; consumers require uncompressed artifacts."
            )
    if step == 4:
        assert Path(run_dir, "metadata_frontier", "frontier_run_config.json").is_file(), (
            "Missing Step-3 resolved configuration: metadata_frontier/frontier_run_config.json"
        )
    return sorted(selected)


def _selected_files(source_subdir: str, is_id_bearing: bool, dataset_ids: Optional[List[int]]) -> list[Path]:
    ids_set = set(dataset_ids) if is_id_bearing and dataset_ids is not None else None
    selected = []
    for path in sorted(Path(source_subdir).rglob("*")):
        assert not (path.is_symlink() and path.is_dir()), f"Directory symlink requires explicit restoration: {path}"
        if not path.is_file():
            continue
        relative = path.relative_to(source_subdir)
        # Profiling describes whole solves/batches, not individual DGPs; preserve it whole.
        run_level_profile = Path(source_subdir).name == "diagnostics_frontier" and relative.parts[0] == "profiling"
        if ids_set is not None and not run_level_profile:
            file_id = _extract_id_from_filename(path.name)
            assert file_id is not None, (
                f"Unclassified file in ID-bearing artifact tree: {path}; "
                "declare a run-level diagnostic policy before copying it"
            )
            if file_id not in ids_set:
                continue
        selected.append(path)
    return selected


def _hardlink_file(src: str, dst: str) -> None:
    """Create a hardlink dst -> src. dst must not already exist."""
    os.link(src, dst)


def _copy_file(src: str, dst: str) -> None:
    """Copy src to dst, preserving metadata."""
    shutil.copy2(src, dst)


def _link_subdir(
    source_subdir: str,
    dest_subdir: str,
    is_id_bearing: bool,
    dataset_ids: Optional[List[int]],
    use_copy: bool,
    dry_run: bool,
) -> int:
    """
    Link (or copy) all relevant files from source_subdir into dest_subdir.

    For ID-bearing dirs with dataset_ids set, only files whose embedded id
    is in dataset_ids are linked. Run-level dirs and unfiltered calls link
    everything.

    Returns the number of files linked/copied.
    """
    if not os.path.isdir(source_subdir):
        return 0

    count = 0
    for src_path in _selected_files(source_subdir, is_id_bearing, dataset_ids):
        relative = src_path.relative_to(source_subdir)
        dst_path = Path(dest_subdir) / relative
        if dry_run:
            action = "copy" if use_copy else "hardlink"
            print(f"  [dry_run] would {action}: {src_path} -> {dst_path}")
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if use_copy:
                _copy_file(str(src_path), str(dst_path))
            else:
                _hardlink_file(str(src_path), str(dst_path))
        count += 1

    return count


# ============================================================================
# Public API
# ============================================================================


def fork_upstream(
    source_run_dir: str,
    dest_run_dir: str,
    step: int,
    dataset_ids: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Fork a run_dir for a given pipeline step.

    Creates dest_run_dir (fails if it already exists) and hardlinks all
    upstream subdirs from source_run_dir into it. When source and dest are
    on different devices, falls back to shutil.copy2 for all files.

    Parameters
    ----------
    source_run_dir:
        Existing run directory to fork from.
    dest_run_dir:
        New run directory to create. Must not already exist.
    step:
        Pipeline step being forked into (2, 3, or 4). Controls which subdirs
        are treated as upstream (see fork table).
    dataset_ids:
        Optional list of dataset ids to filter ID-bearing dirs. When None,
        all files are linked. Run-level dirs are always fully linked.
    dry_run:
        When True, print what would be linked without creating any files or
        directories.

    Returns
    -------
    Dict mapping subdir name -> number of files linked/copied.
    """
    assert step in (2, 3, 4), f"step must be 2, 3, or 4; got {step}"

    source_run_dir = os.path.realpath(os.path.abspath(source_run_dir))
    dest_run_dir = os.path.realpath(os.path.abspath(dest_run_dir))

    assert os.path.isdir(source_run_dir), (
        f"source_run_dir does not exist or is not a directory: {source_run_dir}"
    )
    assert not Path(dest_run_dir).is_relative_to(source_run_dir), (
        "Destination must lie outside the source run; a nested fork would inherit itself"
    )
    missing_required_subdirs = [
        subdir_name
        for subdir_name in _UPSTREAM_SUBDIRS[step]
        if subdir_name not in _OPTIONAL_SUBDIRS
        and not os.path.isdir(os.path.join(source_run_dir, subdir_name))
    ]
    assert not missing_required_subdirs, (
        f"Source run is missing required upstream directories for Step {step}: "
        f"{missing_required_subdirs}"
    )

    assert not os.path.exists(dest_run_dir), f"Destination must be fresh: {dest_run_dir}"
    if dataset_ids is not None:
        dataset_ids = sorted({int(value) for value in dataset_ids})
        assert dataset_ids, "dataset_ids must be None or a non-empty list"
    effective_dataset_ids = _validate_artifact_ids(source_run_dir, step, dataset_ids)
    # Validate before creating a destination, including dry runs. Legacy missing histories
    # are visibly incomplete; recorded failed/running/dry-run attempts are not completion.
    validate_fork_provenance(
        source_run_dir, step, effective_dataset_ids, explicit_subset=dataset_ids is not None
    )

    # Classify nested files before creating anything; no partial destination on a
    # rejected subset policy (e.g. an unknown run-level diagnostic outside profiling/).
    for subdir in _UPSTREAM_SUBDIRS[step]:
        _selected_files(os.path.join(source_run_dir, subdir), subdir in _ID_BEARING_SUBDIRS, dataset_ids)

    if not dry_run:
        os.makedirs(dest_run_dir, exist_ok=False)
        print(f"[fork_run_dir] Created dest_run_dir: {dest_run_dir}")
    else:
        print(f"[fork_run_dir] [dry_run] Would create dest_run_dir: {dest_run_dir}")

    # Upfront device check — compare the device of source and dest.
    # dest_run_dir was just created above, so stat it directly.
    source_dev = os.stat(source_run_dir).st_dev
    device_path = Path(dest_run_dir)
    while not device_path.exists():
        device_path = device_path.parent
    dest_dev = device_path.stat().st_dev
    use_copy = source_dev != dest_dev
    if use_copy:
        print(
            f"[fork_run_dir] WARNING: source and dest are on different devices "
            f"(source_dev={source_dev}, dest_dev={dest_dev}). "
            "Falling back to shutil.copy2 for all files."
        )
    else:
        print(f"[fork_run_dir] Same device — using hardlinks.")

    subdirs_to_link = _UPSTREAM_SUBDIRS[step]

    print(f"[fork_run_dir] Step {step}: linking upstream artifacts for {len(effective_dataset_ids)} IDs")

    counts: Dict[str, int] = {}
    for subdir_name in subdirs_to_link:
        source_subdir = os.path.join(source_run_dir, subdir_name)

        if not os.path.isdir(source_subdir):
            assert subdir_name in _OPTIONAL_SUBDIRS, (
                f"Required upstream directory disappeared during fork: {source_subdir}"
            )
            print(f"[fork_run_dir] Skipping optional subdir (not present): {subdir_name}/")
            counts[subdir_name] = 0
            continue

        dest_subdir = os.path.join(dest_run_dir, subdir_name)
        is_id_bearing = subdir_name in _ID_BEARING_SUBDIRS

        n = _link_subdir(
            source_subdir=source_subdir,
            dest_subdir=dest_subdir,
            is_id_bearing=is_id_bearing,
            dataset_ids=dataset_ids,
            use_copy=use_copy,
            dry_run=dry_run,
        )
        counts[subdir_name] = n
        action = "would link" if dry_run else ("copied" if use_copy else "hardlinked")
        print(f"[fork_run_dir]   {subdir_name}/: {action} {n} file(s)")

    total_linked = sum(counts.values())
    if not dry_run:
        identity = initialize_fork_run_manifest(
            source_run_dir=source_run_dir,
            dest_run_dir=dest_run_dir,
            step=step,
            dataset_ids=dataset_ids,
            effective_dataset_ids=effective_dataset_ids,
            artifact_copy_counts=counts,
        )
        print(
            "[fork_run_dir] Initialized run manifest: "
            f"run_id={identity['run_id']}, "
            f"upstream_provenance_complete={identity['upstream_provenance_complete']}"
        )
    print(
        f"[fork_run_dir] Done. Total files {'would be ' if dry_run else ''}"
        f"{'copied' if use_copy else 'hardlinked'}: {total_linked}"
    )
    return counts


# ============================================================================
# CLI
# ============================================================================


@dataclass
class ForkConfig:
    source_run_dir: str
    dest_run_dir: str
    step: int
    dataset_ids: Optional[List[int]]
    dry_run: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fork a pipeline run_dir via hardlinks (or copies when cross-device). "
            "Creates dest_run_dir and links all upstream subdirs from source_run_dir."
        )
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        required=True,
        help="Existing run directory to fork from.",
    )
    parser.add_argument(
        "--dest_run_dir",
        type=str,
        required=True,
        help="New run directory to create. Must not already exist.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        choices=[2, 3, 4],
        help="Pipeline step being forked into (controls which subdirs are upstream).",
    )
    parser.add_argument(
        "--dataset_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of dataset ids to link from ID-bearing dirs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be linked without creating any files or directories.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ForkConfig:
    return ForkConfig(
        source_run_dir=args.source_run_dir,
        dest_run_dir=args.dest_run_dir,
        step=args.step,
        dataset_ids=args.dataset_ids,
        dry_run=args.dry_run,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    fork_upstream(
        source_run_dir=config.source_run_dir,
        dest_run_dir=config.dest_run_dir,
        step=config.step,
        dataset_ids=config.dataset_ids,
        dry_run=config.dry_run,
    )


if __name__ == "__main__":
    main()
