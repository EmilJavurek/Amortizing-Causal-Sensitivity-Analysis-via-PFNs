"""Data-local provenance records for standard-setting pipeline runs.

The manifest is deliberately append-only at the attempt level.  Each pipeline
invocation receives its own directory under ``run_manifest/steps/`` so failed,
resumed, overwritten, and repeated attempts remain visible.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
MANIFEST_DIRNAME = "run_manifest"
STEPS_DIRNAME = "steps"

ENV_PIPELINE_STEP = "PI_FM_PIPELINE_STEP"
ENV_CONFIG_PATH = "PI_FM_CONFIG_PATH"
ENV_CONFIG_BYTES = "PI_FM_CONFIG_BYTES_BASE64"
ENV_LAUNCHER_ARGV = "PI_FM_LAUNCHER_ARGV_JSON"
ENV_TARGET_ARGV = "PI_FM_TARGET_ARGV_JSON"
ENV_GIT_PROVENANCE = "PI_FM_GIT_PROVENANCE_JSON"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_MANIFEST_STEPS = {
    2: ("01_generate",),
    3: ("01_generate", "02_normalize"),
    4: ("01_generate", "02_normalize", "03_frontier"),
}
_FORK_STEP_NAMES = {2: "02_normalize", 3: "03_frontier", 4: "04_postprocess"}
_FINAL_STATUSES = {"completed", "completed_dry_run", "failed", "interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        assert math.isfinite(value), f"Manifest floats must be finite: {value}"
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    assert False, f"Manifest value is not JSON-serializable: {type(value).__name__}: {value!r}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _manifest_root(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve() / MANIFEST_DIRNAME


def _new_run_identity(
    *,
    creation_kind: str,
    run_created_at_utc: Optional[str],
    source_run_id: Optional[str],
    source_run_dir: Optional[str],
    fork_for_step: Optional[str],
    dataset_ids_filter: Optional[Sequence[int]],
    effective_dataset_ids: Optional[Sequence[int]],
    upstream_provenance_complete: bool,
    missing_upstream_steps: Sequence[str],
    incomplete_reason: Optional[str],
    fork_artifact_copy_counts: Optional[Mapping[str, int]],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "run_created_at_utc": run_created_at_utc,
        "manifest_initialized_at_utc": _utc_now(),
        "creation": {
            "kind": creation_kind,
            "source_run_id": source_run_id,
            "source_run_dir": source_run_dir,
            "fork_for_step": fork_for_step,
            "dataset_ids_filter": (
                None
                if dataset_ids_filter is None
                else sorted({int(value) for value in dataset_ids_filter})
            ),
            "effective_dataset_ids": (
                None
                if effective_dataset_ids is None
                else sorted({int(value) for value in effective_dataset_ids})
            ),
        },
        "upstream_provenance_complete": upstream_provenance_complete,
        "missing_upstream_steps": list(missing_upstream_steps),
        "incomplete_reason": incomplete_reason,
        "fork_artifact_copy_counts": (
            None if fork_artifact_copy_counts is None else dict(fork_artifact_copy_counts)
        ),
    }


def initialize_fresh_run_manifest(run_dir: str | Path) -> Dict[str, Any]:
    """Initialize provenance for a newly created Step-1 run directory."""
    run_dir = Path(run_dir).resolve()
    assert run_dir.is_dir(), f"Run directory must exist before manifest initialization: {run_dir}"
    root = _manifest_root(run_dir)
    assert not root.exists(), f"Run manifest already exists: {root}"
    identity = _new_run_identity(
        creation_kind="fresh_generation",
        run_created_at_utc=_utc_now(),
        source_run_id=None,
        source_run_dir=None,
        fork_for_step=None,
        dataset_ids_filter=None,
        effective_dataset_ids=None,
        upstream_provenance_complete=True,
        missing_upstream_steps=(),
        incomplete_reason=None,
        fork_artifact_copy_counts=None,
    )
    (root / STEPS_DIRNAME).mkdir(parents=True)
    _write_json_atomic(root / "run.json", identity)
    return identity


def load_run_identity(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    root = _manifest_root(run_dir)
    if not root.exists():
        return None
    path = root / "run.json"
    assert path.is_file(), f"Manifest directory exists without run.json: {root}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION, (
        f"Unsupported run-manifest schema in {path}: {payload.get('schema_version')}"
    )
    return payload


def ensure_run_manifest(run_dir: str | Path) -> Dict[str, Any]:
    """Return the run identity, visibly adopting a pre-manifest legacy run if needed."""
    run_dir = Path(run_dir).resolve()
    assert run_dir.is_dir(), f"Run directory does not exist: {run_dir}"
    root = _manifest_root(run_dir)
    identity = load_run_identity(run_dir)
    if identity is not None:
        assert (root / STEPS_DIRNAME).is_dir(), f"Manifest steps directory is missing: {root}"
        return identity

    assert not root.exists(), (
        f"Manifest directory exists without a valid run.json; refusing to infer state: {root}"
    )
    identity = _new_run_identity(
        creation_kind="legacy_adoption",
        run_created_at_utc=None,
        source_run_id=None,
        source_run_dir=None,
        fork_for_step=None,
        dataset_ids_filter=None,
        effective_dataset_ids=None,
        upstream_provenance_complete=False,
        missing_upstream_steps=(),
        incomplete_reason=(
            "The run predates manifest recording; its creation time and upstream "
            "step provenance are unknown."
        ),
        fork_artifact_copy_counts=None,
    )
    (root / STEPS_DIRNAME).mkdir(parents=True)
    _write_json_atomic(root / "run.json", identity)
    return identity


def _attempt_scope(
    attempt: Mapping[str, Any],
    step: str,
    upstream_ids: set[int],
    source_ids: set[int],
) -> set[int]:
    completion = attempt.get("completion") or {}
    arguments = attempt["invocation"]["resolved_arguments"]
    explicit = completion.get("dataset_ids", arguments.get("dataset_ids"))
    if explicit is not None:
        return {int(value) for value in explicit}
    if step == "01_generate":
        return set(range(1, int(arguments["num_datasets"]) + 1))
    # Legacy full-cohort attempts recorded counts, not IDs. Inherited history
    # keeps its original scope even after files are narrowed by a subset fork.
    # A step run after forking instead owns that fork's narrower input cohort.
    requested = completion.get("requested_datasets")
    if requested is None and attempt["status"] != "completed":
        return upstream_ids
    if requested == len(upstream_ids):
        return upstream_ids
    assert requested == len(source_ids), (
        f"Cannot resolve inherited legacy count-only {step} cohort from its recorded scope. "
        "Fork directly from the producer with its original cohort; do not infer IDs from counts."
    )
    return source_ids


def _completed_ids(attempt: Mapping[str, Any], scope: set[int]) -> set[int]:
    completion = attempt.get("completion") or {}
    failed = {int(value) for value in completion.get("failed_dataset_ids", ())}
    assert failed <= scope, "Recorded failed IDs are outside the attempted cohort"
    completed = scope - failed
    count = completion.get("completed_datasets_total", completion.get("completed_datasets"))
    assert count == len(completed), (
        f"Completion count {count!r} does not establish coverage of {len(completed)} IDs"
    )
    assert completion.get("failed_datasets", len(failed)) == len(failed), (
        "Failed count lacks matching failed-ID evidence"
    )
    assert completion.get("failed_batches", 0) == 0, "Frontier batches failed"
    return completed


def validate_fork_provenance(
    source_run_dir: str | Path,
    step: int,
    effective_dataset_ids: Sequence[int],
    *,
    explicit_subset: bool,
) -> list[str]:
    """Validate recorded completion; return steps whose legacy history is absent.

    The latest attempt wins. A same-target failed/running/dry-run attempt cannot
    resurrect earlier completion. An explicit subset can inherit recorded successes
    from partial generation, or an untouched completed cohort during an extension.
    """
    selected = {int(value) for value in effective_dataset_ids}
    assert selected, "A fork must select at least one dataset ID"
    source_ids = {
        int(path.stem.removeprefix("dgp_"))
        for path in Path(source_run_dir, "metadata_raw").glob("dgp_*.json")
    }
    source_identity = load_run_identity(source_run_dir)
    fork_cohort = None if source_identity is None else source_identity["creation"].get("effective_dataset_ids")
    if fork_cohort is not None:
        fork_cohort = {int(value) for value in fork_cohort}
        assert source_ids == fork_cohort, "Source raw IDs differ from its recorded fork cohort"
    # A previously selected cohort remains explicit when re-forked without a
    # redundant filter; this does not authorize IDs outside that recorded cohort.
    explicit_cohort = explicit_subset or (fork_cohort is not None and selected <= fork_cohort)
    upstream_ids = source_ids
    missing = []
    for step_name in _UPSTREAM_MANIFEST_STEPS[step]:
        attempts = load_step_attempts(source_run_dir, step_name)
        if not attempts:
            missing.append(step_name)
            continue
        latest = attempts[-1]
        scope = _attempt_scope(latest, step_name, upstream_ids, source_ids)
        status = latest["status"]
        if status == "completed":
            completed = _completed_ids(latest, scope)
        elif explicit_cohort and step_name == "01_generate" and status == "failed":
            completed = _completed_ids(latest, scope)
        elif explicit_cohort and step_name == "03_frontier" and status in {"failed", "interrupted", "running"}:
            # --extend only writes new IDs. Never reuse completion across a resumed
            # or overwritten same-target invocation, which may have changed its files.
            previous = next((a for a in reversed(attempts[:-1]) if a["status"] == "completed"), None)
            assert previous is not None, "An incomplete frontier has no completed cohort to inherit"
            previous_scope = _attempt_scope(previous, step_name, upstream_ids, source_ids)
            for attempt in attempts[attempts.index(previous) + 1:]:
                arguments = attempt["invocation"]["resolved_arguments"]
                current_scope = _attempt_scope(attempt, step_name, upstream_ids, source_ids)
                assert arguments.get("extend") and previous_scope < current_scope, (
                    "Incomplete latest frontier attempt may have modified the selected cohort; resume it first"
                )
            completed = _completed_ids(previous, previous_scope)
        else:
            assert False, f"Cannot fork {step_name}: latest attempt is {status!r}; finish/resume the producer first"
        assert selected <= completed, (
            f"{step_name} has no completion evidence for requested IDs {sorted(selected - completed)}"
        )
        if step_name == "01_generate" and not explicit_subset and fork_cohort is None:
            assert selected == completed, (
                "Source raw IDs differ from the recorded generation cohort; select an explicit subset"
            )
        upstream_ids = completed
    return missing


def initialize_fork_run_manifest(
    *,
    source_run_dir: str | Path,
    dest_run_dir: str | Path,
    step: int,
    dataset_ids: Optional[Sequence[int]],
    effective_dataset_ids: Sequence[int],
    artifact_copy_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """Create a new run identity and copy only provenance upstream of ``step``."""
    assert step in _UPSTREAM_MANIFEST_STEPS, f"Unsupported fork step: {step}"
    source_run_dir = Path(source_run_dir).resolve()
    dest_run_dir = Path(dest_run_dir).resolve()
    assert source_run_dir.is_dir(), source_run_dir
    assert dest_run_dir.is_dir(), dest_run_dir
    canonical_effective_ids = sorted({int(value) for value in effective_dataset_ids})
    assert canonical_effective_ids, "A fork must contain at least one effective dataset ID"
    if dataset_ids is not None:
        assert sorted({int(value) for value in dataset_ids}) == canonical_effective_ids, (
            "A requested fork filter must match the effective copied dataset IDs"
        )

    missing_steps = validate_fork_provenance(
        source_run_dir, step, canonical_effective_ids, explicit_subset=dataset_ids is not None
    )
    dest_root = _manifest_root(dest_run_dir)
    assert not dest_root.exists(), f"Destination manifest already exists: {dest_root}"
    dest_steps_root = dest_root / STEPS_DIRNAME
    dest_steps_root.mkdir(parents=True)

    source_identity = load_run_identity(source_run_dir)
    source_steps_root = _manifest_root(source_run_dir) / STEPS_DIRNAME
    required_steps = _UPSTREAM_MANIFEST_STEPS[step]

    for step_name in required_steps:
        source_step = source_steps_root / step_name
        if not source_step.is_dir():
            continue
        shutil.copytree(
            source_step,
            dest_steps_root / step_name,
            copy_function=shutil.copy2,
        )

    upstream_complete = (
        source_identity is not None
        and source_identity["upstream_provenance_complete"]
        and not missing_steps
    )
    identity = _new_run_identity(
        creation_kind="fork",
        run_created_at_utc=_utc_now(),
        source_run_id=None if source_identity is None else source_identity["run_id"],
        source_run_dir=str(source_run_dir),
        fork_for_step=_FORK_STEP_NAMES[step],
        dataset_ids_filter=dataset_ids,
        effective_dataset_ids=canonical_effective_ids,
        upstream_provenance_complete=upstream_complete,
        missing_upstream_steps=missing_steps,
        incomplete_reason=(
            None
            if upstream_complete
            else "The source run lacks complete manifest history for the inherited steps."
        ),
        fork_artifact_copy_counts=artifact_copy_counts,
    )
    _write_json_atomic(dest_root / "run.json", identity)
    return identity


def _git_provenance() -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        text=True,
    ).strip()
    dirty_output = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=_PROJECT_ROOT,
        text=True,
    )
    return {"commit": commit, "dirty": bool(dirty_output.strip())}


def _environment_argv(name: str) -> Optional[list[str]]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    parsed = json.loads(raw)
    assert isinstance(parsed, list) and all(isinstance(item, str) for item in parsed), (
        f"{name} must contain a JSON string array"
    )
    return parsed


def _environment_mapping(name: str) -> Optional[Dict[str, Any]]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), f"{name} must contain a JSON object"
    return parsed


def _allocate_attempt_dir(run_dir: Path, step: str) -> tuple[int, Path]:
    step_root = _manifest_root(run_dir) / STEPS_DIRNAME / step
    step_root.mkdir(parents=True, exist_ok=True)
    for attempt_number in range(1, 1_000_000):
        attempt_dir = step_root / f"attempt_{attempt_number:04d}"
        try:
            attempt_dir.mkdir()
        except FileExistsError:
            continue
        return attempt_number, attempt_dir
    assert False, f"Could not allocate an attempt directory under {step_root}"


@dataclass
class StepAttempt:
    path: Path
    record: Dict[str, Any]
    started_monotonic: float
    successful_status: str = "completed"

    def set_completion(
        self,
        values: Mapping[str, Any],
        *,
        status: str = "completed",
    ) -> None:
        assert status in {"completed", "completed_dry_run"}, status
        self.record["completion"] = _json_ready(values)
        self.successful_status = status

    def finalize(self, status: str, error: Optional[BaseException]) -> None:
        assert status in _FINAL_STATUSES, status
        self.record["status"] = status
        self.record["finished_at_utc"] = _utc_now()
        self.record["duration_seconds"] = time.monotonic() - self.started_monotonic
        self.record["error"] = (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        )
        _write_json_atomic(self.path / "attempt.json", self.record)


@contextmanager
def record_pipeline_step(
    *,
    run_dir: str | Path,
    step: str,
    implementation: str,
    resolved_arguments: Mapping[str, Any],
    seeds: Mapping[str, Any],
) -> Iterator[StepAttempt]:
    """Record one pipeline-step attempt and finalize its status atomically."""
    run_dir = Path(run_dir).resolve()
    resolved_arguments_ready = _json_ready(resolved_arguments)
    seeds_ready = _json_ready(seeds)
    git_provenance = _environment_mapping(ENV_GIT_PROVENANCE)
    if git_provenance is None:
        git_provenance = _git_provenance()
    assert set(git_provenance) == {"commit", "dirty"}, git_provenance
    assert isinstance(git_provenance["commit"], str) and git_provenance["commit"]
    assert isinstance(git_provenance["dirty"], bool)

    identity = ensure_run_manifest(run_dir)
    launcher_step = os.environ.get(ENV_PIPELINE_STEP)
    assert launcher_step is None or launcher_step == step, (
        f"Launcher step {launcher_step!r} does not match recorder step {step!r}"
    )

    config_path_raw = os.environ.get(ENV_CONFIG_PATH)
    config_bytes_base64 = os.environ.get(ENV_CONFIG_BYTES)
    config_path = None if config_path_raw is None else Path(config_path_raw).resolve()
    config_bytes = None
    config_sha256 = None
    config_copy = None
    if config_path is not None:
        assert config_bytes_base64 is not None, (
            f"{ENV_CONFIG_BYTES} is required when {ENV_CONFIG_PATH} is set"
        )
        config_bytes = base64.b64decode(config_bytes_base64, validate=True)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        config_copy = "input.yaml"
    else:
        assert config_bytes_base64 is None, (
            f"{ENV_CONFIG_BYTES} was set without {ENV_CONFIG_PATH}"
        )

    launcher_argv = _environment_argv(ENV_LAUNCHER_ARGV)
    target_argv = _environment_argv(ENV_TARGET_ARGV)
    if target_argv is None:
        target_argv = [sys.executable, *sys.argv]

    started_monotonic = time.monotonic()
    attempt_number, attempt_dir = _allocate_attempt_dir(run_dir, step)
    record = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": str(uuid.uuid4()),
        "attempt_number": attempt_number,
        "step": step,
        "implementation": implementation,
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "git": git_provenance,
        "invocation": {
            "working_directory": str(Path.cwd().resolve()),
            "launcher_argv": launcher_argv,
            "target_argv": target_argv,
            "original_config_path": None if config_path is None else str(config_path),
            "config_copy": config_copy,
            "config_sha256": config_sha256,
            "resolved_arguments": resolved_arguments_ready,
            "provenance_complete": config_path is not None and launcher_argv is not None,
        },
        "seeds": seeds_ready,
        "lineage": {
            "run_id": identity["run_id"],
            "run_dir": str(run_dir),
            "source_run_dir": _json_ready(resolved_arguments.get("source_run_dir")),
        },
        "completion": {},
        "error": None,
    }
    attempt = StepAttempt(
        path=attempt_dir,
        record=record,
        started_monotonic=started_monotonic,
    )
    _write_json_atomic(attempt_dir / "attempt.json", record)

    try:
        if config_bytes is not None:
            _write_bytes_atomic(attempt_dir / "input.yaml", config_bytes)
        yield attempt
    except KeyboardInterrupt as error:
        attempt.finalize("interrupted", error)
        raise
    except BaseException as error:
        attempt.finalize("failed", error)
        raise
    else:
        attempt.finalize(attempt.successful_status, None)


def load_step_attempts(run_dir: str | Path, step: str) -> list[Dict[str, Any]]:
    step_root = _manifest_root(run_dir) / STEPS_DIRNAME / step
    if not step_root.is_dir():
        return []
    attempts = []
    for attempt_dir in sorted(step_root.glob("attempt_*")):
        path = attempt_dir / "attempt.json"
        assert path.is_file(), f"Manifest attempt is missing attempt.json: {attempt_dir}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION, path
        payload["attempt_dir"] = str(attempt_dir)
        attempts.append(payload)
    attempts.sort(key=lambda payload: payload["attempt_number"])
    return attempts


def latest_step_attempt(run_dir: str | Path, step: str) -> Optional[Dict[str, Any]]:
    attempts = load_step_attempts(run_dir, step)
    return attempts[-1] if attempts else None


def latest_completed_step_attempt(
    run_dir: str | Path,
    step: str,
) -> Optional[Dict[str, Any]]:
    completed = [
        attempt
        for attempt in load_step_attempts(run_dir, step)
        if attempt["status"] == "completed"
    ]
    return completed[-1] if completed else None
