"""
Subsystem B: batched Pareto frontier construction for partial identification.

Given frozen outcome BNNs (f_BNN) from Subsystem A, sweeps scalarized
Lagrangian lambda values to produce (theta*, Gamma*) training pairs
for the foundation model (Subsystem C).

Script layout:
    0. Configs & small types
    1. Flow + physics       differentiable forward pass, no state management
    2. Solver               single-lambda optimization over a DGP batch, no I/O
    3. Pipeline             DGP batching, lambda sweep, orchestration
    4. I/O                  serialization, deserialization, CLI

Dependency direction: 1 <- 2 <- 3 -> 4. Sections 1 and 4 are independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from Setting_standard.causaldataset import CausalFrontierDataset, resolve_code_relative_path
from Setting_standard.gen_standard_syn import NormalizedOutcomeMLPWrapper, OutcomeMLP


# ============================================================================
# Section 0: Configs, small types, utilities
# ============================================================================


@dataclass
class FlowConfig:
    """Architecture parameters for the 1D spline flow."""
    num_bins: int = 16
    tail_bound: float = 4.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3


@dataclass
class SolverConfig:
    """Hyperparameters for a single scalarized optimization."""
    num_mc_samples_train: int = 32
    num_mc_samples_eval: int = 256
    max_steps: int = 500
    learning_rate: float = 1e-3
    lr_lambda_schedule: str = "none"  # "none" | "sqrt"
    lr_lambda_ref: float = 0.5
    lr_lambda_min_mult: float = 0.25
    max_steps_lambda_schedule: str = "none"  # "none" | "inverse_sqrt_lr"
    max_steps_lambda_max_mult: float = 2.0
    loss_reduction: str = "batch_mean"  # "batch_mean" | "per_dgp_sum"

    # Optional adaptive stopping. Disabled by default because fixed-step
    # runs are easier to diagnose and compare.
    early_stop: bool = False
    early_stop_min_steps: int = 75
    early_stop_check_every: int = 25
    early_stop_patience: int = 3
    early_stop_abs_tol: float = 1e-4
    early_stop_rel_tol: float = 1e-4


def validate_lr_lambda_config(cfg: SolverConfig) -> None:
    """Validate lambda-dependent LR schedule settings."""
    if cfg.lr_lambda_schedule not in {"none", "sqrt"}:
        raise ValueError(f"Unknown lr_lambda_schedule={cfg.lr_lambda_schedule!r}")
    if cfg.lr_lambda_ref <= 0:
        raise ValueError(f"lr_lambda_ref must be positive, got {cfg.lr_lambda_ref}")
    if not (0 < cfg.lr_lambda_min_mult <= 1):
        raise ValueError(
            f"lr_lambda_min_mult must be in (0, 1], got {cfg.lr_lambda_min_mult}"
        )


def lr_multiplier_for_lambda(lambda_value: float, cfg: SolverConfig) -> float:
    validate_lr_lambda_config(cfg)
    if cfg.lr_lambda_schedule == "none":
        return 1.0
    if cfg.lr_lambda_schedule == "sqrt":
        mult = math.sqrt(float(lambda_value) / float(cfg.lr_lambda_ref))
        return min(1.0, max(float(cfg.lr_lambda_min_mult), mult))
    raise ValueError(f"Unknown lr_lambda_schedule={cfg.lr_lambda_schedule!r}")


def validate_max_steps_lambda_config(cfg: SolverConfig) -> None:
    """Validate lambda-dependent max-step schedule settings."""
    if cfg.max_steps_lambda_schedule not in {"none", "inverse_sqrt_lr"}:
        raise ValueError(
            f"Unknown max_steps_lambda_schedule={cfg.max_steps_lambda_schedule!r}"
        )
    if cfg.max_steps_lambda_max_mult < 1.0:
        raise ValueError(
            "max_steps_lambda_max_mult must be at least 1.0, "
            f"got {cfg.max_steps_lambda_max_mult}"
        )


def max_steps_for_lambda(lambda_value: float, cfg: SolverConfig, lr_mult: Optional[float] = None) -> Tuple[int, float]:
    validate_max_steps_lambda_config(cfg)
    if cfg.max_steps_lambda_schedule == "none":
        return int(cfg.max_steps), 1.0
    if cfg.max_steps_lambda_schedule == "inverse_sqrt_lr":
        effective_lr_mult = lr_multiplier_for_lambda(lambda_value, cfg) if lr_mult is None else float(lr_mult)
        step_mult = 1.0 / math.sqrt(effective_lr_mult)
        step_mult = min(float(cfg.max_steps_lambda_max_mult), max(1.0, step_mult))
        return int(math.ceil(int(cfg.max_steps) * step_mult)), float(step_mult)
    raise ValueError(
        f"Unknown max_steps_lambda_schedule={cfg.max_steps_lambda_schedule!r}"
    )


@dataclass
class QueryConfig:
    """Controls for query point selection within each dataset."""
    num_query_points: int = 16
    sampling_mode: str = "ordered_proxy"       # "random" | "ordered_proxy" | "stratified_proxy"
    random_seed: int = 0
    treatment_arms: Tuple[int, ...] = (0, 1)


@dataclass
class LambdaGridConfig:
    """Controls for lambda grid construction."""
    lambda_min: float
    lambda_max: float
    lambda_n: int = 30
    lambda_spacing: str = "log"   # "log" | "linear"


@dataclass
class ProfilingConfig:
    """Optional diagnostic profiling controls. Disabled by default."""
    runtime_profile: bool = False
    runtime_profile_first_n_solves: int = 0
    runtime_profile_solve_indices: Optional[Tuple[int, ...]] = None
    runtime_profile_sync_cuda: bool = False
    runtime_profile_output_dir: Optional[str] = None

    torch_profile: bool = False
    torch_profile_dir: Optional[str] = None
    torch_profile_first_n_solves: int = 1
    torch_profile_wait: int = 5
    torch_profile_warmup: int = 5
    torch_profile_active: int = 10
    torch_profile_repeat: int = 1


@dataclass
class PipelineConfig:
    """
    Top-level config assembling all sub-configs plus pipeline-level settings.

    Functions in sections 1-2 never see this object. They receive the
    relevant sub-config or explicit arguments. Only section 3 (pipeline)
    and section 4 (I/O) use PipelineConfig directly.
    """
    root_dir: str
    output_dir: str
    dataset_ids: Optional[List[int]] = None
    sensitivity_model: str = "msm"
    estimand_type: str = "capo"
    device: str = "cpu"
    save_diagnostics: bool = False
    save_spline_params: bool = False
    copy_dataset_csvs: bool = False
    dgp_batch_size: int = 8
    warm_start_lambdas: bool = True

    flow: FlowConfig = field(default_factory=FlowConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    lambda_grid_cfg: LambdaGridConfig = field(default_factory=LambdaGridConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    use_bucket_forward: bool = True


@dataclass
class QuerySpec:
    """One query point for frontier construction. Lightweight, no metadata."""
    query_id: int
    source_row_index: int
    a_query: int
    x_query: torch.Tensor                            # shape [x_dim]
    pi_query: float


@dataclass
class SolveResult:
    """
    Return value of optimize_lambda_batch. Holds per-DGP, per-query arrays.

    theta_star and gamma_star are numpy arrays of shape [B, m], already
    detached from the computation graph and moved to CPU via the eval pass.
    spline_params_final is shape [B, m, d_sp], also detached and on CPU,
    and is used by the caller as the explicit warm-start for the next lambda.
    """
    theta_star: np.ndarray                           # [B, m]
    gamma_star: np.ndarray                           # [B, m]
    spline_params_final: np.ndarray                  # [B, m, d_sp]
    lambda_value: float
    bound_type: str
    num_steps: int
    stop_reason: str                                 # "max_steps" | "early_stop" | "nonfinite"
    runtime_seconds: float
    trace: List[Dict[str, float]]                    # per-step aggregates
    learning_rate_base: float
    learning_rate_multiplier: float
    learning_rate_effective: float
    lr_lambda_schedule: str
    max_steps_base: int
    max_steps_effective: int
    max_steps_multiplier: float
    max_steps_lambda_schedule: str


@dataclass
class OutputPaths:
    """Resolved directory paths for all output artifacts."""
    root: str
    datasets_dir: str
    queries_dir: str
    frontiers_dir: str
    metadata_dir: str
    diagnostics_dir: str
    spline_params_dir: str


class RuntimeSolveProfiler:
    """Aggregate scoped wall-clock timings for one lambda solve."""

    def __init__(
        self,
        solve_index: int,
        lambda_index: Optional[int],
        dataset_batch_ids: Sequence[int],
        bound_type: str,
        lambda_value: float,
        num_queries: int,
        train_mc_samples: int,
        sync_cuda: bool,
        device: str,
    ) -> None:
        self.solve_index = int(solve_index)
        self.lambda_index = lambda_index
        self.dataset_batch_ids = list(dataset_batch_ids)
        self.bound_type = bound_type
        self.lambda_value = float(lambda_value)
        self.num_queries = int(num_queries)
        self.train_mc_samples = int(train_mc_samples)
        self.sync_cuda = bool(sync_cuda and torch.cuda.is_available() and torch.device(device).type == "cuda")
        self.sections: Dict[str, Dict[str, float]] = {}

    def _sync_if_needed(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        self._sync_if_needed()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync_if_needed()
            elapsed = time.perf_counter() - t0
            stats = self.sections.setdefault(name, {"count": 0.0, "total_seconds": 0.0})
            stats["count"] += 1.0
            stats["total_seconds"] += elapsed

    def rows(self) -> List[Dict[str, object]]:
        total = self.sections.get("optimize_lambda_batch_total", {}).get("total_seconds", 0.0)
        if total <= 0.0:
            total = sum(stats["total_seconds"] for stats in self.sections.values())

        rows: List[Dict[str, object]] = []
        dataset_ids = " ".join(str(x) for x in self.dataset_batch_ids)
        for section in sorted(self.sections):
            stats = self.sections[section]
            count = int(stats["count"])
            total_seconds = float(stats["total_seconds"])
            rows.append({
                "solve_index": self.solve_index,
                "dataset_batch_ids": dataset_ids,
                "bound_type": self.bound_type,
                "lambda_index": self.lambda_index if self.lambda_index is not None else "",
                "lambda_value": self.lambda_value,
                "section": section,
                "count": count,
                "total_seconds": total_seconds,
                "mean_seconds": total_seconds / max(count, 1),
                "fraction_of_profiled_total": total_seconds / total if total > 0.0 else 0.0,
            })
        return rows


class ProfilingState:
    """Mutable run-level profiling counters and collected manual timing rows."""

    def __init__(self, config: ProfilingConfig, output_paths: OutputPaths) -> None:
        self.config = config
        self.solve_count = 0
        self.runtime_output_dir = resolve_code_relative_path(
            config.runtime_profile_output_dir
            if config.runtime_profile_output_dir is not None
            else os.path.join(output_paths.diagnostics_dir, "profiling")
        )
        self.torch_profile_dir = resolve_code_relative_path(
            config.torch_profile_dir
            if config.torch_profile_dir is not None
            else os.path.join(output_paths.diagnostics_dir, "profiling", "torch_profile")
        )
        self.runtime_rows: List[Dict[str, object]] = []
        self.runtime_solve_meta: Dict[int, Dict[str, object]] = {}
        self.run_start_time = time.perf_counter()

    def begin_solve(
        self,
        dataset_batch_ids: Sequence[int],
        bound_type: str,
        lambda_index: Optional[int],
        lambda_value: float,
        num_queries: int,
        train_mc_samples: int,
        device: str,
    ) -> Tuple[int, Optional[RuntimeSolveProfiler], bool]:
        solve_index = self.solve_count
        self.solve_count += 1

        runtime_limit = self.config.runtime_profile_first_n_solves
        runtime_indices = self.config.runtime_profile_solve_indices
        runtime_enabled = (
            self.config.runtime_profile
            and (
                (runtime_indices is not None and solve_index in runtime_indices)
                or (runtime_indices is None and (runtime_limit == 0 or solve_index < runtime_limit))
            )
        )
        runtime_profiler = None
        if runtime_enabled:
            runtime_profiler = RuntimeSolveProfiler(
                solve_index=solve_index,
                lambda_index=lambda_index,
                dataset_batch_ids=dataset_batch_ids,
                bound_type=bound_type,
                lambda_value=lambda_value,
                num_queries=num_queries,
                train_mc_samples=train_mc_samples,
                sync_cuda=self.config.runtime_profile_sync_cuda,
                device=device,
            )

        torch_enabled = (
            self.config.torch_profile
            and solve_index < self.config.torch_profile_first_n_solves
        )
        return solve_index, runtime_profiler, torch_enabled

    def finish_runtime_solve(self, profiler: Optional[RuntimeSolveProfiler]) -> None:
        if profiler is None:
            return
        rows = profiler.rows()
        self.runtime_rows.extend(rows)
        self.runtime_solve_meta[profiler.solve_index] = {
            "solve_index": profiler.solve_index,
            "lambda_index": profiler.lambda_index,
            "dataset_batch_ids": " ".join(str(x) for x in profiler.dataset_batch_ids),
            "bound_type": profiler.bound_type,
            "lambda_value": profiler.lambda_value,
            "num_queries": profiler.num_queries,
            "train_mc_samples": profiler.train_mc_samples,
        }
        top_rows = sorted(rows, key=lambda row: float(row["total_seconds"]), reverse=True)[:8]
        tqdm.write(
            f"[runtime-profile] solve={profiler.solve_index} "
            f"bound={profiler.bound_type} lambda_index={profiler.lambda_index} "
            f"lambda={profiler.lambda_value:.6g}"
        )
        for row in top_rows:
            tqdm.write(
                "  {section:30s} {total_seconds:10.4f}s "
                "{fraction_of_profiled_total:7.2%} count={count}".format(**row)
            )

    def write_runtime_outputs(self) -> None:
        if not self.config.runtime_profile:
            return
        os.makedirs(self.runtime_output_dir, exist_ok=True)
        csv_path = os.path.join(self.runtime_output_dir, "runtime_profile_summary.csv")
        json_path = os.path.join(self.runtime_output_dir, "runtime_profile_summary.json")
        throughput_csv_path = os.path.join(self.runtime_output_dir, "runtime_profile_throughput.csv")
        throughput_json_path = os.path.join(self.runtime_output_dir, "runtime_profile_throughput.json")
        columns = [
            "solve_index",
            "dataset_batch_ids",
            "bound_type",
            "lambda_index",
            "lambda_value",
            "section",
            "count",
            "total_seconds",
            "mean_seconds",
            "fraction_of_profiled_total",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.runtime_rows)
        payload = {
            "runtime_profile_output_dir": self.runtime_output_dir,
            "num_profiled_rows": len(self.runtime_rows),
            "rows": self.runtime_rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        throughput = self._throughput_summary()
        with open(throughput_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(throughput.keys()))
            writer.writeheader()
            writer.writerow(throughput)
        with open(throughput_json_path, "w", encoding="utf-8") as f:
            json.dump(throughput, f, indent=2)
        tqdm.write("[runtime-profile] throughput summary:")
        for key in (
            "total_wall_clock_seconds",
            "total_profiled_solve_seconds",
            "num_profiled_solves",
            "optimizer_steps_per_second",
            "query_mc_evals_per_second",
        ):
            tqdm.write(f"  {key}: {throughput[key]}")
        tqdm.write(f"[runtime-profile] wrote {csv_path}")
        tqdm.write(f"[runtime-profile] wrote {json_path}")
        tqdm.write(f"[runtime-profile] wrote {throughput_csv_path}")
        tqdm.write(f"[runtime-profile] wrote {throughput_json_path}")

    def _throughput_summary(self) -> Dict[str, object]:
        run_seconds = time.perf_counter() - self.run_start_time
        total_profiled_seconds = 0.0
        optimizer_steps = 0
        query_mc_evals = 0
        profiled_solve_indices = set()

        for row in self.runtime_rows:
            solve_index = int(row["solve_index"])
            if row["section"] == "optimize_lambda_batch_total":
                total_profiled_seconds += float(row["total_seconds"])
                profiled_solve_indices.add(solve_index)
            if row["section"] == "optimizer_step":
                optimizer_steps += int(row["count"])
            if row["section"] == "forward_train":
                meta = self.runtime_solve_meta[solve_index]
                query_mc_evals += (
                    int(row["count"])
                    * int(meta["num_queries"])
                    * int(meta["train_mc_samples"])
                )

        return {
            "total_wall_clock_seconds": run_seconds,
            "total_profiled_solve_seconds": total_profiled_seconds,
            "num_profiled_solves": len(profiled_solve_indices),
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_per_second": optimizer_steps / total_profiled_seconds
            if total_profiled_seconds > 0.0 else 0.0,
            "approx_query_mc_evaluations": query_mc_evals,
            "query_mc_evals_per_second": query_mc_evals / total_profiled_seconds
            if total_profiled_seconds > 0.0 else 0.0,
        }


def _profile_section(
    profiler: Optional[RuntimeSolveProfiler],
    section_name: str,
) -> Iterator[None]:
    if profiler is None:
        return nullcontext()
    return profiler.section(section_name)


def _inverse_softplus_target(target: float) -> float:
    return math.log(math.exp(target) - 1.0)


def standard_normal_logprob(u: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2.0 * math.pi) + u.pow(2))


def _compute_gradient_norm(parameters: Sequence[torch.Tensor]) -> float:
    norm_tensor = _compute_gradient_norm_tensor(parameters)
    return _gradient_norm_to_float(norm_tensor)


def _compute_gradient_norm_tensor(parameters: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    grads = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    if not grads:
        return None
    return torch.linalg.vector_norm(torch.cat(grads))


def _gradient_norm_to_float(norm_tensor: Optional[torch.Tensor]) -> float:
    if norm_tensor is None:
        return 0.0
    return float(norm_tensor.detach().cpu())


def _clip_spline_param_grad_norm_(spline_params: torch.Tensor, max_norm: float) -> None:
    """Clip each (DGP, query) spline gradient independently."""
    assert spline_params.dim() == 3
    if spline_params.grad is None:
        return
    grad = spline_params.grad
    grad_norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
    clip_coef = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
    grad.mul_(clip_coef)


def _all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _noop_log(_: str) -> None:
    return None


def _parse_int_csv(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse optional comma-separated integer CLI values."""
    if value is None:
        return None
    pieces = [piece.strip() for piece in value.split(",")]
    assert all(piece for piece in pieces), f"Invalid comma-separated integer list: {value!r}"
    parsed = tuple(int(piece) for piece in pieces)
    assert all(index >= 0 for index in parsed), f"Solve indices must be non-negative: {value!r}"
    return parsed


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _signed_theta(theta: np.ndarray, bound_type: str) -> np.ndarray:
    """Return theta in maximization orientation: upper=theta, lower=-theta."""
    assert bound_type in ("upper", "lower")
    if bound_type == "upper":
        return theta
    return -theta


def _objective_from_theta_gamma(
    theta: np.ndarray,
    gamma: np.ndarray,
    lambda_value: float,
    bound_type: str,
) -> np.ndarray:
    """Evaluate the scalarized objective from final-eval theta/gamma arrays."""
    return _signed_theta(theta, bound_type) - lambda_value * gamma


def draw_base_u(
    batch_size: int,
    num_queries: int,
    num_mc_samples: int,
    device: str,
    runtime_profiler: Optional[RuntimeSolveProfiler] = None,
) -> torch.Tensor:
    """
    Draw base latent samples [B, m, S] as i.i.d. standard normals.

    QMC was tried in v4 (precomputed Sobol cache, then Cranley-Patterson
    rotation) and dropped: the variance benefit was not materializing in
    practice, the precompute path added plumbing complexity, and any change
    in the per-step sample sequence produced trajectory-level disagreements
    with v3 that confused validation. Adam with the current k handles
    per-step gradient noise without QMC.
    """
    assert batch_size >= 1 and num_queries >= 1 and num_mc_samples >= 1
    with _profile_section(runtime_profiler, "draw_base_u_randn"):
        return torch.randn(batch_size, num_queries, num_mc_samples, device=device)

# ============================================================================
# Section 1: Flow + physics
# ============================================================================


def _searchsorted_bin_locations(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    bin_idx = torch.searchsorted(bin_locations, inputs.unsqueeze(-1), right=True) - 1
    return bin_idx.clamp(min=0, max=bin_locations.shape[-1] - 2)


def rational_quadratic_spline_1d(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1D rational-quadratic spline with linear tails. Core numerics.

    Adapted from nflows (https://github.com/bayesiains/nflows).
    Inlined here to keep Subsystem B self-contained.
    Returns (outputs, logabsdet) both shape [batch].
    """
    assert inputs.dim() == 1, "Expected 1D inputs for the spline transform"
    assert unnormalized_widths.shape == unnormalized_heights.shape, "Width/height parameter shapes must match"
    assert unnormalized_widths.shape[0] == inputs.shape[0], "Expected one spline parameter set per input"
    assert unnormalized_derivatives.shape[-1] == unnormalized_widths.shape[-1] - 1, "Expected num_bins - 1 derivatives"

    num_bins = unnormalized_widths.shape[-1]
    assert min_bin_width * num_bins < 1.0, "min_bin_width too large for number of spline bins"
    assert min_bin_height * num_bins < 1.0, "min_bin_height too large for number of spline bins"

    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    if not torch.any(inside_interval_mask):
        return outputs, logabsdet

    inputs_inside = inputs[inside_interval_mask]
    widths_inside = unnormalized_widths[inside_interval_mask]
    heights_inside = unnormalized_heights[inside_interval_mask]
    derivatives_inside = unnormalized_derivatives[inside_interval_mask]

    derivatives_inside = nn.functional.pad(derivatives_inside, pad=(1, 1))
    boundary_constant = _inverse_softplus_target(1.0 - min_derivative)
    derivatives_inside[..., 0] = boundary_constant
    derivatives_inside[..., -1] = boundary_constant

    widths = nn.functional.softmax(widths_inside, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = 2.0 * tail_bound * cumwidths - tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = nn.functional.softmax(heights_inside, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = 2.0 * tail_bound * cumheights - tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = min_derivative + nn.functional.softplus(derivatives_inside)
    delta = heights / widths

    bin_idx = _searchsorted_bin_locations(cumheights if inverse else cumwidths, inputs_inside)

    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx).squeeze(-1)
    input_delta = delta.gather(-1, bin_idx).squeeze(-1)
    input_derivatives = derivatives[..., :-1].gather(-1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx).squeeze(-1)

    if inverse:
        normalized_input = inputs_inside - input_cumheights
        a = (
            normalized_input * (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
            + input_bin_heights * (input_delta - input_derivatives)
        )
        b = (
            input_bin_heights * input_derivatives
            - normalized_input * (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
        )
        c = -input_delta * normalized_input

        discriminant = b.pow(2) - 4.0 * a * c
        assert torch.all(discriminant >= 0.0), "Spline inverse encountered a negative discriminant"
        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        theta = root.clamp(0.0, 1.0)
        outputs_inside = theta * input_bin_widths + input_cumwidths
    else:
        theta = (inputs_inside - input_cumwidths) / input_bin_widths
        theta = theta.clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1.0 - theta)
        numerator = input_bin_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) * theta_one_minus_theta
        outputs_inside = input_cumheights + numerator / denominator

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = input_delta + (
        input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    ) * theta_one_minus_theta
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).pow(2)
    )
    logabsdet_inside = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    if inverse:
        logabsdet_inside = -logabsdet_inside

    outputs[inside_interval_mask] = outputs_inside
    logabsdet[inside_interval_mask] = logabsdet_inside
    return outputs, logabsdet


def batched_spline_forward(
    base_u: torch.Tensor,          # [B, m, S]
    spline_params: torch.Tensor,   # [B, m, d_sp]
    flow_config: FlowConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the 1D rational-quadratic spline transform in parallel across
    (B, m, S) using per-(b, i) spline parameters broadcast over S.

    Returns:
        u:         [B, m, S] transformed samples
        log_p_eta: [B, m, S] log-density of u under the flow
    """
    assert base_u.dim() == 3
    assert spline_params.dim() == 3
    B, m, S = base_u.shape
    num_bins = flow_config.num_bins
    d_sp = 2 * num_bins + (num_bins - 1)
    assert spline_params.shape == (B, m, d_sp), (
        f"Expected spline_params shape [{B}, {m}, {d_sp}], got {tuple(spline_params.shape)}"
    )
    base_u = base_u.contiguous()
    widths = spline_params[..., :num_bins]                                  # [B, m, num_bins]
    heights = spline_params[..., num_bins:2 * num_bins]                     # [B, m, num_bins]
    derivatives = spline_params[..., 2 * num_bins:]                         # [B, m, num_bins-1]
    assert derivatives.shape[-1] == num_bins - 1

    assert flow_config.min_bin_width * num_bins < 1.0, "min_bin_width too large for number of spline bins"
    assert flow_config.min_bin_height * num_bins < 1.0, "min_bin_height too large for number of spline bins"

    widths = nn.functional.softmax(widths, dim=-1)
    widths = flow_config.min_bin_width + (1.0 - flow_config.min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = 2.0 * flow_config.tail_bound * cumwidths - flow_config.tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = nn.functional.softmax(heights, dim=-1)
    heights = flow_config.min_bin_height + (1.0 - flow_config.min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = 2.0 * flow_config.tail_bound * cumheights - flow_config.tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = nn.functional.pad(derivatives, pad=(1, 1))
    boundary_constant = _inverse_softplus_target(1.0 - flow_config.min_derivative)
    derivatives[..., 0] = boundary_constant
    derivatives[..., -1] = boundary_constant
    derivatives = flow_config.min_derivative + nn.functional.softplus(derivatives)

    delta = heights / widths

    inside_interval_mask = (
        (base_u >= -flow_config.tail_bound) & (base_u <= flow_config.tail_bound)
    )
    # base_u is materialized contiguous once at the top of this function.
    bin_idx = torch.searchsorted(cumwidths, base_u, right=True) - 1
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx)
    input_bin_widths = widths.gather(-1, bin_idx)
    input_cumheights = cumheights.gather(-1, bin_idx)
    input_bin_heights = heights.gather(-1, bin_idx)
    input_delta = delta.gather(-1, bin_idx)
    input_derivatives = derivatives[..., :-1].gather(-1, bin_idx)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)

    theta = (base_u - input_cumwidths) / input_bin_widths
    theta = theta.clamp(0.0, 1.0)
    theta_one_minus_theta = theta * (1.0 - theta)
    numerator = input_bin_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    ) * theta_one_minus_theta
    u_inside = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).pow(2)
    )
    logabsdet_inside = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)

    u = torch.where(inside_interval_mask, u_inside, base_u)
    logabsdet = torch.where(inside_interval_mask, logabsdet_inside, torch.zeros_like(base_u))

    # Change of variables: log p_eta(u) = log p_base(base_u) - log|df/d(base_u)|
    log_base = standard_normal_logprob(base_u)
    log_p_eta = log_base - logabsdet
    return u, log_p_eta


def estimate_capo_batched(y_samples: torch.Tensor) -> torch.Tensor:
    """
    CAPO estimate for each (DGP, query): mean of f_BNN evaluations over
    the S latent samples. Differentiable through y_samples.
    Returns [B, m].
    """
    assert y_samples.dim() == 3
    return y_samples.mean(dim=-1)


def estimate_divergence_batched(
    log_r: torch.Tensor,       # [B, m, S]
    sensitivity_model: str,
) -> torch.Tensor:             # [B, m]
    """
    Estimate the GTSM divergence for each (DGP, query) from samples of the
    log density ratio r_nu(u) = nu(u) / phi(u), where nu is the
    counterfactual-arm latent. Samples are drawn from nu.

    MSM:        max(max_j r_j, 1 / min_j r_j)
    KL:         max(E_p_eta[r * log r] approx, E_p_eta[-log r])
    Rosenbaum:  max_j r_j / min_j r_j
    """
    assert log_r.dim() == 3

    if sensitivity_model == "msm":
        r = torch.exp(log_r)
        max_r = torch.amax(r, dim=-1)
        min_r = torch.amin(r, dim=-1)
        return torch.maximum(max_r, 1.0 / min_r)

    if sensitivity_model == "kl":
        kl_peta_phi = log_r.mean(dim=-1)
        kl_phi_peta = (-log_r * torch.exp(-log_r)).mean(dim=-1)
        gamma = torch.maximum(kl_peta_phi, kl_phi_peta)
        return torch.clamp(gamma, min=0.0) # clamping against numerical errors; true gamma is non-negative 

    if sensitivity_model == "rosenbaum":
        r = torch.exp(log_r)
        return torch.amax(r, dim=-1) / torch.amin(r, dim=-1)

    assert False, f"Unsupported sensitivity_model: {sensitivity_model}"


# ============================================================================
# Section 1.5: Outcome-BNN depth bucketing
# ============================================================================


@dataclass
class DepthBucket:
    """
    All DGPs in a process_dgp_batch with a common num_layers, padded to a
    common width. Constructed once per DGP batch and reused across all
    optimizer steps and lambda solves.
    """
    depth: int
    width_padded: int
    dgp_indices: torch.Tensor
    layer_weights: List[torch.Tensor]
    layer_biases: List[torch.Tensor]
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


@dataclass
class OutcomeForwardPlan:
    """Named policy for evaluating frozen outcome BNNs inside the optimizer."""
    use_bucket_forward: bool
    depth_buckets: Optional[List[DepthBucket]]
    sequential_f_bnns: Optional[List[nn.Module]] = None


def _outcome_linear_layers(raw_model: OutcomeMLP) -> List[nn.Linear]:
    assert raw_model.activation_name == "tanh"
    assert raw_model.input_size == 12, f"Expected OutcomeMLP input_size=12, got {raw_model.input_size}"
    assert raw_model.num_layers in {3, 4, 5}, f"Unexpected OutcomeMLP num_layers={raw_model.num_layers}"
    modules = list(raw_model.network)
    expected_num_modules = 2 * (raw_model.num_layers - 2) + 1
    assert len(modules) == expected_num_modules, (
        f"Expected {expected_num_modules} modules for depth {raw_model.num_layers}, got {len(modules)}"
    )

    linears: List[nn.Linear] = []
    for hidden_index in range(raw_model.num_layers - 2):
        linear_position = 2 * hidden_index
        activation_position = linear_position + 1
        assert isinstance(modules[linear_position], nn.Linear)
        assert isinstance(modules[activation_position], nn.Tanh)
        linears.append(modules[linear_position])
    assert isinstance(modules[-1], nn.Linear)
    linears.append(modules[-1])

    assert len(linears) == raw_model.num_layers - 1
    for layer_index, linear in enumerate(linears):
        if layer_index == 0:
            expected_in = raw_model.input_size
            expected_out = raw_model.hidden_size
        elif layer_index == len(linears) - 1:
            expected_in = raw_model.hidden_size
            expected_out = 1
        else:
            expected_in = raw_model.hidden_size
            expected_out = raw_model.hidden_size
        assert linear.in_features == expected_in and linear.out_features == expected_out, (
            f"Layer {layer_index} has shape ({linear.in_features}, {linear.out_features}); "
            f"expected ({expected_in}, {expected_out})"
        )
    return linears


def build_depth_buckets(
    f_bnns: List[nn.Module],
    device: str,
) -> List[DepthBucket]:
    """
    Inspect each wrapper, group DGPs by raw_model.num_layers, pad hidden weights
    to the per-bucket maximum width with zeros, and stack bmm-friendly constants.
    """
    assert len(f_bnns) >= 1
    grouped: Dict[int, List[Tuple[int, NormalizedOutcomeMLPWrapper]]] = {}
    for dgp_index, f_bnn in enumerate(f_bnns):
        assert isinstance(f_bnn, NormalizedOutcomeMLPWrapper), (
            f"Expected NormalizedOutcomeMLPWrapper, got {type(f_bnn).__name__}"
        )
        raw_model = f_bnn.raw_model
        assert isinstance(raw_model, OutcomeMLP), (
            f"Expected wrapped raw_model OutcomeMLP, got {type(raw_model).__name__}"
        )
        _outcome_linear_layers(raw_model)
        assert f_bnn.x_mean.shape == (1, raw_model.input_size - 2)
        assert f_bnn.x_std.shape == (1, raw_model.input_size - 2)
        assert f_bnn.y_mean.shape == (1, 1)
        assert f_bnn.y_std.shape == (1, 1)
        grouped.setdefault(raw_model.num_layers, []).append((dgp_index, f_bnn))

    buckets: List[DepthBucket] = []
    for depth, entries in grouped.items():
        width_padded = max(int(wrapper.raw_model.hidden_size) for _, wrapper in entries)
        num_linears = depth - 1
        layer_weights: List[torch.Tensor] = []
        layer_biases: List[torch.Tensor] = []
        for layer_index in range(num_linears):
            if layer_index == 0:
                in_pad, out_pad = 12, width_padded
            elif layer_index == num_linears - 1:
                in_pad, out_pad = width_padded, 1
            else:
                in_pad, out_pad = width_padded, width_padded

            weights_padded: List[torch.Tensor] = []
            biases_padded: List[torch.Tensor] = []
            for _, wrapper in entries:
                linears = _outcome_linear_layers(wrapper.raw_model)
                linear = linears[layer_index]
                weight = linear.weight.detach().transpose(0, 1).clone()
                bias = linear.bias.detach().clone()
                padded_weight = torch.zeros(
                    in_pad, out_pad, dtype=weight.dtype, device=device
                )
                padded_bias = torch.zeros(out_pad, dtype=bias.dtype, device=device)
                padded_weight[:linear.in_features, :linear.out_features] = weight.to(device=device)
                padded_bias[:linear.out_features] = bias.to(device=device)
                weights_padded.append(padded_weight)
                biases_padded.append(padded_bias)
            layer_weights.append(torch.stack(weights_padded, dim=0).detach())
            layer_biases.append(torch.stack(biases_padded, dim=0).detach())

        dgp_indices = torch.tensor(
            [dgp_index for dgp_index, _ in entries], dtype=torch.int64, device=device
        )
        x_mean = torch.stack(
            [wrapper.x_mean.detach().reshape(-1).clone().to(device=device) for _, wrapper in entries],
            dim=0,
        )
        x_std = torch.stack(
            [wrapper.x_std.detach().reshape(-1).clone().to(device=device) for _, wrapper in entries],
            dim=0,
        )
        y_mean = torch.stack(
            [wrapper.y_mean.detach().reshape(-1).clone().to(device=device) for _, wrapper in entries],
            dim=0,
        )
        y_std = torch.stack(
            [wrapper.y_std.detach().reshape(-1).clone().to(device=device) for _, wrapper in entries],
            dim=0,
        )

        for tensor in layer_weights + layer_biases + [x_mean, x_std, y_mean, y_std]:
            assert not tensor.requires_grad
        buckets.append(DepthBucket(
            depth=depth,
            width_padded=width_padded,
            dgp_indices=dgp_indices,
            layer_weights=layer_weights,
            layer_biases=layer_biases,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        ))
    return buckets


def bucket_forward(
    bucket: DepthBucket,
    x_q: torch.Tensor,
    a_q: torch.Tensor,
    u_q: torch.Tensor,
) -> torch.Tensor:
    """
    Width-padded, tanh-activated, batched MLP forward across all DGPs in the
    bucket. Mirrors NormalizedOutcomeMLPWrapper.
    """
    assert x_q.dim() == 3 and a_q.dim() == 3 and u_q.dim() == 3
    G, m, d_x = x_q.shape
    S = u_q.shape[-1]
    assert a_q.shape == (G, m, 1)
    assert u_q.shape == (G, m, S)
    assert bucket.x_mean.shape == (G, d_x)
    assert bucket.x_std.shape == (G, d_x)
    assert bucket.y_mean.shape == (G, 1)
    assert bucket.y_std.shape == (G, 1)

    x_raw_q = x_q * bucket.x_std.unsqueeze(1) + bucket.x_mean.unsqueeze(1)
    x_exp = x_raw_q.unsqueeze(2).expand(G, m, S, d_x).reshape(G, m * S, d_x)
    a_exp = a_q.unsqueeze(2).expand(G, m, S, 1).reshape(G, m * S, 1)
    u_exp = u_q.unsqueeze(-1).reshape(G, m * S, 1)

    h = torch.cat([x_exp, a_exp, u_exp], dim=-1)
    assert h.shape == (G, m * S, 12)

    num_linears = bucket.depth - 1
    assert len(bucket.layer_weights) == num_linears
    assert len(bucket.layer_biases) == num_linears
    for layer_index in range(num_linears):
        W = bucket.layer_weights[layer_index]
        b = bucket.layer_biases[layer_index]
        assert W.shape[0] == G and b.shape[0] == G
        assert h.shape[-1] == W.shape[1]
        h = torch.bmm(h, W) + b.unsqueeze(1)
        if layer_index < num_linears - 1:
            h = torch.tanh(h)

    y_raw = h.reshape(G, m, S)
    return (y_raw - bucket.y_mean.unsqueeze(1)) / bucket.y_std.unsqueeze(1)


def sequential_outcome_forward(
    f_bnns: List[nn.Module],
    x_queries: torch.Tensor,
    a_queries: torch.Tensor,
    u_query: torch.Tensor,
) -> torch.Tensor:
    B, m, d_x = x_queries.shape
    S = u_query.shape[-1]
    assert len(f_bnns) == B
    y_list: List[torch.Tensor] = []
    for b in range(B):
        x_b = x_queries[b].unsqueeze(-2).expand(m, S, d_x).reshape(m * S, d_x)
        a_b = a_queries[b].unsqueeze(-2).expand(m, S, 1).reshape(m * S, 1)
        u_b = u_query[b].unsqueeze(-1).reshape(m * S, 1)
        y_flat = f_bnns[b](x_b, a_b, u_b)
        y_list.append(y_flat.reshape(m, S))
    return torch.stack(y_list, dim=0)


def evaluate_objective_batched(
    spline_params: torch.Tensor,           # [B, m, d_sp]
    outcome_plan: OutcomeForwardPlan,
    x_queries: torch.Tensor,           # [B, m, d_x]
    a_queries: torch.Tensor,           # [B, m, 1]
    pi_queries: torch.Tensor,          # [B, m]
    lambda_value: float,
    bound_type: str,
    sensitivity_model: str,
    flow_config: FlowConfig,
    num_mc_samples: int,
    device: str,
    runtime_profiler: Optional[RuntimeSolveProfiler] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single differentiable forward pass: spline flow -> f_BNN ->
    scalarized objective. Returns (theta, gamma, objective), all [B, m].
    """
    assert bound_type in ("upper", "lower"), f"Unsupported bound_type: {bound_type}"
    assert x_queries.dim() == 3 and a_queries.dim() == 3
    B, m, d_x = x_queries.shape
    assert a_queries.shape == (B, m, 1)
    assert pi_queries.dtype == torch.float32
    assert pi_queries.shape == (B, m)
    num_bins = flow_config.num_bins
    d_sp = 2 * num_bins + (num_bins - 1)
    assert spline_params.shape == (B, m, d_sp)
    if outcome_plan.use_bucket_forward:
        assert outcome_plan.depth_buckets is not None
    else:
        assert outcome_plan.sequential_f_bnns is not None, (
            "Sequential fallback requires f_bnns; bucket forward is disabled"
        )
        assert len(outcome_plan.sequential_f_bnns) == B
    S = num_mc_samples

    with _profile_section(runtime_profiler, "evaluate_objective_total"):
        # Sample base and transform
        with _profile_section(runtime_profiler, "draw_base_u"):
            base_u = draw_base_u(
                batch_size=B,
                num_queries=m,
                num_mc_samples=S,
                device=device,
                runtime_profiler=runtime_profiler,
            )
        with _profile_section(runtime_profiler, "batched_spline_forward"):
            nu_samples, log_p_eta = batched_spline_forward(
                base_u, spline_params, flow_config
            )   # both [B, m, S]

        with _profile_section(runtime_profiler, "log_phi_log_r"):
            log_phi_nu = standard_normal_logprob(nu_samples)      # [B, m, S]
            log_r_nu = log_p_eta - log_phi_nu                     # [B, m, S]

        with _profile_section(runtime_profiler, "draw_phi_query_u"):
            phi_samples = draw_base_u(
                batch_size=B,
                num_queries=m,
                num_mc_samples=S,
                device=device,
                runtime_profiler=runtime_profiler,
            )

        with _profile_section(runtime_profiler, "mixture_query_u"):
            pi_expanded = pi_queries.unsqueeze(-1).expand(-1, -1, S)
            xi = torch.bernoulli(pi_expanded)
            assert xi.shape == nu_samples.shape == phi_samples.shape == (B, m, S)
            u_query = xi * phi_samples + (1.0 - xi) * nu_samples

        if outcome_plan.use_bucket_forward:
            with _profile_section(runtime_profiler, "f_bnn_bucket_forward"):
                assert outcome_plan.depth_buckets is not None
                y_samples = torch.empty(B, m, S, device=device, dtype=u_query.dtype)
                for bucket in outcome_plan.depth_buckets:
                    idx = bucket.dgp_indices
                    x_q_b = x_queries.index_select(0, idx)
                    a_q_b = a_queries.index_select(0, idx)
                    u_q_b = u_query.index_select(0, idx)
                    y_b = bucket_forward(bucket, x_q_b, a_q_b, u_q_b)
                    y_samples.index_copy_(0, idx, y_b)
        else:
            with _profile_section(runtime_profiler, "f_bnn_forward_loop"):
                assert outcome_plan.sequential_f_bnns is not None
                y_samples = sequential_outcome_forward(
                    outcome_plan.sequential_f_bnns, x_queries, a_queries, u_query
                )

        theta = estimate_capo_batched(y_samples)                  # [B, m]
        with _profile_section(runtime_profiler, "estimate_divergence"):
            gamma = estimate_divergence_batched(log_r_nu, sensitivity_model)  # [B, m]

        with _profile_section(runtime_profiler, "objective_assembly"):
            if bound_type == "upper":
                objective = theta - lambda_value * gamma
            else:
                objective = -theta - lambda_value * gamma

    return theta, gamma, objective


# ============================================================================
# Section 2: Solver
# ============================================================================


def optimize_lambda_batch(
    outcome_plan: OutcomeForwardPlan,
    x_queries: torch.Tensor,
    a_queries: torch.Tensor,
    pi_queries: torch.Tensor,
    lambda_value: float,
    bound_type: str,
    sensitivity_model: str,
    spline_params_init: Optional[np.ndarray],
    flow_config: FlowConfig,
    solver_config: SolverConfig,
    device: str,
    lambda_index: Optional[int] = None,
    profiling_state: Optional[ProfilingState] = None,
    dataset_ids: Optional[Sequence[int]] = None,
    save_diagnostics: bool = False,
) -> SolveResult:
    """
    Optimize the scalarized objective at a fixed lambda, jointly across all
    (DGP, query) pairs in the batch. No patience, no best-step snapshotting:
    fixed step budget, then a high-sample-count eval pass.

    spline_params_init carries the explicit warm-start from the previous lambda.
    Adam is recreated here (fresh momentum) because the objective landscape
    shifts with lambda and stale momentum destabilizes the next solve.
    """
    assert x_queries.dim() == 3 and a_queries.dim() == 3
    B, m, _ = x_queries.shape
    assert pi_queries.dtype == torch.float32
    assert pi_queries.shape == (B, m)
    num_bins = flow_config.num_bins
    d_sp = 2 * num_bins + (num_bins - 1)
    batch_ids = list(dataset_ids) if dataset_ids is not None else []
    runtime_profiler: Optional[RuntimeSolveProfiler] = None
    torch_profile_enabled = False
    if profiling_state is not None:
        _, runtime_profiler, torch_profile_enabled = profiling_state.begin_solve(
            dataset_batch_ids=batch_ids,
            bound_type=bound_type,
            lambda_index=lambda_index,
            lambda_value=lambda_value,
            num_queries=m,
            train_mc_samples=solver_config.num_mc_samples_train,
            device=device,
        )

    try:
        with _profile_section(runtime_profiler, "optimize_lambda_batch_total"):
            if spline_params_init is None:
                derivative_start = 2 * num_bins
                derivative_init = _inverse_softplus_target(1.0 - flow_config.min_derivative)
                spline_params = torch.zeros(B, m, d_sp, dtype=torch.float32, device=device)
                spline_params[..., derivative_start:].fill_(derivative_init)
            else:
                assert spline_params_init.shape == (B, m, d_sp), (
                    f"Expected spline_params_init shape {(B, m, d_sp)}, "
                    f"got {spline_params_init.shape}"
                )
                spline_params = torch.as_tensor(spline_params_init, dtype=torch.float32, device=device).clone()

            spline_params_param = nn.Parameter(spline_params)
            lr_mult = lr_multiplier_for_lambda(lambda_value, solver_config)
            effective_lr = float(solver_config.learning_rate) * lr_mult
            effective_max_steps, max_steps_mult = max_steps_for_lambda(
                lambda_value, solver_config, lr_mult=lr_mult
            )
            optimizer = torch.optim.Adam([spline_params_param], lr=effective_lr)
            trace: List[Dict[str, float]] = []
            stop_reason = "max_steps"
            steps_completed = 0
            production_mode = (
                not solver_config.early_stop
                and runtime_profiler is None
                and not torch_profile_enabled
                and not save_diagnostics
            )
            t0 = time.perf_counter()

            best_objective = -math.inf
            best_at_last_check = -math.inf
            stale_checks = 0

            torch_profiler_context = nullcontext()
            if torch_profile_enabled:
                assert profiling_state is not None
                os.makedirs(profiling_state.torch_profile_dir, exist_ok=True)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                torch_profiler_context = torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(
                        wait=profiling_state.config.torch_profile_wait,
                        warmup=profiling_state.config.torch_profile_warmup,
                        active=profiling_state.config.torch_profile_active,
                        repeat=profiling_state.config.torch_profile_repeat,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        profiling_state.torch_profile_dir
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )

            with torch_profiler_context as prof:
                for step in range(effective_max_steps):
                    optimizer.zero_grad(set_to_none=True)

                    with _profile_section(runtime_profiler, "forward_train"):
                        theta, gamma, objective = evaluate_objective_batched(
                            spline_params=spline_params_param,
                            outcome_plan=outcome_plan,
                            x_queries=x_queries,
                            a_queries=a_queries,
                            pi_queries=pi_queries,
                            lambda_value=lambda_value,
                            bound_type=bound_type,
                            sensitivity_model=sensitivity_model,
                            flow_config=flow_config,
                            num_mc_samples=solver_config.num_mc_samples_train,
                            device=device,
                            runtime_profiler=runtime_profiler,
                        )

                    if production_mode:
                        if (step + 1) % 50 == 0 and not torch.isfinite(objective).all().item():
                            stop_reason = "nonfinite"
                            break
                    else:
                        if not _all_finite(theta, gamma, objective):
                            stop_reason = "nonfinite"
                            break

                    with _profile_section(runtime_profiler, "loss_backward"):
                        if solver_config.loss_reduction == "batch_mean":
                            loss = -objective.mean()
                        elif solver_config.loss_reduction == "per_dgp_sum":
                            loss = -objective.mean(dim=1).sum()
                        else:
                            raise ValueError(
                                f"Unknown loss_reduction={solver_config.loss_reduction!r}"
                            )
                        loss.backward()
                    if not production_mode:
                        with _profile_section(runtime_profiler, "gradient_norm"):
                            with _profile_section(runtime_profiler, "gradient_norm_compute"):
                                grad_norm_tensor = _compute_gradient_norm_tensor([spline_params_param])
                            with _profile_section(runtime_profiler, "gradient_norm_scalar_extraction"):
                                grad_norm = _gradient_norm_to_float(grad_norm_tensor)
                        if not math.isfinite(grad_norm):
                            stop_reason = "nonfinite"
                            break

                    # MSM can create sharp gradient spikes; clip per query to avoid cross-query coupling.
                    with _profile_section(runtime_profiler, "gradient_clip"):
                        _clip_spline_param_grad_norm_(spline_params_param, max_norm=1.0)
                    with _profile_section(runtime_profiler, "optimizer_step"):
                        optimizer.step()
                    if torch_profile_enabled:
                        assert prof is not None
                        prof.step()
                    steps_completed = step + 1

                    if not production_mode:
                        with _profile_section(runtime_profiler, "trace_scalar_extraction"):
                            objective_value = float(objective.mean().detach().cpu())
                            theta_mean = float(theta.mean().detach().cpu())
                            gamma_mean = float(gamma.mean().detach().cpu())

                        with _profile_section(runtime_profiler, "trace_append"):
                            trace.append({
                                "step": step,
                                "objective": objective_value,
                                "theta_mean": theta_mean,
                                "gamma_mean": gamma_mean,
                                "grad_norm": grad_norm,
                            })

                        if objective_value > best_objective:
                            best_objective = objective_value

                        if (
                            solver_config.early_stop
                            and (step + 1) >= solver_config.early_stop_min_steps
                            and (step + 1) % solver_config.early_stop_check_every == 0
                        ):
                            if not math.isfinite(best_at_last_check):
                                best_at_last_check = best_objective
                                stale_checks = 0
                            else:
                                scale = max(1.0, abs(best_at_last_check))
                                tolerance = solver_config.early_stop_abs_tol + solver_config.early_stop_rel_tol * scale

                                if best_objective <= best_at_last_check + tolerance:
                                    stale_checks += 1
                                else:
                                    stale_checks = 0
                                    best_at_last_check = best_objective

                                if stale_checks >= solver_config.early_stop_patience:
                                    stop_reason = "early_stop"
                                    break

            if torch_profile_enabled:
                assert prof is not None
                sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
                tqdm.write(prof.key_averages().table(sort_by=sort_by, row_limit=30))

            # Final eval at high sample count, no grad.
            #
            with _profile_section(runtime_profiler, "final_eval"):
                with torch.no_grad():
                    theta_eval, gamma_eval, _ = evaluate_objective_batched(
                        spline_params=spline_params_param,
                        outcome_plan=outcome_plan,
                        x_queries=x_queries,
                        a_queries=a_queries,
                        pi_queries=pi_queries,
                        lambda_value=lambda_value,
                        bound_type=bound_type,
                        sensitivity_model=sensitivity_model,
                        flow_config=flow_config,
                        num_mc_samples=solver_config.num_mc_samples_eval,
                        device=device,
                        runtime_profiler=runtime_profiler,
                    )

            return SolveResult(
                theta_star=theta_eval.detach().cpu().numpy(),
                gamma_star=gamma_eval.detach().cpu().numpy(),
                spline_params_final=spline_params_param.detach().cpu().numpy(),
                lambda_value=float(lambda_value),
                bound_type=bound_type,
                num_steps=steps_completed if production_mode else len(trace),
                stop_reason=stop_reason,
                runtime_seconds=time.perf_counter() - t0,
                trace=trace,
                learning_rate_base=float(solver_config.learning_rate),
                learning_rate_multiplier=float(lr_mult),
                learning_rate_effective=float(effective_lr),
                lr_lambda_schedule=solver_config.lr_lambda_schedule,
                max_steps_base=int(solver_config.max_steps),
                max_steps_effective=int(effective_max_steps),
                max_steps_multiplier=float(max_steps_mult),
                max_steps_lambda_schedule=solver_config.max_steps_lambda_schedule,
            )
    finally:
        if profiling_state is not None:
            profiling_state.finish_runtime_solve(runtime_profiler)


# ============================================================================
# Section 3: Pipeline
# ============================================================================


def compute_query_ordering_scores(X: torch.Tensor) -> torch.Tensor:
    """
    Cheap ordering proxy for deterministic subsampling. Linear score on
    features. Returns shape [n_rows].
    """
    feature_weights = torch.arange(1, X.shape[1] + 1, dtype=X.dtype, device=X.device)
    return X @ feature_weights


def select_query_row_indices(
    X: torch.Tensor,
    config: QueryConfig,
) -> List[int]:
    """
    Select and order row indices from X for query point construction.

    Supports: "random", "ordered_proxy", "stratified_proxy".
    Returns row indices ordered by the proxy score.
    """
    assert X.dim() == 2, "Expected X with shape [n_rows, x_dim]"
    num_rows = X.shape[0]
    assert num_rows > 0, "Cannot sample query points from an empty dataset"

    num_query_points = min(config.num_query_points, num_rows)

    if config.sampling_mode == "random":
        rng = random.Random(config.random_seed)
        selected = sorted(rng.sample(list(range(num_rows)), num_query_points))
        return selected

    scores = compute_query_ordering_scores(X).detach().cpu()
    ordered_indices = [idx for idx, _ in sorted(enumerate(scores.tolist()), key=lambda item: item[1])]

    if config.sampling_mode == "ordered_proxy":
        if num_query_points == num_rows:
            return ordered_indices
        positions = torch.linspace(0, num_rows - 1, steps=num_query_points)
        selected_positions = sorted(set(int(round(p.item())) for p in positions))
        while len(selected_positions) < num_query_points:
            candidate = selected_positions[-1] + 1
            if candidate < num_rows:
                selected_positions.append(candidate)
            else:
                break
        return [ordered_indices[p] for p in selected_positions[:num_query_points]]

    if config.sampling_mode == "stratified_proxy":
        if num_query_points == num_rows:
            return ordered_indices
        selected: List[int] = []
        for bucket in range(num_query_points):
            start = int(round(bucket * num_rows / num_query_points))
            end = int(round((bucket + 1) * num_rows / num_query_points))
            assert start < end, "Expected non-empty stratified proxy bucket"
            midpoint = start + (end - start - 1) // 2
            selected.append(ordered_indices[midpoint])
        return selected

    assert False, f"Unsupported sampling_mode: {config.sampling_mode}"


def build_query_specs(
    X: torch.Tensor,
    config: QueryConfig,
    propensities_per_row: torch.Tensor,
) -> List[QuerySpec]:
    """
    Build full list of QuerySpecs for one dataset. Selects rows via
    select_query_row_indices, crosses with treatment arms.
    """
    selected_rows = select_query_row_indices(X=X, config=config)
    propensities_per_row = propensities_per_row.detach().cpu().reshape(-1)
    assert propensities_per_row.shape == (len(selected_rows),)
    query_specs: List[QuerySpec] = []
    query_id = 0
    for row_offset, row_index in enumerate(selected_rows):
        x_query = X[row_index].clone()
        p_a1 = float(propensities_per_row[row_offset].item())
        for a_query in config.treatment_arms:
            pi_query = p_a1 if int(a_query) == 1 else 1.0 - p_a1
            query_specs.append(
                QuerySpec(
                    query_id=query_id,
                    source_row_index=row_index,
                    a_query=int(a_query),
                    x_query=x_query,
                    pi_query=float(pi_query),
                )
            )
            query_id += 1
    return query_specs


def sweep_lambda_grid_batch(
    outcome_plan: OutcomeForwardPlan,
    x_queries_batch: torch.Tensor,               # [B, m, d_x]
    a_queries_batch: torch.Tensor,               # [B, m, 1]
    pi_queries_batch: torch.Tensor,              # [B, m]
    dataset_ids: List[int],
    query_specs_batch: List[List[QuerySpec]],
    bound_type: str,
    lambda_grid: List[float],
    sensitivity_model: str,
    flow_config: FlowConfig,
    solver_config: SolverConfig,
    device: str,
    profiling_state: Optional[ProfilingState] = None,
    save_diagnostics: bool = False,
    warm_start_lambdas: bool = True,
) -> List[SolveResult]:
    """
    Sweep all lambdas for one (DGP batch, bound_type).

    By default, each lambda receives the solved spline parameters from the
    previous lambda. Disable warm-start for speed-vs-quality benchmarking.
    """
    assert x_queries_batch.dim() == 3 and a_queries_batch.dim() == 3
    B, m, _ = x_queries_batch.shape
    assert a_queries_batch.shape == (B, m, 1)
    assert pi_queries_batch.dtype == torch.float32
    assert pi_queries_batch.shape == (B, m)
    assert len(dataset_ids) == B and len(query_specs_batch) == B

    results: List[SolveResult] = []
    current_params_init: Optional[np.ndarray] = None
    for lambda_index, lambda_value in enumerate(lambda_grid):
        init_for_lambda = current_params_init if warm_start_lambdas else None
        result = optimize_lambda_batch(
            outcome_plan=outcome_plan,
            x_queries=x_queries_batch,
            a_queries=a_queries_batch,
            pi_queries=pi_queries_batch,
            lambda_value=lambda_value,
            bound_type=bound_type,
            lambda_index=lambda_index,
            sensitivity_model=sensitivity_model,
            spline_params_init=init_for_lambda,
            flow_config=flow_config,
            solver_config=solver_config,
            device=device,
            profiling_state=profiling_state,
            dataset_ids=dataset_ids,
            save_diagnostics=save_diagnostics,
        )
        results.append(result)
        if warm_start_lambdas:
            current_params_init = result.spline_params_final

    return results


def process_dgp_batch(
    dataset_records: List[Dict],
    config: PipelineConfig,
    output_paths: OutputPaths,
    lambda_grid: List[float],
    profiling_state: Optional[ProfilingState] = None,
    log_fn: Callable[[str], None] = tqdm.write,
) -> None:
    """
    Full frontier construction for a batch of B DGPs. All queries for all
    DGPs are optimized jointly with independent per-query spline parameters.
    """
    B = len(dataset_records)
    assert B >= 1

    f_bnns: List[nn.Module] = []
    dataset_ids: List[int] = []
    query_specs_batch: List[List[QuerySpec]] = []
    x_tensors: List[torch.Tensor] = []
    a_tensors: List[torch.Tensor] = []
    pi_tensors: List[torch.Tensor] = []

    for record in dataset_records:
        dataset_id = record["dataset_id"]
        dataset_ids.append(dataset_id)

        f_bnn = record["f_BNN"].to(config.device)
        f_bnn.eval()
        for p in f_bnn.parameters():
            p.requires_grad_(False)
        f_bnns.append(f_bnn)

        f_a = record["f_A"].to(config.device)
        f_a.eval()
        for p in f_a.parameters():
            p.requires_grad_(False)

        X = record["X"]
        selected_rows = select_query_row_indices(X=X, config=config.query)
        x_selected = X[selected_rows].to(device=config.device, dtype=torch.float32)
        with torch.no_grad():
            p_a1_selected = f_a(x_selected).reshape(-1).to(dtype=torch.float32)
        assert p_a1_selected.shape == (len(selected_rows),)
        assert (p_a1_selected >= 0).all() and (p_a1_selected <= 1).all()

        queries = build_query_specs(
            X=X,
            config=config.query,
            propensities_per_row=p_a1_selected,
        )
        query_specs_batch.append(queries)

        x_q = torch.stack([q.x_query for q in queries], dim=0).to(
            device=config.device, dtype=torch.float32
        )
        a_q = torch.tensor(
            [[q.a_query] for q in queries], dtype=torch.float32, device=config.device
        )
        pi_q = torch.tensor(
            [q.pi_query for q in queries], dtype=torch.float32, device=config.device
        )
        x_tensors.append(x_q)
        a_tensors.append(a_q)
        pi_tensors.append(pi_q)

    depth_buckets = build_depth_buckets(f_bnns=f_bnns, device=config.device)
    bucket_distribution = (
        f"DGP batch {dataset_ids}: depth buckets = "
        + ", ".join(
            f"depth={bucket.depth} width={bucket.width_padded} G={int(bucket.dgp_indices.numel())}"
            for bucket in depth_buckets
        )
    )
    log_fn(bucket_distribution)
    outcome_plan = OutcomeForwardPlan(
        use_bucket_forward=config.use_bucket_forward,
        depth_buckets=depth_buckets,
        sequential_f_bnns=f_bnns if not config.use_bucket_forward else None,
    )

    # Batching requires a uniform (m, d_x) across the DGPs in the batch.
    ref_shape = x_tensors[0].shape
    assert all(x.shape == ref_shape for x in x_tensors), (
        f"DGP batch has non-uniform query shapes: {[tuple(x.shape) for x in x_tensors]}"
    )
    x_queries_batch = torch.stack(x_tensors, dim=0)   # [B, m, d_x]
    a_queries_batch = torch.stack(a_tensors, dim=0)   # [B, m, 1]
    pi_queries_batch = torch.stack(pi_tensors, dim=0) # [B, m]

    m = x_queries_batch.shape[1]
    log_fn(
        f"DGP batch {dataset_ids}: B={B} x m={m} queries x 2 bounds x "
        f"{len(lambda_grid)} lambdas"
    )

    all_results: Dict[str, List[SolveResult]] = {}
    for bound_type in ("upper", "lower"):
        results = sweep_lambda_grid_batch(
            outcome_plan=outcome_plan,
            x_queries_batch=x_queries_batch,
            a_queries_batch=a_queries_batch,
            pi_queries_batch=pi_queries_batch,
            dataset_ids=dataset_ids,
            query_specs_batch=query_specs_batch,
            bound_type=bound_type,
            lambda_grid=lambda_grid,
            sensitivity_model=config.sensitivity_model,
            flow_config=config.flow,
            solver_config=config.solver,
            device=config.device,
            profiling_state=profiling_state,
            save_diagnostics=config.save_diagnostics,
            warm_start_lambdas=config.warm_start_lambdas,
        )
        all_results[bound_type] = results

    for b, dataset_id in enumerate(dataset_ids):
        queries = query_specs_batch[b]
        write_queries_csv(queries, output_paths, dataset_id)
        write_frontier_csv(all_results, b, queries, output_paths, dataset_id)
        if config.save_diagnostics:
            write_diagnostics(all_results, b, queries, output_paths, dataset_id)
            write_frontier_diagnostics_csv(all_results, b, queries, output_paths, dataset_id)
        
        if config.save_spline_params:
            write_spline_params_npz(all_results, b, queries, output_paths, dataset_id)


def run_pipeline(config: PipelineConfig) -> None:
    """
    Top-level entry point. Loads datasets, skips already-completed ones,
    groups the rest into DGP batches, and delegates to process_dgp_batch.
    """
    assert config.estimand_type == "capo", "v4 supports capo only"
    assert config.sensitivity_model in ("msm", "kl", "rosenbaum"), (
        f"Unsupported sensitivity_model: {config.sensitivity_model}"
    )
    assert config.dgp_batch_size >= 1
    validate_lr_lambda_config(config.solver)
    validate_max_steps_lambda_config(config.solver)

    _seed_everything(config.query.random_seed)

    lambda_grid = make_lambda_grid(config.lambda_grid_cfg)
    assert len(lambda_grid) > 0, "lambda_grid must be non-empty"
    assert lambda_grid == sorted(lambda_grid, reverse=True), \
        "lambda_grid must be ordered large to small"

    root_dir = resolve_code_relative_path(config.root_dir)
    output_dir = resolve_code_relative_path(config.output_dir)
    output_paths = build_output_paths(output_dir)
    profiling_state = ProfilingState(config.profiling, output_paths)
    if config.copy_dataset_csvs:
        _copy_datasets_if_requested(root_dir, output_paths)
    write_run_metadata(config, output_paths, lambda_grid=lambda_grid)

    dataset = CausalFrontierDataset(
        root_dir=root_dir,
        model_device=config.device,
        log_fn=lambda msg: None,
    )

    if config.dataset_ids is None:
        indices = list(range(len(dataset)))
    else:
        id_to_idx = {a["dataset_id"]: i for i, a in enumerate(dataset.artifacts)}
        missing = [d for d in config.dataset_ids if d not in id_to_idx]
        assert not missing, f"Requested dataset ids not found: {missing}"
        indices = [id_to_idx[d] for d in config.dataset_ids]

    indices_to_process: List[int] = []
    for idx in indices:
        artifact = dataset.artifacts[idx]
        dataset_id = artifact["dataset_id"]
        csv_path = os.path.join(output_paths.frontiers_dir, f"frontier_points_{dataset_id}.csv")
        if os.path.exists(csv_path):
            tqdm.write(f"Skipping dataset {dataset_id}: frontier already exists at {csv_path}")
            continue
        indices_to_process.append(idx)

    tqdm.write(
        f"Frontier v4 run: {len(indices_to_process)} datasets to process "
        f"(skipped {len(indices) - len(indices_to_process)}) | "
        f"dgp_batch_size={config.dgp_batch_size} | "
        f"{config.query.num_query_points} query rows x {len(config.query.treatment_arms)} arms | "
        f"{len(lambda_grid)} lambdas"
    )

    batch_starts = list(range(0, len(indices_to_process), config.dgp_batch_size))
    failures: List[str] = []

    for batch_start in tqdm(batch_starts, desc="DGP batches"):
        batch_indices = indices_to_process[batch_start:batch_start + config.dgp_batch_size]
        records: List[Dict] = []
        batch_ids: List[int] = []
        for idx in batch_indices:
            record = dataset[idx]
            record["dataset_path"] = dataset.artifacts[idx].get("dataset_path", "")
            records.append(record)
            batch_ids.append(record["dataset_id"])
        try:
            process_dgp_batch(
                records,
                config,
                output_paths,
                lambda_grid=lambda_grid,
                profiling_state=profiling_state,
            )
        except Exception as exc:
            failures.append(f"batch {batch_ids}: {type(exc).__name__}: {exc}")

    profiling_state.write_runtime_outputs()

    if torch.cuda.is_available() and torch.device(config.device).type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device=torch.device(config.device)) / 1e9
        tqdm.write(f"[cuda-memory] peak_allocated_gb={peak_memory_gb:.6f}")

    if failures:
        summary = "\n".join(failures)
        raise RuntimeError(
            f"Frontier v4 pipeline finished with {len(failures)} batch failure(s):\n{summary}"
        )


def make_lambda_grid(cfg: LambdaGridConfig) -> List[float]:
    """Returns lambda values ordered large -> small (tight -> permissive bounds)."""
    assert cfg.lambda_min > 0, "lambda_min must be positive"
    assert cfg.lambda_max > cfg.lambda_min, "lambda_max must exceed lambda_min"
    assert cfg.lambda_n >= 2, "lambda_n must be at least 2"
    assert cfg.lambda_spacing in ("log", "linear"), f"Unknown lambda_spacing: {cfg.lambda_spacing}"

    if cfg.lambda_spacing == "log":
        grid = np.logspace(np.log10(cfg.lambda_min), np.log10(cfg.lambda_max), cfg.lambda_n)
    else:
        grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_n)

    return grid[::-1].tolist()  # large to small


# ============================================================================
# Section 4: I/O
# ============================================================================


def build_output_paths(output_dir: str) -> OutputPaths:
    """Create output directory structure and return resolved paths."""
    def _mkdir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    return OutputPaths(
        root=_mkdir(output_dir),
        datasets_dir=_mkdir(os.path.join(output_dir, "datasets")),
        queries_dir=_mkdir(os.path.join(output_dir, "queries")),
        frontiers_dir=_mkdir(os.path.join(output_dir, "frontiers")),
        metadata_dir=_mkdir(os.path.join(output_dir, "metadata")),
        diagnostics_dir=_mkdir(os.path.join(output_dir, "diagnostics")),
        spline_params_dir=_mkdir(os.path.join(output_dir, "spline_params")),
    )


def _copy_datasets_if_requested(root_dir: str, output_paths: OutputPaths) -> None:
    """Bulk-copy the datasets subfolder from root_dir into the output tree once."""
    src = os.path.join(root_dir, "datasets")
    assert os.path.isdir(src), f"Expected datasets subfolder at {src}"
    shutil.copytree(src, output_paths.datasets_dir, dirs_exist_ok=True)


def write_queries_csv(
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write queries CSV for one dataset.

    Columns: query_id, source_row_index, a_query, x_0, x_1, ..., x_{d_x-1}
    """
    q_list = list(queries)
    x_stack = torch.stack([q.x_query for q in q_list], dim=0).detach().cpu().numpy()  # [n, d_x]

    columns: Dict[str, np.ndarray] = {
        "query_id": np.array([q.query_id for q in q_list], dtype=np.int64),
        "source_row_index": np.array([q.source_row_index for q in q_list], dtype=np.int64),
        "a_query": np.array([q.a_query for q in q_list], dtype=np.int64),
    }
    for i in range(x_stack.shape[1]):
        columns[f"x_{i}"] = x_stack[:, i]

    df = pd.DataFrame(columns)
    path = os.path.join(output_paths.queries_dir, f"queries_{dataset_id}.csv")
    df.to_csv(path, index=False)
    return path


def _flatten_dgp_results(
    all_results: Dict[str, List[SolveResult]],
    dgp_index: int,
    queries: Sequence[QuerySpec],
) -> Dict[str, np.ndarray]:
    """
    Flatten (bound_type, lambda, query) results for one DGP into 1D column arrays.
    """
    m = len(queries)
    query_ids = np.array([q.query_id for q in queries], dtype=np.int64)

    query_id_cols: List[np.ndarray] = []
    bound_type_cols: List[str] = []
    gamma_cols: List[np.ndarray] = []
    theta_cols: List[np.ndarray] = []

    for bound_type, results in all_results.items():
        for solve_result in results:
            assert solve_result.theta_star.shape[0] > dgp_index, (
                f"SolveResult theta_star has shape {solve_result.theta_star.shape}, "
                f"cannot index dgp {dgp_index}"
            )
            assert solve_result.theta_star.shape[1] == m
            theta_b = solve_result.theta_star[dgp_index, :]
            gamma_b = solve_result.gamma_star[dgp_index, :]
            query_id_cols.append(query_ids)
            bound_type_cols.extend([bound_type] * m)
            gamma_cols.append(gamma_b)
            theta_cols.append(theta_b)

    return {
        "query_id": np.concatenate(query_id_cols),
        "bound_type": np.array(bound_type_cols, dtype=object),
        "gamma_star": np.concatenate(gamma_cols),
        "theta_star": np.concatenate(theta_cols),
    }


def write_frontier_csv(
    all_results: Dict[str, List[SolveResult]],
    dgp_index: int,
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write frontier CSV for one dataset (training-facing handoff to Subsystem C).

    Columns: query_id, bound_type, gamma_star, theta_star
    """
    cols = _flatten_dgp_results(all_results, dgp_index, queries)
    df = pd.DataFrame(
        {
            "query_id": cols["query_id"],
            "bound_type": cols["bound_type"],
            "gamma_star": cols["gamma_star"],
            "theta_star": cols["theta_star"],
        }
    )
    path = os.path.join(output_paths.frontiers_dir, f"frontier_points_{dataset_id}.csv")
    df.to_csv(path, index=False)
    return path



def write_diagnostics(
    all_results: Dict[str, List[SolveResult]],
    dgp_index: int,
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write optional diagnostics JSON for one dataset. Compact JSON, heavy
    artifact (full optimization traces).

    This file is optimizer-facing, not FM-training-facing. It may include
    lambda provenance and final-eval scalarized objectives.
    """
    query_meta = [
        {
            "query_id": q.query_id,
            "source_row_index": q.source_row_index,
            "a_query": q.a_query,
        }
        for q in queries
    ]

    results_payload: List[Dict] = []
    for bound_type, results in all_results.items():
        for lambda_index, r in enumerate(results):
            theta_b = r.theta_star[dgp_index, :]
            gamma_b = r.gamma_star[dgp_index, :]
            objective_eval = _objective_from_theta_gamma(
                theta=theta_b,
                gamma=gamma_b,
                lambda_value=r.lambda_value,
                bound_type=bound_type,
            )

            initial_trace_objective = None
            final_trace_objective = None
            objective_gain = None
            if r.trace:
                initial_trace_objective = float(r.trace[0]["objective"])
                final_trace_objective = float(r.trace[-1]["objective"])
                objective_gain = final_trace_objective - initial_trace_objective

            results_payload.append({
                "lambda_index": int(lambda_index),
                "lambda_value": r.lambda_value,
                "bound_type": bound_type,
                "theta_mean": float(theta_b.mean()),
                "gamma_mean": float(gamma_b.mean()),
                "objective_eval_mean": float(objective_eval.mean()),
                "objective_eval_min": float(objective_eval.min()),
                "objective_eval_max": float(objective_eval.max()),
                "initial_trace_objective": initial_trace_objective,
                "final_trace_objective": final_trace_objective,
                "objective_gain": objective_gain,
                "converged": r.stop_reason != "nonfinite",
                "stop_reason": r.stop_reason,
                "num_steps": r.num_steps,
                "runtime_seconds": r.runtime_seconds,
                "learning_rate_base": r.learning_rate_base,
                "learning_rate_multiplier": r.learning_rate_multiplier,
                "learning_rate_effective": r.learning_rate_effective,
                "lr_lambda_schedule": r.lr_lambda_schedule,
                "max_steps_base": r.max_steps_base,
                "max_steps_effective": r.max_steps_effective,
                "max_steps_multiplier": r.max_steps_multiplier,
                "max_steps_lambda_schedule": r.max_steps_lambda_schedule,
                "trace": r.trace,
            })

    payload = {
        "dataset_id": dataset_id,
        "queries": query_meta,
        "solve_results": results_payload,
    }
    path = os.path.join(output_paths.diagnostics_dir, f"diagnostics_{dataset_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path

def write_frontier_diagnostics_csv(
    all_results: Dict[str, List[SolveResult]],
    dgp_index: int,
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write optimizer-facing per-query frontier diagnostics.

    This intentionally duplicates the FM-facing frontier points and adds
    lambda provenance. Do not use this file as the Subsystem C training
    interface unless the loader explicitly opts into diagnostic columns.
    """
    rows: List[Dict[str, object]] = []

    for bound_type, results in all_results.items():
        for lambda_index, r in enumerate(results):
            theta_b = r.theta_star[dgp_index, :]
            gamma_b = r.gamma_star[dgp_index, :]
            objective_b = _objective_from_theta_gamma(
                theta=theta_b,
                gamma=gamma_b,
                lambda_value=r.lambda_value,
                bound_type=bound_type,
            )

            for j, q in enumerate(queries):
                rows.append({
                    "dataset_id": int(dataset_id),
                    "query_id": int(q.query_id),
                    "bound_type": bound_type,
                    "lambda_index": int(lambda_index),
                    "lambda_value": float(r.lambda_value),
                    "gamma_star": float(gamma_b[j]),
                    "theta_star": float(theta_b[j]),
                    "signed_theta_star": float(_signed_theta(theta_b[j:j + 1], bound_type)[0]),
                    "objective_eval": float(objective_b[j]),
                    "source_row_index": int(q.source_row_index),
                    "a_query": int(q.a_query),
                    "stop_reason": r.stop_reason,
                    "num_steps": int(r.num_steps),
                    "runtime_seconds": float(r.runtime_seconds),
                    "learning_rate_base": float(r.learning_rate_base),
                    "learning_rate_multiplier": float(r.learning_rate_multiplier),
                    "learning_rate_effective": float(r.learning_rate_effective),
                    "lr_lambda_schedule": r.lr_lambda_schedule,
                    "max_steps_base": int(r.max_steps_base),
                    "max_steps_effective": int(r.max_steps_effective),
                    "max_steps_multiplier": float(r.max_steps_multiplier),
                    "max_steps_lambda_schedule": r.max_steps_lambda_schedule,
                })

    df = pd.DataFrame(rows)
    path = os.path.join(output_paths.diagnostics_dir, f"frontier_points_diagnostics_{dataset_id}.csv")
    df.to_csv(path, index=False)
    return path

def write_spline_params_npz(
    all_results: Dict[str, List[SolveResult]],
    dgp_index: int,
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write optional final spline parameters for one dataset.

    This is not part of the FM-facing interface. It is an optimizer diagnostic
    artifact that lets us re-evaluate solved frontiers with larger MC budgets
    without rerunning optimization.
    """
    assert "upper" in all_results and "lower" in all_results
    assert len(all_results["upper"]) == len(all_results["lower"])

    query_ids = np.array([q.query_id for q in queries], dtype=np.int64)
    source_row_indices = np.array([q.source_row_index for q in queries], dtype=np.int64)
    a_queries = np.array([q.a_query for q in queries], dtype=np.int64)

    payload: Dict[str, np.ndarray] = {
        "dataset_id": np.array([dataset_id], dtype=np.int64),
        "query_id": query_ids,
        "source_row_index": source_row_indices,
        "a_query": a_queries,
    }

    for bound_type, results in all_results.items():
        lambda_values = np.array([r.lambda_value for r in results], dtype=np.float64)

        spline_params = np.stack(
            [r.spline_params_final[dgp_index, :, :] for r in results],
            axis=0,
        ).astype(np.float32)

        theta_star = np.stack(
            [r.theta_star[dgp_index, :] for r in results],
            axis=0,
        ).astype(np.float32)

        gamma_star = np.stack(
            [r.gamma_star[dgp_index, :] for r in results],
            axis=0,
        ).astype(np.float32)

        payload[f"{bound_type}_lambda_value"] = lambda_values
        payload[f"{bound_type}_spline_params"] = spline_params
        payload[f"{bound_type}_theta_star"] = theta_star
        payload[f"{bound_type}_gamma_star"] = gamma_star

    path = os.path.join(output_paths.spline_params_dir, f"spline_params_{dataset_id}.npz")
    np.savez_compressed(path, **payload)
    return path


def write_run_metadata(
    config: PipelineConfig,
    output_paths: OutputPaths,
    lambda_grid: List[float],
) -> str:
    """Write run-level metadata JSON for reproducibility."""
    payload = {
        "root_dir": config.root_dir,
        "output_dir": config.output_dir,
        "lambda_grid": lambda_grid,
        "output_schema": {
            "datasets": "Optional copied normalized dataset CSVs.",
            "queries": "Per-dataset query definitions (query_id, x_query, a_query).",
            "frontiers": "Training-facing Pareto points (query_id, bound_type, gamma*, theta*).",
            "metadata": "Run-level config for reproducibility.",
            "diagnostics": "Optional heavy optimization traces and optimizer-facing CSVs.",
            "spline_params": "Optional final per-lambda spline parameters for high-MC re-evaluation.",
        },
        "config": asdict(config),
    }
    path = os.path.join(output_paths.metadata_dir, "frontier_run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser mapping to PipelineConfig fields.
    """
    parser = argparse.ArgumentParser(
        description="Subsystem B v4: batched Pareto frontier construction via direct per-query spline parameters."
    )
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Input artifact root (Subsystem A output, typically training_data_normalized).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output root for frontier artifacts.")
    parser.add_argument("--dataset-ids", type=int, nargs="*", default=None,
                        help="Optional subset of dataset ids to process.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for both model hosting and optimization.")
    parser.add_argument("--save-diagnostics", action="store_true",
                        help="Save heavy optimization diagnostics JSON.")
    parser.add_argument("--save-spline-params", action="store_true",
                        help="Save final per-lambda spline parameters for high-MC re-evaluation diagnostics.")
    parser.add_argument("--copy-dataset-csvs", action="store_true",
                        help="Copy normalized dataset CSVs into the frontier root.")
    parser.add_argument("--dgp-batch-size", type=int, default=8,
                        help="Number of DGPs processed jointly.")
    parser.add_argument("--sensitivity-model", type=str, default="msm",
                        choices=["msm", "kl", "rosenbaum"])
    parser.add_argument("--use-bucket-forward", type=_parse_bool, default=True,
                        help="Use depth-bucketed outcome-BNN forward; pass False for sequential fallback.")
    parser.add_argument("--warm-start-lambdas", type=_parse_bool, default=True,
                        help="Initialize each lambda solve from the previous lambda solution within the same bound sweep.")

    # Lambda grid config
    parser.add_argument("--lambda-min", type=float, required=True)
    parser.add_argument("--lambda-max", type=float, required=True)
    parser.add_argument("--lambda-n", type=int, default=30)
    parser.add_argument("--lambda-spacing", type=str, default="log",
                        choices=["log", "linear"])

    # Query config
    parser.add_argument("--num-query-points", type=int, default=16,
                        help="Number of query rows per dataset (before crossing with treatment arms).")
    parser.add_argument("--query-sampling-mode", type=str, default="ordered_proxy",
                        choices=["random", "ordered_proxy", "stratified_proxy"])
    parser.add_argument("--query-random-seed", type=int, default=0)

    # Flow config
    parser.add_argument("--flow-num-bins", type=int, default=32)
    parser.add_argument("--flow-tail-bound", type=float, default=6.0)
    parser.add_argument("--flow-min-bin-width", type=float, default=1e-3)
    parser.add_argument("--flow-min-bin-height", type=float, default=1e-3)
    parser.add_argument("--flow-min-derivative", type=float, default=1e-3)

    # Solver config
    parser.add_argument("--num-mc-samples-train", type=int, default=128)
    parser.add_argument("--num-mc-samples-eval", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-lambda-schedule", type=str, default="none",
                        choices=["none", "sqrt"],
                        help="Lambda-dependent learning-rate schedule.")
    parser.add_argument("--lr-lambda-ref", type=float, default=0.5,
                        help="Reference lambda for LR schedule saturation.")
    parser.add_argument("--lr-lambda-min-mult", type=float, default=0.25,
                        help="Minimum lambda-dependent LR multiplier.")
    parser.add_argument("--max-steps-lambda-schedule", type=str, default="none",
                        choices=["none", "inverse_sqrt_lr"],
                        help="Lambda-dependent max-step schedule.")
    parser.add_argument("--max-steps-lambda-max-mult", type=float, default=2.0,
                        help="Maximum lambda-dependent max-step multiplier.")
    parser.add_argument("--loss-reduction", type=str, default="per_dgp_sum",
                        choices=["batch_mean", "per_dgp_sum"],
                        help="Scalar loss reduction for optimizer trajectory experiments.")
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable conservative adaptive early stopping based on objective plateau.")
    parser.add_argument("--early-stop-min-steps", type=int, default=75)
    parser.add_argument("--early-stop-check-every", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-abs-tol", type=float, default=1e-4)
    parser.add_argument("--early-stop-rel-tol", type=float, default=1e-4)

    # Profiling config
    parser.add_argument("--runtime-profile", action="store_true",
                        help="Collect manual scoped runtime timings.")
    parser.add_argument("--runtime-profile-first-n-solves", type=int, default=0,
                        help="Number of optimize_lambda_batch calls to manually profile; 0 profiles all solves.")
    parser.add_argument("--runtime-profile-solve-indices", type=str, default=None,
                        help="Comma-separated global solve indices to manually profile; overrides first-N behavior.")
    parser.add_argument("--runtime-profile-sync-cuda", action="store_true",
                        help="Synchronize CUDA before and after manual timed blocks.")
    parser.add_argument("--runtime-profile-output-dir", type=str, default=None,
                        help="Directory for manual runtime profiling CSV/JSON outputs.")
    parser.add_argument("--torch-profile", action="store_true",
                        help="Enable torch.profiler for the first selected solves.")
    parser.add_argument("--torch-profile-dir", type=str, default=None,
                        help="Directory for torch profiler TensorBoard traces.")
    parser.add_argument("--torch-profile-first-n-solves", type=int, default=1,
                        help="Number of optimize_lambda_batch calls to profile with torch.profiler.")
    parser.add_argument("--torch-profile-wait", type=int, default=5)
    parser.add_argument("--torch-profile-warmup", type=int, default=5)
    parser.add_argument("--torch-profile-active", type=int, default=10)
    parser.add_argument("--torch-profile-repeat", type=int, default=1)

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct PipelineConfig from parsed CLI arguments."""
    return PipelineConfig(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        dataset_ids=args.dataset_ids,
        sensitivity_model=args.sensitivity_model,
        device=args.device,
        save_diagnostics=args.save_diagnostics,
        save_spline_params=args.save_spline_params,
        copy_dataset_csvs=args.copy_dataset_csvs,
        dgp_batch_size=args.dgp_batch_size,
        use_bucket_forward=args.use_bucket_forward,
        warm_start_lambdas=args.warm_start_lambdas,
        flow=FlowConfig(
            num_bins=args.flow_num_bins,
            tail_bound=args.flow_tail_bound,
            min_bin_width=args.flow_min_bin_width,
            min_bin_height=args.flow_min_bin_height,
            min_derivative=args.flow_min_derivative,
        ),
        solver=SolverConfig(
            num_mc_samples_train=args.num_mc_samples_train,
            num_mc_samples_eval=args.num_mc_samples_eval,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_lambda_schedule=args.lr_lambda_schedule,
            lr_lambda_ref=args.lr_lambda_ref,
            lr_lambda_min_mult=args.lr_lambda_min_mult,
            max_steps_lambda_schedule=args.max_steps_lambda_schedule,
            max_steps_lambda_max_mult=args.max_steps_lambda_max_mult,
            loss_reduction=args.loss_reduction,
            early_stop=args.early_stop,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_check_every=args.early_stop_check_every,
            early_stop_patience=args.early_stop_patience,
            early_stop_abs_tol=args.early_stop_abs_tol,
            early_stop_rel_tol=args.early_stop_rel_tol,
        ),
        query=QueryConfig(
            num_query_points=args.num_query_points,
            sampling_mode=args.query_sampling_mode,
            random_seed=args.query_random_seed,
        ),
        lambda_grid_cfg=LambdaGridConfig(
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            lambda_n=args.lambda_n,
            lambda_spacing=args.lambda_spacing,
        ),
        profiling=ProfilingConfig(
            runtime_profile=args.runtime_profile,
            runtime_profile_first_n_solves=args.runtime_profile_first_n_solves,
            runtime_profile_solve_indices=_parse_int_csv(args.runtime_profile_solve_indices),
            runtime_profile_sync_cuda=args.runtime_profile_sync_cuda,
            runtime_profile_output_dir=args.runtime_profile_output_dir,
            torch_profile=args.torch_profile,
            torch_profile_dir=args.torch_profile_dir,
            torch_profile_first_n_solves=args.torch_profile_first_n_solves,
            torch_profile_wait=args.torch_profile_wait,
            torch_profile_warmup=args.torch_profile_warmup,
            torch_profile_active=args.torch_profile_active,
            torch_profile_repeat=args.torch_profile_repeat,
        ),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
