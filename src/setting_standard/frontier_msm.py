"""
Subsystem B: analytical standard binary MSM frontier construction for partial identification.

Given frozen outcome BNNs (f_BNN) from Subsystem A, evaluates the
closed-form empirical propensity-aware MSM optimum on a Gamma grid to produce
(theta*, Gamma*) training pairs for the foundation model (Subsystem C).

Script layout:
    0. Configs & small types
    1. Analytical MSM compute  - pure computation, no state management
    2. Solver                  - one Gamma value, no I/O
    3. Pipeline                - loop nesting and orchestration
    4. I/O                     - serialization, deserialization, CLI

Dependency direction: 1 <- 2 <- 3 -> 4. Sections 1 and 4 are independent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


from setting_standard.causaldataset import CausalFrontierDataset, resolve_code_relative_path
from setting_standard.fork_run_dir import fork_upstream
from setting_standard.run_manifest import record_pipeline_step


# ============================================================================
# Section 0: Configs, small types, utilities
# ============================================================================


@dataclass
class SolverConfig:
    """Hyperparameters for the analytical MSM evaluation."""
    num_mc_samples: int = 128


@dataclass
class QueryConfig:
    """Controls for query point selection within each dataset."""
    num_query_points: int = 16
    sampling_mode: str = "ordered_proxy"       # "random" | "ordered_proxy" | "stratified_proxy"
    random_seed: int = 0
    treatment_arms: Tuple[int, ...] = (0, 1)


@dataclass
class GammaGridConfig:
    """Controls for gamma grid construction. The pipeline expects an explicit list"""
    gamma_min: float
    gamma_max: float
    gamma_n: int = 30
    gamma_spacing: str = "log"   # "log" | "linear" | "bounded_pareto"
    # Bounded Pareto parameters
    pareto_beta: float = 1.0
    pareto_seed: int = 42


@dataclass
class PipelineConfig:
    """
    Top-level config assembling all sub-configs plus pipeline-level settings.

    Sections 1-2 only receive the sub-configs and explicit tensors they need.
    Only section 3 (pipeline) and section 4 (I/O) use PipelineConfig directly.
    """
    run_dir: str
    dataset_ids: Optional[List[int]] = None          # None = process all
    sensitivity_model: str = "msm"                   # MSM only
    estimand_type: str = "capo"                      # CAPO only
    device: str = "cpu"
    num_workers: Optional[int] = None
    save_diagnostics: bool = False
    source_run_dir: Optional[str] = None

    solver: SolverConfig = field(default_factory=SolverConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    gamma_grid_cfg: GammaGridConfig = field(default_factory=GammaGridConfig)


@dataclass
class QuerySpec:
    """One query point for frontier construction. Lightweight, no metadata."""
    query_id: int
    source_row_index: int
    a_query: int
    x_query: torch.Tensor                            # shape [x_dim]


@dataclass
class QuerySampleBank:
    """
    Shared Monte Carlo sample bank for one query.

    Reused across all Gamma values and both bound types for the same
    (x_query, a_query) problem.
    """
    u_samples: torch.Tensor                          # shape [k, 1], on CPU
    y_samples: torch.Tensor                          # shape [k], on CPU


@dataclass
class OrderedOutcomeCache:
    """
    Precomputed ordering metadata for one bound direction on a fixed sample bank.

    This lets the Gamma sweep reuse the same sorted outcomes and tie structure
    instead of rebuilding them for every Gamma value.
    """
    sorted_y_samples: np.ndarray                     # shape [k], on CPU
    prefix_sums: np.ndarray                          # shape [k], on CPU
    tie_group_starts: np.ndarray                     # shape [k], on CPU
    tie_group_counts: np.ndarray                     # shape [k], on CPU
    total_sum: float


@dataclass
class SolveResult:
    """
    Return value of solve_single_gamma_analytical.

    Contains the training-facing Pareto point plus analytical diagnostics.
    """
    gamma_star: float                                # target Gamma used for this solve
    theta_star: float                                # analytical CAPO estimate under empirical optimum
    empirical_gamma_hat: float                       # implied Gamma from propensity-aware realized weight ratio
    pi: float                                        # query propensity pi(a_query | x_query)
    quantile_level: float
    threshold: float
    selected_count: int
    tail_mass_target: float
    tail_mass_realized: float
    tail_mass_gap: float
    threshold_tie_count: int
    selected_at_threshold_count: int
    runtime_seconds: float
    mc_sample_count: int
    trace: List[Dict[str, float]]


@dataclass
class FrontierRow:
    """
    Schema documentation for frontier CSV. Not constructed at runtime after vectorization.
    """
    query_id: int
    bound_type: str                                  # "upper" | "lower"
    gamma_star: float
    theta_star: float


@dataclass
class SweepResult:
    """Raw output of one (query, bound_type) gamma sweep. Holds vectorized batch arrays."""
    query_id: int
    bound_type: str                  # "upper" | "lower"
    batch: Dict[str, np.ndarray]     # arrays of shape [G], keyed as in compute_analytical_msm_weights_vectorized


@dataclass
class OutputPaths:
    """Resolved directory paths for all output artifacts."""
    root: str
    datasets_dir: str
    queries_dir: str
    frontiers_raw_dir: str
    metadata_frontier_dir: str
    diagnostics_frontier_dir: str


def _all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _query_sample_seed(dataset_id: int, query_id: int, base_seed: int) -> int:
    return int(base_seed + 1009 * dataset_id + 1000003 * query_id)


def get_worker_count(max_fraction: float = 0.5, min_free: int = 4) -> int:
    total = os.cpu_count() or 1
    return max(1, min(total - min_free, int(total * max_fraction)))


def _noop_log(_: str) -> None:
    return None


def _init_worker() -> None:
    torch.set_num_threads(1)
    if not hasattr(os, "nice"):
        return
    try:
        os.nice(5)
    except OSError:
        pass


# ============================================================================
# Section 1: Analytical MSM compute
# ============================================================================


def compute_query_ordering_scores(X: torch.Tensor) -> torch.Tensor:
    """
    Cheap ordering proxy for deterministic subsampling order.
    Linear score on features. Returns shape [n_rows].
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
) -> List[QuerySpec]:
    """
    Build full list of QuerySpecs for one dataset.

    Selects rows via select_query_row_indices, crosses with treatment arms.
    """
    selected_rows = select_query_row_indices(X=X, config=config)
    query_specs: List[QuerySpec] = []
    query_id = 0
    for row_index in selected_rows:
        x_query = X[row_index].clone()
        for a_query in config.treatment_arms:
            query_specs.append(
                QuerySpec(
                    query_id=query_id,
                    source_row_index=row_index,
                    a_query=int(a_query),
                    x_query=x_query,
                )
            )
            query_id += 1
    return query_specs


def evaluate_capo_outcomes(
    f_bnn: nn.Module,
    x_query: torch.Tensor,            # shape [x_dim]
    a_query: int,
    u_samples: torch.Tensor,          # shape [num_samples, 1]
    device: str,
) -> torch.Tensor:
    """
    Evaluate f_BNN(x_query, a_query, u) on a shared Monte Carlo sample bank.

    Returns a 1D tensor of shape [num_samples] on CPU.
    """
    assert u_samples.dim() == 2 and u_samples.shape[1] == 1
    x_query = x_query.to(device=device, dtype=torch.float32)
    u_samples = u_samples.to(device=device, dtype=torch.float32)
    x_batch = x_query.unsqueeze(0).expand(u_samples.shape[0], -1)
    a_batch = torch.full((u_samples.shape[0], 1), float(a_query), dtype=torch.float32, device=device)
    y_samples = f_bnn(x_batch, a_batch, u_samples).reshape(-1)
    assert _all_finite(y_samples), "Encountered non-finite y_samples from f_BNN"
    return y_samples.detach().cpu()


def build_query_sample_bank(
    f_bnn: nn.Module,
    query: QuerySpec,
    config: SolverConfig,
    device: str,
    seed: int,
) -> QuerySampleBank:
    """
    Draw one latent Monte Carlo bank for a query and evaluate outcomes once.

    This bank is reused across all Gamma values and both bound types.
    """
    assert config.num_mc_samples >= 2, "Analytical MSM needs at least two MC samples"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    u_samples = torch.randn(config.num_mc_samples, 1, generator=generator)
    y_samples = evaluate_capo_outcomes(
        f_bnn=f_bnn,
        x_query=query.x_query,
        a_query=query.a_query,
        u_samples=u_samples,
        device=device,
    )
    return QuerySampleBank(u_samples=u_samples, y_samples=y_samples)


def build_ordered_outcome_cache(
    y_samples: torch.Tensor,
    bound_type: str,
) -> OrderedOutcomeCache:
    """
    Precompute bound-specific ordering metadata for a fixed Monte Carlo bank.

    The Gamma sweep only changes how many ordered samples receive the large
    weight. The sorted outcomes, prefix sums, and tie-group structure are
    invariant across the entire sweep and should only be built once.
    """
    assert y_samples.dim() == 1, "Expected y_samples with shape [k]"
    assert _all_finite(y_samples), "Expected finite y_samples"
    assert bound_type in ("upper", "lower"), f"Unsupported bound_type: {bound_type}"

    descending = bound_type == "upper"
    y_np = y_samples.detach().cpu().numpy()
    sorted_y_samples = np.sort(y_np)
    if descending:
        sorted_y_samples = sorted_y_samples[::-1].copy()

    prefix_sums = np.cumsum(sorted_y_samples, dtype=np.float64)
    _, tie_counts = np.unique(sorted_y_samples, return_counts=True)
    tie_starts = np.cumsum(tie_counts, dtype=np.int64) - tie_counts
    tie_group_starts = np.repeat(tie_starts, tie_counts)
    tie_group_counts = np.repeat(tie_counts, tie_counts)

    return OrderedOutcomeCache(
        sorted_y_samples=sorted_y_samples,
        prefix_sums=prefix_sums,
        tie_group_starts=tie_group_starts,
        tie_group_counts=tie_group_counts,
        total_sum=float(prefix_sums[-1]),
    )


def compute_analytical_msm_weights_vectorized(
    ordered_cache: OrderedOutcomeCache,
    gamma_grid: List[float],
    bound_type: str,
    pi: float,
) -> Dict[str, np.ndarray]:
    """
    Vectorized analytical MSM weights across all gamma values simultaneously.

    All returned arrays have shape [G] where G = len(gamma_grid).
    Sort direction (upper/lower) is already baked into ordered_cache.
    """
    assert bound_type in ("upper", "lower")
    assert 0.0 <= pi <= 1.0, f"Propensity pi={pi} out of bounds for gamma_grid={gamma_grid}"

    gammas = np.array(gamma_grid, dtype=np.float64)           # [G]
    assert np.all(gammas >= 1.0), f"Expected MSM Gamma >= 1 in gamma_grid={gamma_grid}"
    num_samples = ordered_cache.sorted_y_samples.shape[0]
    prefix_sums = ordered_cache.prefix_sums                   # [k]
    sorted_y    = ordered_cache.sorted_y_samples.astype(np.float64, copy=False)  # [k]
    tie_starts  = ordered_cache.tie_group_starts              # [k]
    tie_counts  = ordered_cache.tie_group_counts              # [k]
    total_sum   = float(ordered_cache.total_sum)

    # Nearest admissible tail count approximating mass 1 / (Gamma + 1), shape [G].
    m = np.clip(np.round(num_samples / (gammas + 1.0)).astype(int), 1, num_samples - 1)

    # prefix-sum lookups, shape [G]
    selected_sum   = prefix_sums[m - 1]
    complement_sum = total_sum - selected_sum

    # Propensity-aware raw reciprocal scales, shape [G].
    # selected_weight applies to the extreme tail of size m.
    inv_s_plus  = (1.0 - 1.0 / gammas) * pi + 1.0 / gammas
    inv_s_minus = (1.0 - gammas) * pi + gammas

    # normalized weights, shape [G]
    raw_weight_sum    = m * inv_s_minus + (num_samples - m) * inv_s_plus
    selected_weight   = inv_s_minus / raw_weight_sum
    complement_weight = inv_s_plus / raw_weight_sum

    theta_star = selected_weight * selected_sum + complement_weight * complement_sum
    # No sensitivity (Gamma=1) or a fully observed arm (pi=1) identifies its mean.
    identified = np.ones_like(gammas, dtype=bool) if pi == 1.0 else gammas == 1.0
    empirical_mean = ordered_cache.sorted_y_samples.mean(dtype=np.float64)
    assert np.allclose(theta_star[identified], empirical_mean, rtol=1e-10, atol=1e-10), (
        "MSM identified bounds must equal the observational outcome mean"
    )

    # diagnostics, shape [G]
    threshold                  = sorted_y[m - 1]
    threshold_group_start      = tie_starts[m - 1]
    threshold_tie_count        = tie_counts[m - 1]
    selected_at_threshold      = m - threshold_group_start
    empirical_gamma_hat = _empirical_gamma_hat(
        selected_weight=selected_weight,
        complement_weight=complement_weight,
        pi=pi,
    )
    quantile_level   = gammas / (gammas + 1.0) if bound_type == "upper" else 1.0 / (gammas + 1.0)
    tail_mass_target = 1.0 / (gammas + 1.0)
    tail_mass_realized = m / num_samples

    return {
        "theta_star":                   theta_star,
        "selected_count":               m,
        "threshold":                    threshold,
        "quantile_level":               quantile_level,
        "tail_mass_target":             tail_mass_target,
        "tail_mass_realized":           tail_mass_realized,
        "tail_mass_gap":                tail_mass_realized - tail_mass_target,
        "threshold_tie_count":          threshold_tie_count,
        "selected_at_threshold_count":  selected_at_threshold,
        "empirical_gamma_hat":          empirical_gamma_hat,
        "pi":                           np.full_like(gammas, pi, dtype=np.float64),
        "selected_weight":              selected_weight,
        "complement_weight":            complement_weight,
        "inv_s_plus":                   inv_s_plus,
        "inv_s_minus":                  inv_s_minus,
    }


def _empirical_gamma_hat(
    selected_weight: np.ndarray,
    complement_weight: np.ndarray,
    pi: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Implied Gamma from realized propensity-aware weight ratio.

    Solves rho(Gamma; pi) = R for Gamma >= 1 via the closed-form quadratic.
    Returns NaN at pi == 1, where rho is identically 1 for any Gamma.
    """
    R = selected_weight / np.maximum(complement_weight, eps)

    if pi >= 1.0 - eps:
        return np.full_like(R, np.nan)
    if pi <= eps:
        return np.sqrt(np.maximum(R, 0.0))

    a = 1.0 - pi
    b = pi * (1.0 - R)
    c = -R * (1.0 - pi)

    discriminant = b * b - 4.0 * a * c
    assert np.all(discriminant >= -1e-9), (
        "Negative discriminant in empirical_gamma_hat solve; "
        "check propensity and weight ratio inputs."
    )
    discriminant = np.maximum(discriminant, 0.0)
    return (-b + np.sqrt(discriminant)) / (2.0 * a)

def _sample_bounded_pareto(
    n: int,
    low: float,
    high: float,
    beta: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Sample n points from Bounded Pareto(low, high, beta) via inverse CDF.

    CDF: F(x) = 1 - (low/x)^beta * (1 - (low/high)^beta)^{-1}  ... scaled to [low, high]
    Inverse CDF: x = low * (1 - u * (1 - (low/high)^beta))^{-1/beta}
    """
    assert beta >= 0.0, f"Expected beta >= 0, got {beta}"
    assert low > 0.0, "Expected low > 0 for log-uniform limiting case"
    assert high > low, f"Expected high > low, got {high} <= {low}"

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n)

    if beta == 0.0:
        # Log-uniform: uniform on [log(low), log(high)], exponentiated
        log_low, log_high = np.log(low), np.log(high)
        samples = np.exp(log_low + u * (log_high - log_low))
    else:
        L_beta = low ** beta
        H_beta = high ** beta
        # Standard bounded Pareto inverse CDF
        samples = (-(u * H_beta - u * L_beta - H_beta) / (L_beta * H_beta)) ** (-1.0 / beta)
    return np.clip(samples, low, high)

# ============================================================================
# Section 2: Solver
# ============================================================================

# Scalar helper for diagnostics; the pipeline uses the vectorized sweep.
def solve_single_gamma_analytical(
    ordered_cache: OrderedOutcomeCache,
    gamma_value: float,
    bound_type: str,
    sensitivity_model: str,
    pi: float,
) -> SolveResult:
    assert sensitivity_model == "msm"
    batch = compute_analytical_msm_weights_vectorized(ordered_cache, [gamma_value], bound_type, pi)
    return SolveResult(
        gamma_star=float(gamma_value),
        theta_star=float(batch["theta_star"][0]),
        empirical_gamma_hat=float(batch["empirical_gamma_hat"][0]),
        pi=float(batch["pi"][0]),
        quantile_level=float(batch["quantile_level"][0]),
        threshold=float(batch["threshold"][0]),
        selected_count=int(batch["selected_count"][0]),
        tail_mass_target=float(batch["tail_mass_target"][0]),
        tail_mass_realized=float(batch["tail_mass_realized"][0]),
        tail_mass_gap=float(batch["tail_mass_gap"][0]),
        threshold_tie_count=int(batch["threshold_tie_count"][0]),
        selected_at_threshold_count=int(batch["selected_at_threshold_count"][0]),
        runtime_seconds=0.0,
        mc_sample_count=ordered_cache.sorted_y_samples.shape[0],
        trace=[],
    )


# ============================================================================
# Section 3: Pipeline
# ============================================================================


def sweep_gamma_grid(
    sample_bank: QuerySampleBank,
    query: QuerySpec,
    bound_type: str,
    gamma_grid: List[float],
    sensitivity_model: str,
    pi: float,
) -> SweepResult:
    """
    Sweep all Gamma values for one (query, bound_type) problem.

    Gamma loop is fully vectorized: one batch call replaces 50 serial solves.
    The raw batch arrays are returned and flattened later by the I/O layer.
    """
    assert sensitivity_model == "msm", f"Unsupported sensitivity_model: {sensitivity_model}"

    ordered_cache = build_ordered_outcome_cache(sample_bank.y_samples, bound_type)
    batch = compute_analytical_msm_weights_vectorized(ordered_cache, gamma_grid, bound_type, pi)
    return SweepResult(
        query_id=query.query_id,
        bound_type=bound_type,
        batch=batch,
    )

def process_single_dataset(
    dataset_record: Dict,
    config: PipelineConfig,
    output_paths: OutputPaths,
    log_fn: Callable[[str], None] = tqdm.write,
) -> None:
    """
    Full analytical frontier construction for one dataset.

    Loop order stays explicit:
        - build query specs
        - build one shared sample bank per query
        - sweep Gamma values separately for upper and lower bounds
    """
    dataset_id = dataset_record["dataset_id"]
    X = dataset_record["X"]
    f_bnn = dataset_record["f_BNN"].to(config.device)
    f_bnn.eval()
    f_A = dataset_record["f_A"].to(config.device)
    f_A.eval()
    for p in f_A.parameters():
        p.requires_grad_(False)

    gamma_grid = make_gamma_grid(config.gamma_grid_cfg, dataset_id=dataset_id)
    assert len(gamma_grid) > 0, "gamma_grid must be non-empty"
    assert all(gamma >= 1.0 for gamma in gamma_grid), "MSM gamma_grid values must be at least 1"
    assert gamma_grid == sorted(gamma_grid), "gamma_grid must be ordered small to large"


    queries = build_query_specs(X=X, config=config.query)
    query_propensities: Dict[int, float] = {}
    for query in queries:
        x_q = query.x_query.to(device=config.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob_a1 = float(f_A(x_q).reshape(()).item())
        pi = prob_a1 if query.a_query == 1 else 1.0 - prob_a1
        assert 0.0 < pi < 1.0, (
            f"Propensity out of bounds for dataset_id={dataset_id} "
            f"query_id={query.query_id}: pi={pi}"
        )
        query_propensities[query.query_id] = pi

    query_sample_banks: Dict[int, QuerySampleBank] = {}
    for query in queries:
        seed = _query_sample_seed(
            dataset_id=dataset_id,
            query_id=query.query_id,
            base_seed=config.query.random_seed,
        )
        query_sample_banks[query.query_id] = build_query_sample_bank(
            f_bnn=f_bnn,
            query=query,
            config=config.solver,
            device=config.device,
            seed=seed,
        )

    sweep_results: List[SweepResult] = []

    num_solves = len(queries) * 2 * len(gamma_grid)
    log_fn(
        f"Dataset {dataset_id}: {len(queries)} queries x 2 bounds x "
        f"{len(gamma_grid)} gammas = {num_solves} analytical evaluations"
    )

    for bound_type in ("upper", "lower"):
        for query in queries:
            sweep_results.append(sweep_gamma_grid(
                sample_bank=query_sample_banks[query.query_id],
                query=query,
                bound_type=bound_type,
                gamma_grid=gamma_grid,
                sensitivity_model=config.sensitivity_model,
                pi=query_propensities[query.query_id],
            ))

    write_queries_csv(queries, output_paths, dataset_id)
    write_frontier_csv(sweep_results, gamma_grid, output_paths, dataset_id)
    if config.save_diagnostics:
        write_diagnostics(sweep_results, gamma_grid, queries, output_paths, dataset_id)


def _process_dataset_worker(
    idx: int,
    config: PipelineConfig,
    output_paths: OutputPaths,
) -> Dict[str, object]:
    start_time = time.perf_counter()
    dataset = CausalFrontierDataset(
        root_dir=config.run_dir,
        model_device="cpu",
        log_fn=_noop_log,
    )
    artifact = dataset.artifacts[idx]
    dataset_id = artifact["dataset_id"]

    try:
        record = dataset[idx]
        record["dataset_path"] = artifact.get("dataset_path", "")
        process_single_dataset(record, config, output_paths, log_fn=_noop_log)
        return {
            "success": True,
            "idx": idx,
            "dataset_id": dataset_id,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    except Exception as exc:
        return {
            "success": False,
            "idx": idx,
            "dataset_id": dataset_id,
            "elapsed_seconds": time.perf_counter() - start_time,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    """Execute analytical frontier construction in an already-prepared run."""
    run_dir = resolve_code_relative_path(config.run_dir)
    output_paths = build_output_paths(run_dir)
    write_run_metadata(config, output_paths)

    dataset = CausalFrontierDataset(root_dir=run_dir, model_device="cpu", log_fn=lambda msg: None)

    # Determine dataset indices to process
    if config.dataset_ids is None:
        indices = list(range(len(dataset)))
    else:
        id_to_idx = {a["dataset_id"]: i for i, a in enumerate(dataset.artifacts)}
        missing = [d for d in config.dataset_ids if d not in id_to_idx]
        assert not missing, f"Requested dataset ids not found: {missing}"
        indices = [id_to_idx[d] for d in config.dataset_ids]

    tqdm.write(
        f"Analytical MSM run: {len(indices)} datasets | "
        f"{config.query.num_query_points} query rows x {len(config.query.treatment_arms)} arms | "
        f"{config.gamma_grid_cfg.gamma_n} gammas | k={config.solver.num_mc_samples}"
    )

    # Get worker count
    max_sensible_workers = len(indices)
    max_allowed_workers = get_worker_count()
    requested_workers = max_allowed_workers if config.num_workers is None else config.num_workers
    effective_workers = min(max_sensible_workers, requested_workers, max_allowed_workers)
    tqdm.write(f"Using {effective_workers} worker(s) for dataset processing")

    failures: List[str] = []
    # Serial processing
    if effective_workers <= 1:
        for idx in tqdm(indices, desc="Analytical Frontier Datasets"):
            artifact = dataset.artifacts[idx]
            dataset_id = artifact["dataset_id"]
            try:
                record = dataset[idx]
                record["dataset_path"] = artifact.get("dataset_path", "")
                process_single_dataset(record, config, output_paths, log_fn=tqdm.write)
            except Exception as exc:
                failures.append(f"dataset_id={dataset_id} idx={idx}: {type(exc).__name__}: {exc}")
    # Parallel processing 
    else:
        with ProcessPoolExecutor(max_workers=effective_workers, initializer=_init_worker) as executor:
            futures = {
                executor.submit(_process_dataset_worker, idx, config, output_paths): idx
                for idx in indices
            }
            with tqdm(total=len(indices), desc="Analytical Frontier Datasets") as progress_bar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append(f"dataset idx={idx}: {type(exc).__name__}: {exc}")
                    else:
                        if not result["success"]:
                            failures.append(
                                f"dataset_id={result['dataset_id']} idx={result['idx']}: {result['error']}"
                            )
                    progress_bar.update(1)

    if failures:
        summary = "\n".join(failures)
        raise RuntimeError(f"Analytical MSM pipeline finished with {len(failures)} dataset failure(s):\n{summary}")

    requested_dataset_ids = [dataset.artifacts[idx]["dataset_id"] for idx in indices]
    missing_outputs = []
    for dataset_id in requested_dataset_ids:
        required_paths = [
            os.path.join(output_paths.queries_dir, f"queries_{dataset_id}.csv"),
            os.path.join(output_paths.frontiers_raw_dir, f"frontier_points_{dataset_id}.csv"),
        ]
        if config.save_diagnostics:
            required_paths.append(
                os.path.join(output_paths.diagnostics_frontier_dir, f"diagnostics_{dataset_id}.json")
            )
        if not all(os.path.isfile(path) for path in required_paths):
            missing_outputs.append(dataset_id)
    if missing_outputs:
        raise RuntimeError(
            f"Analytical MSM frontier is missing configured outputs for DGPs: {missing_outputs}"
        )
    return {
        "requested_datasets": len(requested_dataset_ids),
        "completed_datasets": len(requested_dataset_ids),
        "failed_datasets": 0,
        "dataset_ids": requested_dataset_ids,
    }


def run_pipeline(config: PipelineConfig) -> None:
    """Prepare the run and record one analytical-frontier attempt."""
    assert config.estimand_type == "capo", "Analytical MSM supports CAPO only"
    assert config.sensitivity_model == "msm", "Analytical frontier supports MSM only"
    assert config.device == "cpu", "CPU-only execution is supported in this script; pass --device cpu"
    assert config.num_workers is None or config.num_workers >= 1, "--num-workers must be at least 1"

    _seed_everything(config.query.random_seed)
    run_dir = resolve_code_relative_path(config.run_dir)
    if config.source_run_dir is not None:
        fork_upstream(
            source_run_dir=config.source_run_dir,
            dest_run_dir=run_dir,
            step=3,
            dataset_ids=config.dataset_ids,
        )

    seeds = {
        "uses_randomness": True,
        "query_random_seed": config.query.random_seed,
        "mc_seed_rule": (
            "query_random_seed + 1009 * dataset_id + 1000003 * query_id"
        ),
        "pareto_seed": config.gamma_grid_cfg.pareto_seed,
        "pareto_seed_rule": (
            "fixed pareto_seed when set; otherwise dataset_id"
        ),
    }
    with record_pipeline_step(
        run_dir=run_dir,
        step="03_frontier",
        implementation="src/setting_standard/frontier_msm.py",
        resolved_arguments=asdict(config),
        seeds=seeds,
    ) as attempt:
        completion = _run_pipeline(config)
        attempt.set_completion(completion)


def make_gamma_grid(cfg: GammaGridConfig, dataset_id: int = 0) -> List[float]:
    """Returns gamma values ordered small -> large (permissive -> tight bounds)."""
    assert cfg.gamma_min >= 1.0, "gamma_min must be at least 1"
    assert cfg.gamma_max > cfg.gamma_min, "gamma_max must exceed gamma_min"
    assert cfg.gamma_n >= 2, "gamma_n must be at least 2"
    assert cfg.gamma_spacing in ("log", "linear", "bounded_pareto"), f"Unknown gamma_spacing: {cfg.gamma_spacing}"

    if cfg.gamma_spacing == "log":
        grid = np.logspace(np.log10(cfg.gamma_min), np.log10(cfg.gamma_max), cfg.gamma_n)
    elif cfg.gamma_spacing == "bounded_pareto":
        seed = cfg.pareto_seed if cfg.pareto_seed is not None else dataset_id
        grid = _sample_bounded_pareto(
            n=cfg.gamma_n,
            low=cfg.gamma_min,
            high=cfg.gamma_max,
            beta=cfg.pareto_beta,
            seed=seed,
        )
    elif cfg.gamma_spacing == "linear":
        grid = np.linspace(cfg.gamma_min, cfg.gamma_max, cfg.gamma_n)
    else:
        raise ValueError(f"Unsupported gamma_spacing: {cfg.gamma_spacing}")
    
    return sorted(grid.tolist())  # small to large



# ============================================================================
# Section 4: I/O
# ============================================================================


def build_output_paths(run_dir: str) -> OutputPaths:
    """Create output directory structure and return resolved paths.

    Input guard: datasets/ must already exist under run_dir (step 2 output).

    Output guards: queries/, frontiers_raw/, and metadata_frontier/ must NOT exist yet.

    diagnostics_frontier/ is created lazily when first written to.
    """
    assert os.path.isdir(os.path.join(run_dir, "datasets")), \
        f"Expected datasets/ under run_dir but not found: {os.path.join(run_dir, 'datasets')}"
    assert os.path.isdir(os.path.join(run_dir, "models")), \
        f"Expected models/ under run_dir but not found: {os.path.join(run_dir, 'models')}"
    assert os.path.isdir(os.path.join(run_dir, "metadata")), \
        f"Expected metadata/ under run_dir but not found: {os.path.join(run_dir, 'metadata')}"
    assert not os.path.exists(os.path.join(run_dir, "queries")), \
        f"queries/ already exists under {run_dir}; this script does not support resume"
    assert not os.path.exists(os.path.join(run_dir, "frontiers_raw")), \
        f"frontiers_raw/ already exists under {run_dir}; this script does not support resume"
    assert not os.path.exists(os.path.join(run_dir, "metadata_frontier")), \
        f"metadata_frontier/ already exists under {run_dir}; this script does not support resume"

    def _mkdir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    return OutputPaths(
        root=run_dir,
        datasets_dir=os.path.join(run_dir, "datasets"),
        queries_dir=_mkdir(os.path.join(run_dir, "queries")),
        frontiers_raw_dir=_mkdir(os.path.join(run_dir, "frontiers_raw")),
        metadata_frontier_dir=_mkdir(os.path.join(run_dir, "metadata_frontier")),
        diagnostics_frontier_dir=os.path.join(run_dir, "diagnostics_frontier"),
    )



def write_queries_csv(
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write queries CSV for one dataset.

    Columns: query_id, source_row_index, a_query, x_0, x_1, ..., x_p
    """
    rows = []
    for q in queries:
        x_vals = q.x_query.detach().cpu().tolist()
        row: Dict = {
            "query_id": q.query_id,
            "source_row_index": q.source_row_index,
            "a_query": q.a_query,
        }
        for i, value in enumerate(x_vals):
            row[f"x_{i}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_paths.queries_dir, f"queries_{dataset_id}.csv")
    df.to_csv(path, index=False)
    return path


def _flatten_sweep_results(
    sweep_results: Sequence[SweepResult],
    gamma_grid: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Flatten per-sweep vectorized arrays into per-gamma output columns."""
    num_sweeps = len(sweep_results)
    gamma_array = np.asarray(gamma_grid, dtype=np.float64)
    num_gammas = gamma_array.shape[0]

    return {
        "query_id": np.repeat(np.array([s.query_id for s in sweep_results], dtype=np.int64), num_gammas),
        "bound_type": np.repeat(np.array([s.bound_type for s in sweep_results], dtype=object), num_gammas),
        "gamma_star": np.tile(gamma_array, num_sweeps),
        "theta_star": np.concatenate([s.batch["theta_star"] for s in sweep_results]),
        "empirical_gamma_hat": np.concatenate([s.batch["empirical_gamma_hat"] for s in sweep_results]),
        "pi": np.concatenate([s.batch["pi"] for s in sweep_results]),
        "quantile_level": np.concatenate([s.batch["quantile_level"] for s in sweep_results]),
        "threshold": np.concatenate([s.batch["threshold"] for s in sweep_results]),
        "selected_count": np.concatenate([s.batch["selected_count"] for s in sweep_results]),
        "tail_mass_target": np.concatenate([s.batch["tail_mass_target"] for s in sweep_results]),
        "tail_mass_realized": np.concatenate([s.batch["tail_mass_realized"] for s in sweep_results]),
        "tail_mass_gap": np.concatenate([s.batch["tail_mass_gap"] for s in sweep_results]),
        "threshold_tie_count": np.concatenate([s.batch["threshold_tie_count"] for s in sweep_results]),
        "selected_at_threshold_count": np.concatenate([s.batch["selected_at_threshold_count"] for s in sweep_results]),
    }


def write_frontier_csv(
    sweep_results: Sequence[SweepResult],
    gamma_grid: List[float],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write frontier CSV for one dataset.

    Columns: query_id, bound_type, gamma_star, theta_star
    """
    columns = _flatten_sweep_results(sweep_results, gamma_grid)
    df = pd.DataFrame(
        {
            "query_id": columns["query_id"],
            "bound_type": columns["bound_type"],
            "gamma_star": columns["gamma_star"],
            "theta_star": columns["theta_star"],
        }
    )
    path = os.path.join(output_paths.frontiers_raw_dir, f"frontier_points_{dataset_id}.csv")
    df.to_csv(path, index=False)
    return path


def write_diagnostics(
    sweep_results: Sequence[SweepResult],
    gamma_grid: List[float],
    queries: Sequence[QuerySpec],
    output_paths: OutputPaths,
    dataset_id: int,
) -> str:
    """
    Write optional diagnostics JSON for one dataset.

    Includes per-query analytical summaries and finite-sample discrepancy
    diagnostics. The payload can be large, so write compact JSON rather than
    pretty-printed output.
    """
    columns = _flatten_sweep_results(sweep_results, gamma_grid)
    query_meta = [
        {
            "query_id": q.query_id,
            "source_row_index": q.source_row_index,
            "a_query": q.a_query,
        }
        for q in queries
    ]
    payload = {
        "dataset_id": dataset_id,
        "queries": query_meta,
        "solve_results": {key: value.tolist() for key, value in columns.items()},
    }
    os.makedirs(output_paths.diagnostics_frontier_dir, exist_ok=True)
    path = os.path.join(output_paths.diagnostics_frontier_dir, f"diagnostics_{dataset_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def write_run_metadata(
    config: PipelineConfig,
    output_paths: OutputPaths,
) -> str:
    """Write run-level metadata JSON for reproducibility."""
    payload = {
        "run_dir": config.run_dir,
        "output_schema": {
            "datasets": "Normalized dataset CSVs (written by previous pipeline step).",
            "queries": "Per-dataset query definitions (query_id, x_query, a_query).",
            "frontiers_raw": "Training-facing Pareto points (query_id, bound_type, gamma*, theta*).",
            "metadata_frontier": "Run-level config for reproducibility.",
            "diagnostics_frontier": "Analytical MSM summaries plus finite-sample discrepancy diagnostics.",
        },
        "config": asdict(config),
    }
    path = os.path.join(output_paths.metadata_frontier_dir, "frontier_run_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Subsystem B: analytical MSM frontier construction for partial identification.")
    parser.add_argument("--run_dir", type=str, required=True,
        help="Run directory containing datasets/ and queries/ from previous pipeline steps. "
             "Frontier artifacts are written as subdirectories here.")
    parser.add_argument("--source_run_dir", type=str, default=None,
        help="Optional existing run directory to fork from. When supplied, upstream subdirs "
             "(steps 1+2 outputs) are hardlinked into run_dir before frontier construction runs.")
    parser.add_argument("--dataset_ids", type=int, nargs="*", default=None,
        help="Optional subset of dataset ids to process.")
    parser.add_argument("--device", type=str, default="cpu",
        help="Torch device for model hosting. Only 'cpu' is supported for this script.")
    parser.add_argument("--num_workers", type=int, default=None,
        help="Optional number of CPU worker processes. Omit to use the automatic worker heuristic.")
    parser.add_argument("--save_diagnostics", action="store_true",
        help="Save analytical diagnostics JSON.")

    # Gamma grid config
    parser.add_argument("--gamma_min", type=float, required=True)
    parser.add_argument("--gamma_max", type=float, required=True)
    parser.add_argument("--gamma_n", type=int, default=30)
    parser.add_argument("--gamma_spacing", type=str, default="log",
        choices=["log", "linear", "bounded_pareto"])
    ## Bounded Pareto parameters (only used if --gamma-spacing is bounded_pareto)
    parser.add_argument("--pareto_beta", type=float, default=0.0,
        help="Beta parameter for bounded Pareto gamma spacing. Higher beta concentrates points near gamma_min. Default beta=0 gives log-uniform spacing.")
    parser.add_argument("--pareto_seed", type=int, default=42,
        help="Seed for bounded Pareto sampling. Omit for per-DGP sampling (pass --pareto-seed-per-dgp).")
    parser.add_argument("--pareto_seed_per_dgp", action="store_true",
        help="Use dataset_id as seed, giving each DGP a distinct gamma grid.")

    # Query config
    parser.add_argument("--num_query_points", type=int, default=16,
        help="Number of query rows per dataset (before crossing with treatment arms).")
    parser.add_argument("--query_sampling_mode", type=str, default="ordered_proxy",
        choices=["random", "ordered_proxy", "stratified_proxy"])
    parser.add_argument("--query_random_seed", type=int, default=0)

    # Solver config
    parser.add_argument("--num_mc_samples", type=int, default=128,
        help="Number of latent Monte Carlo samples per query.")

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct PipelineConfig from parsed CLI arguments."""
    return PipelineConfig(
        run_dir=args.run_dir,
        source_run_dir=args.source_run_dir,
        dataset_ids=args.dataset_ids,
        device=args.device,
        num_workers=args.num_workers,
        save_diagnostics=args.save_diagnostics,
        solver=SolverConfig(
            num_mc_samples=args.num_mc_samples,
        ),
        query=QueryConfig(
            num_query_points=args.num_query_points,
            sampling_mode=args.query_sampling_mode,
            random_seed=args.query_random_seed,
        ),
        gamma_grid_cfg=GammaGridConfig(
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            gamma_n=args.gamma_n,
            gamma_spacing=args.gamma_spacing,
            pareto_beta=args.pareto_beta,
            pareto_seed=None if args.pareto_seed_per_dgp else args.pareto_seed, # None signals per-DGP sampling using dataset_id as seed
        ),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
