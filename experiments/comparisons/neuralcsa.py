"""NeuralCSA fitting, independent sensitivity solves and matched endpoint evaluation.

Design reference: github.com/DennisFrauen/NeuralCSA@e8d3f952276379cc2c26ad553a142bdf5083ef52.

Stage 1 fits an invertible conditional outcome map; Stage 2 cold-starts every sensitivity
level and bound direction. The oracle-stage control changes only the outcome map.
KL reference interpolation never extrapolates beyond the attained frontier support."""

from __future__ import annotations

from pathlib import Path
from experiments.shared.paths import OBSERVED_RUNS
import math
import time
from dataclasses import asdict, dataclass
from typing import Protocol
import numpy as np
import torch
from scipy.stats import kstest
from torch import nn
from setting_standard.frontier_optimization import (
    FlowConfig,
    _inverse_softplus_target,
    batched_spline_forward,
    estimate_divergence_batched,
    rational_quadratic_spline_1d,
    standard_normal_logprob,
)
import pandas as pd
from setting_standard.causaldataset import (
    build_outcome_model_from_bundle,
    load_outcome_model_bundle,
)
from experiments.shared.evaluation import make_query_grid, normalized_dataset, pifm_predict_frame, propensity_stratified_row_indices
from dataclasses import replace
from experiments.shared.runtime import write as write_json
import argparse
from experiments.shared.evaluation import load_pifm, MSM_CHECKPOINT, KL_CHECKPOINT

def _assert_finite(*values: torch.Tensor) -> None:
    assert all(torch.isfinite(value).all() for value in values), "Non-finite solver quantity"

@dataclass(frozen=True)
class Stage1Config:
    hidden_size: int = 64
    hidden_layers: int = 2
    num_bins: int = 8
    tail_bound: float = 6.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    max_steps: int = 1000
    batch_size: int = 512
    validation_fraction: float = 0.2
    log_every: int = 50
    seed: int = 42

    def flow_config(self) -> FlowConfig:
        return FlowConfig(
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

@dataclass(frozen=True)
class Stage2Config:
    num_bins: int = 8
    tail_bound: float = 6.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    learning_rate: float = 3e-3
    train_mc_samples: int = 256
    q0_mc_samples: int = 2048
    eval_mc_samples: int = 4096
    steps_per_penalty: int = 125
    penalty_weights: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    log_every: int = 25
    feasibility_tolerance: float = 0.02
    msm_dense_grid_points: int = 4097
    seed: int = 314159

    def flow_config(self) -> FlowConfig:
        return FlowConfig(
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

@dataclass
class Stage1FitResult:
    model: "ConditionalOutcomeSpline"
    trajectory: list[dict]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_nll: float
    validation_pit_ks: float
    validation_pit_pvalue: float
    elapsed_seconds: float

@dataclass
class Stage2Result:
    theta: np.ndarray
    gamma: np.ndarray
    gamma_forward: np.ndarray
    gamma_reverse: np.ndarray
    violation: np.ndarray
    q0: np.ndarray
    spline_params: np.ndarray
    selected_penalty_stage: np.ndarray
    trajectory: list[dict]
    elapsed_seconds: float
    config: dict

class OutcomeMap(Protocol):
    def evaluate(
        self,
        x_query: torch.Tensor,
        a_query: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return outcomes with shape ``[M, S]`` for latent input ``[M, S]``."""

class ConditionalOutcomeSpline(nn.Module):
    """Conditional monotone RQS pushforward from standard normal latent to outcome."""

    def __init__(self, input_dim_x: int, config: Stage1Config):
        super().__init__()
        self.input_dim_x = input_dim_x
        self.config = config
        self.flow = config.flow_config()
        d_spline = 3 * config.num_bins - 1

        layers: list[nn.Module] = []
        input_size = input_dim_x + 1
        for _ in range(config.hidden_layers):
            layers.extend([nn.Linear(input_size, config.hidden_size), nn.Tanh()])
            input_size = config.hidden_size
        final = nn.Linear(input_size, d_spline)
        layers.append(final)
        self.conditioner = nn.Sequential(*layers)

        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        derivative_start = 2 * config.num_bins
        identity_derivative = _inverse_softplus_target(1.0 - config.min_derivative)
        with torch.no_grad():
            final.bias[derivative_start:].fill_(identity_derivative)

    def spline_params(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == self.input_dim_x
        if a.ndim == 1:
            a = a[:, None]
        assert a.shape == (x.shape[0], 1)
        params = self.conditioner(torch.cat([x, a], dim=1))
        assert params.shape == (x.shape[0], 3 * self.config.num_bins - 1)
        return params

    def inverse_and_logabsdet(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = y.reshape(-1)
        params = self.spline_params(x, a)
        bins = self.config.num_bins
        z, logabsdet = rational_quadratic_spline_1d(
            inputs=y,
            unnormalized_widths=params[:, :bins],
            unnormalized_heights=params[:, bins : 2 * bins],
            unnormalized_derivatives=params[:, 2 * bins :],
            inverse=True,
            tail_bound=self.config.tail_bound,
            min_bin_width=self.config.min_bin_width,
            min_bin_height=self.config.min_bin_height,
            min_derivative=self.config.min_derivative,
        )
        return z, logabsdet

    def nll(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z, inverse_logabsdet = self.inverse_and_logabsdet(x, a, y)
        log_probability = standard_normal_logprob(z) + inverse_logabsdet
        _assert_finite(z, inverse_logabsdet, log_probability)
        return -log_probability.mean()

    def evaluate(
        self,
        x_query: torch.Tensor,
        a_query: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        assert latent.ndim == 2
        params = self.spline_params(x_query, a_query)
        values, _ = batched_spline_forward(
            latent.unsqueeze(0),
            params.unsqueeze(0),
            self.flow,
        )
        return values.squeeze(0)

class OracleOutcomeMap:
    """Adapter for the immutable normalized outcome BNN of one DGP."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def evaluate(
        self,
        x_query: torch.Tensor,
        a_query: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        assert x_query.ndim == 2
        assert latent.ndim == 2
        m, samples = latent.shape
        assert x_query.shape[0] == m
        assert a_query.reshape(-1).shape == (m,)
        x = x_query[:, None, :].expand(m, samples, x_query.shape[1]).reshape(
            m * samples, x_query.shape[1]
        )
        a = a_query.reshape(m, 1, 1).expand(m, samples, 1).reshape(m * samples, 1)
        u = latent.reshape(m * samples, 1)
        result = self.model(x, a, u).reshape(m, samples)
        _assert_finite(result)
        return result

def fit_stage1_outcome_flow(
    X: torch.Tensor,
    A: torch.Tensor,
    Y: torch.Tensor,
    config: Stage1Config,
    device: torch.device,
) -> Stage1FitResult:
    """Fit NeuralCSA Stage 1 by conditional maximum likelihood."""

    assert X.ndim == 2
    A = A.reshape(-1)
    Y = Y.reshape(-1)
    assert len(X) == len(A) == len(Y)
    assert 0.0 < config.validation_fraction < 1.0
    assert config.max_steps >= 1

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    permutation = torch.randperm(len(X), generator=generator).numpy()
    n_validation = max(1, int(len(X) * config.validation_fraction))
    validation_indices = permutation[:n_validation]
    train_indices = permutation[n_validation:]
    assert len(train_indices) > 0

    model = ConditionalOutcomeSpline(X.shape[1], config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    X_device = X.to(device=device, dtype=torch.float32)
    A_device = A.to(device=device, dtype=torch.float32)
    Y_device = Y.to(device=device, dtype=torch.float32)
    train_tensor = torch.as_tensor(train_indices, device=device, dtype=torch.long)
    validation_tensor = torch.as_tensor(
        validation_indices, device=device, dtype=torch.long
    )

    trajectory: list[dict] = []
    started = time.perf_counter()
    model.train()
    for step in range(config.max_steps):
        positions = torch.randint(
            low=0,
            high=len(train_tensor),
            size=(min(config.batch_size, len(train_tensor)),),
            device=device,
        )
        indices = train_tensor[positions]
        loss = model.nll(X_device[indices], A_device[indices], Y_device[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step == 0 or (step + 1) % config.log_every == 0 or step + 1 == config.max_steps:
            model.eval()
            with torch.no_grad():
                validation_nll = model.nll(
                    X_device[validation_tensor],
                    A_device[validation_tensor],
                    Y_device[validation_tensor],
                )
            trajectory.append(
                {
                    "step": step + 1,
                    "train_batch_nll": float(loss.detach().cpu()),
                    "validation_nll": float(validation_nll.detach().cpu()),
                }
            )
            model.train()

    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        validation_nll_tensor = model.nll(
            X_device[validation_tensor],
            A_device[validation_tensor],
            Y_device[validation_tensor],
        )
        z, _ = model.inverse_and_logabsdet(
            X_device[validation_tensor],
            A_device[validation_tensor],
            Y_device[validation_tensor],
        )
        pit = torch.distributions.Normal(0.0, 1.0).cdf(z).cpu().numpy()
    ks = kstest(pit, "uniform")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Stage1FitResult(
        model=model,
        trajectory=trajectory,
        train_indices=train_indices,
        validation_indices=validation_indices,
        validation_nll=float(validation_nll_tensor.cpu()),
        validation_pit_ks=float(ks.statistic),
        validation_pit_pvalue=float(ks.pvalue),
        elapsed_seconds=elapsed,
    )

def identity_spline_params(
    instances: int,
    config: Stage2Config,
    device: torch.device,
) -> torch.Tensor:
    d_spline = 3 * config.num_bins - 1
    values = torch.zeros(instances, d_spline, device=device)
    derivative_start = 2 * config.num_bins
    derivative = _inverse_softplus_target(1.0 - config.min_derivative)
    values[:, derivative_start:].fill_(derivative)
    return values

def _seeded_normal(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return torch.randn(*shape, generator=generator, device=device)

def _theta_and_gamma(
    spline_params: torch.Tensor,
    outcome_map: OutcomeMap,
    x_query: torch.Tensor,
    a_query: torch.Tensor,
    pi_query: torch.Tensor,
    q0: torch.Tensor,
    sensitivity_model: str,
    base_samples: torch.Tensor,
    flow_config: FlowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    nu, log_nu = batched_spline_forward(
        base_samples.unsqueeze(0),
        spline_params.unsqueeze(0),
        flow_config,
    )
    nu = nu.squeeze(0)
    log_nu = log_nu.squeeze(0)
    log_r = log_nu - standard_normal_logprob(nu)
    gamma = estimate_divergence_batched(
        log_r.unsqueeze(0), sensitivity_model
    ).squeeze(0)
    counterfactual = outcome_map.evaluate(x_query, a_query, nu).mean(dim=1)
    theta = pi_query * q0 + (1.0 - pi_query) * counterfactual
    _assert_finite(theta, gamma)
    return theta, gamma

def _spline_log_density_at(
    values: torch.Tensor,
    spline_params: torch.Tensor,
    flow_config: FlowConfig,
) -> torch.Tensor:
    """Evaluate log nu at arbitrary values for independent reverse-KL checks."""

    assert values.ndim == 2
    instances, samples = values.shape
    assert spline_params.shape[0] == instances
    bins = flow_config.num_bins
    repeated = spline_params[:, None, :].expand(
        instances, samples, spline_params.shape[1]
    ).reshape(instances * samples, spline_params.shape[1])
    base, inverse_logabsdet = rational_quadratic_spline_1d(
        inputs=values.reshape(-1),
        unnormalized_widths=repeated[:, :bins],
        unnormalized_heights=repeated[:, bins : 2 * bins],
        unnormalized_derivatives=repeated[:, 2 * bins :],
        inverse=True,
        tail_bound=flow_config.tail_bound,
        min_bin_width=flow_config.min_bin_width,
        min_bin_height=flow_config.min_bin_height,
        min_derivative=flow_config.min_derivative,
    )
    log_density = standard_normal_logprob(base) + inverse_logabsdet
    return log_density.reshape(instances, samples)

def _independent_gamma_diagnostics(
    spline_params: torch.Tensor,
    sensitivity_model: str,
    config: Stage2Config,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return robust gamma plus its two defining directional components."""

    instances = spline_params.shape[0]
    flow_config = config.flow_config()
    if sensitivity_model == "kl":
        nu_base = _seeded_normal(
            (instances, config.eval_mc_samples), seed, device
        )
        nu, log_nu = batched_spline_forward(
            nu_base.unsqueeze(0), spline_params.unsqueeze(0), flow_config
        )
        nu = nu.squeeze(0)
        log_nu = log_nu.squeeze(0)
        forward = (
            log_nu - standard_normal_logprob(nu)
        ).mean(dim=1)

        phi = _seeded_normal(
            (instances, config.eval_mc_samples), seed + 1, device
        )
        log_nu_at_phi = _spline_log_density_at(phi, spline_params, flow_config)
        reverse = (
            standard_normal_logprob(phi) - log_nu_at_phi
        ).mean(dim=1)
        forward = torch.clamp(forward, min=0.0)
        reverse = torch.clamp(reverse, min=0.0)
        gamma = torch.maximum(forward, reverse)
        _assert_finite(forward, reverse, gamma)
        return gamma, forward, reverse

    assert sensitivity_model == "msm"
    assert config.msm_dense_grid_points >= 257
    base_grid = torch.linspace(
        -max(8.0, flow_config.tail_bound + 1.0),
        max(8.0, flow_config.tail_bound + 1.0),
        config.msm_dense_grid_points,
        device=device,
    )
    base = base_grid[None, :].expand(instances, -1)
    nu, log_nu = batched_spline_forward(
        base.unsqueeze(0), spline_params.unsqueeze(0), flow_config
    )
    nu = nu.squeeze(0)
    log_r = log_nu.squeeze(0) - standard_normal_logprob(nu)
    max_ratio = torch.exp(log_r).amax(dim=1)
    max_inverse_ratio = torch.exp(-log_r).amax(dim=1)
    gamma = torch.maximum(max_ratio, max_inverse_ratio)
    _assert_finite(max_ratio, max_inverse_ratio, gamma)
    return gamma, max_ratio, max_inverse_ratio

def _clip_spline_grad_per_instance_(
    spline_params: nn.Parameter,
    max_norm: float,
) -> None:
    if spline_params.grad is None:
        return
    grad = spline_params.grad
    assert grad.ndim == 2
    norms = torch.linalg.vector_norm(grad, dim=1, keepdim=True)
    scales = torch.clamp(max_norm / torch.clamp(norms, min=1e-12), max=1.0)
    grad.mul_(scales)

def solve_fixed_gamma_batched(
    outcome_map: OutcomeMap,
    x_query: torch.Tensor,
    a_query: torch.Tensor,
    pi_query: torch.Tensor,
    gamma_target: torch.Tensor,
    bound_type: str,
    sensitivity_model: str,
    config: Stage2Config,
    device: torch.device,
) -> Stage2Result:
    """Cold-start independent prescribed-Gamma solves, batched only for throughput."""

    assert bound_type in {"upper", "lower"}
    assert sensitivity_model in {"msm", "kl"}
    assert x_query.ndim == 2
    instances = x_query.shape[0]
    a_query = a_query.reshape(instances).to(device=device, dtype=torch.float32)
    pi_query = pi_query.reshape(instances).to(device=device, dtype=torch.float32)
    gamma_target = gamma_target.reshape(instances).to(
        device=device, dtype=torch.float32
    )
    x_query = x_query.to(device=device, dtype=torch.float32)
    assert torch.all((pi_query > 0.0) & (pi_query < 1.0))
    if sensitivity_model == "msm":
        assert torch.all(gamma_target >= 1.0)
    else:
        assert torch.all(gamma_target >= 0.0)
    assert len(config.penalty_weights) >= 1
    assert config.feasibility_tolerance >= 0.0

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    flow_config = config.flow_config()
    q0_base = _seeded_normal(
        (instances, config.q0_mc_samples),
        config.seed + 1,
        device,
    )
    with torch.no_grad():
        q0 = outcome_map.evaluate(x_query, a_query, q0_base).mean(dim=1)
    _assert_finite(q0)

    spline_params = nn.Parameter(identity_spline_params(instances, config, device))
    optimizer = torch.optim.Adam([spline_params], lr=config.learning_rate)
    multiplier = torch.zeros(instances, device=device)
    sign = 1.0 if bound_type == "upper" else -1.0
    trajectory: list[dict] = []
    global_step = 0
    started = time.perf_counter()

    with torch.no_grad():
        initial_gamma, initial_forward, initial_reverse = _independent_gamma_diagnostics(
            spline_params,
            sensitivity_model,
            config,
            config.seed + 500_000,
            device,
        )
        initial_base = _seeded_normal(
            (instances, config.eval_mc_samples),
            config.seed + 500_002,
            device,
        )
        initial_theta, _ = _theta_and_gamma(
            spline_params=spline_params,
            outcome_map=outcome_map,
            x_query=x_query,
            a_query=a_query,
            pi_query=pi_query,
            q0=q0,
            sensitivity_model=sensitivity_model,
            base_samples=initial_base,
            flow_config=flow_config,
        )
    best_params = spline_params.detach().clone()
    best_signed_theta = sign * initial_theta
    best_violation = torch.relu(initial_gamma - gamma_target)
    best_gamma = initial_gamma
    best_forward = initial_forward
    best_reverse = initial_reverse
    selected_stage = torch.full(
        (instances,), -1, device=device, dtype=torch.int64
    )

    for penalty_index, penalty_weight in enumerate(config.penalty_weights):
        assert penalty_weight > 0.0
        for local_step in range(config.steps_per_penalty):
            global_step += 1
            base = torch.randn(
                instances,
                config.train_mc_samples,
                device=device,
            )
            theta, gamma = _theta_and_gamma(
                spline_params=spline_params,
                outcome_map=outcome_map,
                x_query=x_query,
                a_query=a_query,
                pi_query=pi_query,
                q0=q0,
                sensitivity_model=sensitivity_model,
                base_samples=base,
                flow_config=flow_config,
            )
            constraint = gamma - gamma_target
            # Powell-Hestenes-Rockafellar inequality augmented Lagrangian.
            shifted = torch.relu(multiplier + penalty_weight * constraint)
            penalty = (
                shifted.square() - multiplier.square()
            ) / (2.0 * penalty_weight)
            signed_theta = sign * theta
            per_instance_loss = -signed_theta + penalty
            loss = per_instance_loss.sum()
            _assert_finite(loss, constraint, shifted)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _clip_spline_grad_per_instance_(spline_params, max_norm=5.0)
            optimizer.step()

            should_log = (
                global_step == 1
                or (local_step + 1) % config.log_every == 0
                or local_step + 1 == config.steps_per_penalty
            )
            if should_log:
                trajectory.append(
                    {
                        "step": global_step,
                        "penalty_index": penalty_index,
                        "penalty_weight": float(penalty_weight),
                        "theta_mean": float(theta.detach().mean().cpu()),
                        "gamma_mean": float(gamma.detach().mean().cpu()),
                        "target_mean": float(gamma_target.mean().cpu()),
                        "positive_violation_mean": float(
                            torch.relu(constraint).detach().mean().cpu()
                        ),
                        "positive_violation_max": float(
                            torch.relu(constraint).detach().max().cpu()
                        ),
                        "multiplier_mean": float(multiplier.mean().cpu()),
                        "loss_mean": float(per_instance_loss.detach().mean().cpu()),
                    }
                )

        with torch.no_grad():
            update_base = _seeded_normal(
                (instances, config.eval_mc_samples),
                config.seed + 1000 + penalty_index,
                device,
            )
            _, update_gamma = _theta_and_gamma(
                spline_params=spline_params,
                outcome_map=outcome_map,
                x_query=x_query,
                a_query=a_query,
                pi_query=pi_query,
                q0=q0,
                sensitivity_model=sensitivity_model,
                base_samples=update_base,
                flow_config=flow_config,
            )
            multiplier = torch.relu(
                multiplier + penalty_weight * (update_gamma - gamma_target)
            )
            robust_gamma, robust_forward, robust_reverse = (
                _independent_gamma_diagnostics(
                    spline_params,
                    sensitivity_model,
                    config,
                    config.seed + 600_000 + 10 * penalty_index,
                    device,
                )
            )
            candidate_theta, _ = _theta_and_gamma(
                spline_params=spline_params,
                outcome_map=outcome_map,
                x_query=x_query,
                a_query=a_query,
                pi_query=pi_query,
                q0=q0,
                sensitivity_model=sensitivity_model,
                base_samples=update_base,
                flow_config=flow_config,
            )
            candidate_signed = sign * candidate_theta
            candidate_violation = torch.relu(robust_gamma - gamma_target)
            candidate_feasible = (
                candidate_violation <= config.feasibility_tolerance
            )
            best_feasible = best_violation <= config.feasibility_tolerance
            improves = (
                candidate_feasible
                & (
                    (~best_feasible)
                    | (candidate_signed > best_signed_theta)
                )
            ) | (
                (~candidate_feasible)
                & (~best_feasible)
                & (candidate_violation < best_violation)
            )
            if torch.any(improves):
                best_params[improves] = spline_params.detach()[improves]
                best_signed_theta[improves] = candidate_signed[improves]
                best_violation[improves] = candidate_violation[improves]
                best_gamma[improves] = robust_gamma[improves]
                best_forward[improves] = robust_forward[improves]
                best_reverse[improves] = robust_reverse[improves]
                selected_stage[improves] = penalty_index

    with torch.no_grad():
        spline_params.copy_(best_params)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        eval_base = _seeded_normal(
            (instances, config.eval_mc_samples),
            config.seed + 999_999,
            device,
        )
        theta, _ = _theta_and_gamma(
            spline_params=spline_params,
            outcome_map=outcome_map,
            x_query=x_query,
            a_query=a_query,
            pi_query=pi_query,
            q0=q0,
            sensitivity_model=sensitivity_model,
            base_samples=eval_base,
            flow_config=flow_config,
        )
        gamma, gamma_forward, gamma_reverse = _independent_gamma_diagnostics(
            spline_params,
            sensitivity_model,
            config,
            config.seed + 999_997,
            device,
        )
        violation = torch.relu(gamma - gamma_target)
    return Stage2Result(
        theta=theta.cpu().numpy(),
        gamma=gamma.cpu().numpy(),
        gamma_forward=gamma_forward.cpu().numpy(),
        gamma_reverse=gamma_reverse.cpu().numpy(),
        violation=violation.cpu().numpy(),
        q0=q0.cpu().numpy(),
        spline_params=spline_params.detach().cpu().numpy(),
        selected_penalty_stage=selected_stage.cpu().numpy(),
        trajectory=trajectory,
        elapsed_seconds=elapsed,
        config=asdict(config),
    )

def solve_fixed_gamma_sequential(
    outcome_map: OutcomeMap,
    x_query: torch.Tensor,
    a_query: torch.Tensor,
    pi_query: torch.Tensor,
    gamma_target: torch.Tensor,
    bound_type: str,
    sensitivity_model: str,
    config: Stage2Config,
    device: torch.device,
) -> tuple[list[Stage2Result], float]:
    """Strong timing reference that preserves literal one-instance cold starts."""

    results: list[Stage2Result] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in range(len(x_query)):
        result = solve_fixed_gamma_batched(
            outcome_map=outcome_map,
            x_query=x_query[index : index + 1],
            a_query=a_query[index : index + 1],
            pi_query=pi_query[index : index + 1],
            gamma_target=gamma_target[index : index + 1],
            bound_type=bound_type,
            sensitivity_model=sensitivity_model,
            config=config,
            device=device,
        )
        results.append(result)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return results, time.perf_counter() - started

def analytical_msm_oracle(
    outcome_map: OutcomeMap,
    x_query: torch.Tensor,
    a_query: torch.Tensor,
    pi_query: torch.Tensor,
    gamma_target: torch.Tensor,
    bound_type: str,
    samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """High-MC empirical analytical MSM reference with a fractional threshold.

    Fractional allocation at the threshold hits the requested finite-bank weight mass rather than
    rounding the tail count as the production label generator does.
    """

    assert bound_type in {"upper", "lower"}
    x_query = x_query.to(device)
    a_query = a_query.reshape(-1).to(device)
    pi_query = pi_query.reshape(-1).to(device)
    gamma_target = gamma_target.reshape(-1).to(device)
    base = _seeded_normal((len(x_query), samples), seed, device)
    with torch.no_grad():
        outcomes = outcome_map.evaluate(x_query, a_query, base).cpu().numpy()
    pi = pi_query.cpu().numpy()
    gamma = gamma_target.cpu().numpy()
    answers = np.empty(len(x_query), dtype=np.float64)

    for index in range(len(x_query)):
        values = np.sort(outcomes[index])
        if bound_type == "upper":
            values = values[::-1]
        n = len(values)
        tail_mass = n / (gamma[index] + 1.0)
        full = int(math.floor(tail_mass))
        fraction = tail_mass - full
        inv_high = (1.0 - gamma[index]) * pi[index] + gamma[index]
        inv_low = (1.0 - 1.0 / gamma[index]) * pi[index] + 1.0 / gamma[index]
        raw_sum = tail_mass * inv_high + (n - tail_mass) * inv_low
        high_weight = inv_high / raw_sum
        low_weight = inv_low / raw_sum
        total = values.sum(dtype=np.float64) * low_weight
        if full > 0:
            total += values[:full].sum(dtype=np.float64) * (
                high_weight - low_weight
            )
        if fraction > 0.0 and full < n:
            total += (
                fraction * values[full] * (high_weight - low_weight)
            )
        answers[index] = total
    assert np.isfinite(answers).all()
    return answers

def configs_for_mode(mode: str, seed: int) -> tuple[Stage1Config, Stage2Config, int]:
    if mode == "smoke":
        return (
            Stage1Config(
                hidden_size=32,
                max_steps=100,
                batch_size=256,
                log_every=20,
                seed=seed,
            ),
            Stage2Config(
                train_mc_samples=128,
                q0_mc_samples=512,
                eval_mc_samples=1024,
                steps_per_penalty=30,
                penalty_weights=(10.0, 100.0, 1000.0),
                feasibility_tolerance=0.10,
                msm_dense_grid_points=1025,
                log_every=15,
                seed=seed + 10,
            ),
            20_000,
        )
    return (
        Stage1Config(seed=seed),
        Stage2Config(seed=seed + 10),
        65_536,
    )

def load_dgp(
    artifact: dict,
    device: torch.device,
):
    frame, x_columns = normalized_dataset(artifact)
    bundle = load_outcome_model_bundle(artifact["outcome_model_path"], map_location="cpu")
    model = build_outcome_model_from_bundle(bundle, device=str(device), freeze=True)
    return frame, x_columns, OracleOutcomeMap(model)

def fit_and_save_stage1(
    dataset_id: int,
    frame: pd.DataFrame,
    x_columns: list[str],
    config: Stage1Config,
    device: torch.device,
    output_dir: Path,
):
    result = fit_stage1_outcome_flow(
        X=torch.tensor(frame[x_columns].to_numpy(), dtype=torch.float32),
        A=torch.tensor(frame["treatment"].to_numpy(), dtype=torch.float32),
        Y=torch.tensor(frame["outcome"].to_numpy(), dtype=torch.float32),
        config=config,
        device=device,
    )
    payload_path = output_dir / "stage1_models" / f"conditional_outcome_flow_{dataset_id}.pth"
    torch.save(
        {
            "dataset_id": dataset_id,
            "config": asdict(config),
            "model_state_dict": result.model.state_dict(),
            "validation_nll": result.validation_nll,
            "validation_pit_ks": result.validation_pit_ks,
            "validation_pit_pvalue": result.validation_pit_pvalue,
            "elapsed_seconds": result.elapsed_seconds,
            "train_indices": result.train_indices,
            "validation_indices": result.validation_indices,
        },
        payload_path,
    )
    write_json(
        output_dir / "trajectories" / f"stage1_{dataset_id}.json",
        result.trajectory,
    )
    return result

def stage2_rows(
    *,
    dataset_id: int,
    sensitivity_model: str,
    variant: str,
    bound_type: str,
    query_frame: pd.DataFrame,
    x_columns: list[str],
    result,
    theta_reference: np.ndarray | None,
) -> pd.DataFrame:
    output = query_frame.copy()
    output["dataset_id"] = dataset_id
    output["sensitivity_model"] = sensitivity_model
    output["method"] = f"neuralcsa_{variant}"
    output["bound_type"] = bound_type
    output["theta_prediction"] = result.theta
    output["theta_reference"] = (
        theta_reference if theta_reference is not None else np.nan
    )
    output["gamma_realized"] = result.gamma
    output["gamma_direction_forward"] = result.gamma_forward
    output["gamma_direction_reverse"] = result.gamma_reverse
    output["constraint_violation"] = result.violation
    output["q0"] = result.q0
    output["selected_penalty_stage"] = result.selected_penalty_stage
    output["solve_seconds_total_batch"] = result.elapsed_seconds
    return output

def pifm_accuracy_rows(
    *,
    dataset_id: int,
    sensitivity_model: str,
    prediction: pd.DataFrame,
    theta_reference_by_bound: dict[str, np.ndarray | None],
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for bound_type in ("upper", "lower"):
        frame = prediction.copy()
        frame["dataset_id"] = dataset_id
        frame["sensitivity_model"] = sensitivity_model
        frame["method"] = "pifm"
        frame["bound_type"] = bound_type
        frame["theta_prediction"] = frame[f"{bound_type}_mean"]
        reference = theta_reference_by_bound[bound_type]
        frame["theta_reference"] = reference if reference is not None else np.nan
        frame["gamma_realized"] = frame["gamma"]
        frame["gamma_direction_forward"] = np.nan
        frame["gamma_direction_reverse"] = np.nan
        frame["constraint_violation"] = 0.0
        frame["q0"] = np.nan
        frame["selected_penalty_stage"] = np.nan
        frame["solve_seconds_total_batch"] = np.nan
        outputs.append(frame)
    return outputs

MSM_LEVELS = (1.5, 3.0, 5.0)

KL_LEVELS = (0.05, 0.5, 2.0)

METHODS = {"pifm", "neuralcsa_full", "neuralcsa_oracle_stage"}

KEYS = [
    "dataset_id",
    "source_row_index",
    "a_query",
    "sensitivity_model",
    "method",
    "bound_type",
    "gamma",
]

def expected_rows(mode: str) -> int:
    return 24 if mode == "smoke" else 144

def assert_accuracy_fragment(
    frame: pd.DataFrame,
    dataset_id: int,
    mode: str,
) -> None:
    assert len(frame) == expected_rows(mode)
    assert not frame.duplicated(KEYS).any()
    assert frame["dataset_id"].eq(dataset_id).all()
    assert set(frame["sensitivity_model"]) == {"msm", "kl"}
    assert set(frame["method"]) == METHODS
    assert set(frame["bound_type"]) == {"upper", "lower"}
    assert set(frame["a_query"].astype(int)) == {0, 1}
    assert frame["source_row_index"].nunique() == (1 if mode == "smoke" else 2)
    levels = {
        "msm": {2.0} if mode == "smoke" else set(MSM_LEVELS),
        "kl": {0.5} if mode == "smoke" else set(KL_LEVELS),
    }
    for sensitivity_model, local in frame.groupby("sensitivity_model"):
        assert set(local["gamma"].astype(float)) == levels[str(sensitivity_model)]
    required_finite = [
        "dataset_id",
        "source_row_index",
        "a_query",
        "gamma",
        "theta_prediction",
        "gamma_realized",
        "constraint_violation",
        "p_a1",
        "pi_query",
        "counterfactual_weight",
        "global_position",
    ]
    assert np.isfinite(frame[required_finite].to_numpy(dtype=np.float64)).all()
    msm = frame[frame["sensitivity_model"].eq("msm")]
    assert np.isfinite(msm["theta_reference"].to_numpy()).all()
    kl = frame[frame["sensitivity_model"].eq("kl")]
    assert kl["theta_reference"].isna().all()
    neural = frame[frame["method"].ne("pifm")]
    assert np.isfinite(
        neural[
            [
                "gamma_direction_forward",
                "gamma_direction_reverse",
                "q0",
                "selected_penalty_stage",
                "solve_seconds_total_batch",
            ]
        ].to_numpy(dtype=np.float64)
    ).all()
    assert (neural["constraint_violation"] >= 0).all()
    assert frame["cohort"].eq("fresh").all()
    counts = frame.groupby(["sensitivity_model", "method", "bound_type"]).size()
    per_group = 2 if mode == "smoke" else 12
    assert (counts == per_group).all(), counts

def evaluate_one(
    *,
    artifact: dict,
    dataset_id: int,
    global_position: int,
    mode: str,
    base_stage1,
    base_stage2,
    oracle_mc: int,
    models: dict,
    device: torch.device,
    attempt_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    frame, x_columns, oracle_outcome = load_dgp(artifact, device)
    rows_per_dgp = 1 if mode == "smoke" else 2
    row_indices = propensity_stratified_row_indices(
        frame,
        rows_per_dgp,
        quantiles=np.linspace(0.25, 0.75, rows_per_dgp),
    )
    stage1_config = replace(base_stage1, seed=base_stage1.seed + global_position)
    stage1 = fit_and_save_stage1(
        dataset_id,
        frame,
        x_columns,
        stage1_config,
        device,
        attempt_dir,
    )
    outputs: list[pd.DataFrame] = []
    for sensitivity_model, levels in (
        ("msm", (2.0,) if mode == "smoke" else MSM_LEVELS),
        ("kl", (0.5,) if mode == "smoke" else KL_LEVELS),
    ):
        query = make_query_grid(frame, x_columns, row_indices, levels)
        x_query = torch.tensor(query[x_columns].to_numpy(), dtype=torch.float32)
        a_query = torch.tensor(query["a_query"].to_numpy(), dtype=torch.float32)
        pi_query = torch.tensor(query["pi_query"].to_numpy(), dtype=torch.float32)
        gamma_target = torch.tensor(query["gamma"].to_numpy(), dtype=torch.float32)
        references: dict[str, np.ndarray | None] = {"upper": None, "lower": None}
        if sensitivity_model == "msm":
            for bound_type in ("upper", "lower"):
                references[bound_type] = analytical_msm_oracle(
                    oracle_outcome,
                    x_query,
                    a_query,
                    pi_query,
                    gamma_target,
                    bound_type,
                    oracle_mc,
                    base_stage2.seed + (0 if bound_type == "upper" else 1),
                    device,
                )
        prediction = pifm_predict_frame(
            models[sensitivity_model],
            frame,
            x_columns,
            query,
            device,
        )
        outputs.extend(
            pifm_accuracy_rows(
                dataset_id=dataset_id,
                sensitivity_model=sensitivity_model,
                prediction=prediction,
                theta_reference_by_bound=references,
            )
        )
        for variant, outcome_map in (
            ("oracle_stage", oracle_outcome),
            ("full", stage1.model),
        ):
            for bound_type in ("upper", "lower"):
                local_config = replace(
                    base_stage2,
                    seed=base_stage2.seed
                    + 1_000 * global_position
                    + (0 if sensitivity_model == "msm" else 100)
                    + (0 if bound_type == "upper" else 10),
                )
                result = solve_fixed_gamma_batched(
                    outcome_map=outcome_map,
                    x_query=x_query,
                    a_query=a_query,
                    pi_query=pi_query,
                    gamma_target=gamma_target,
                    bound_type=bound_type,
                    sensitivity_model=sensitivity_model,
                    config=local_config,
                    device=device,
                )
                write_json(
                    attempt_dir
                    / "trajectories"
                    / (
                        f"stage2_{dataset_id}_{sensitivity_model}_{variant}_"
                        f"{bound_type}.json"
                    ),
                    result.trajectory,
                )
                outputs.append(
                    stage2_rows(
                        dataset_id=dataset_id,
                        sensitivity_model=sensitivity_model,
                        variant=variant,
                        bound_type=bound_type,
                        query_frame=query,
                        x_columns=x_columns,
                        result=result,
                        theta_reference=references[bound_type],
                    )
                )
    accuracy = pd.concat(outputs, ignore_index=True)
    accuracy["cohort"] = "fresh"
    accuracy["global_position"] = global_position
    accuracy["gamma_gap"] = accuracy["gamma_realized"] - accuracy["gamma"]
    accuracy["undersaturation"] = np.maximum(-accuracy["gamma_gap"], 0.0)
    accuracy["feasible"] = (
        accuracy["constraint_violation"] <= base_stage2.feasibility_tolerance
    )
    accuracy["target_attained"] = (
        accuracy["gamma_gap"].abs() <= base_stage2.feasibility_tolerance
    )
    accuracy["materially_undersaturated"] = (
        accuracy["undersaturation"] > base_stage2.feasibility_tolerance
    )
    assert_accuracy_fragment(accuracy, dataset_id, mode)
    stage1_metrics = {
        "dataset_id": dataset_id,
        "global_position": global_position,
        "validation_nll": stage1.validation_nll,
        "validation_pit_ks": stage1.validation_pit_ks,
        "validation_pit_pvalue": stage1.validation_pit_pvalue,
        "elapsed_seconds": stage1.elapsed_seconds,
        "stage1_seed": stage1_config.seed,
        "train_count": len(stage1.train_indices),
        "validation_count": len(stage1.validation_indices),
    }
    return accuracy, stage1_metrics

METHOD_LABELS = {
    "pifm": "PI-FM",
    "neuralcsa_full": "Full NeuralCSA",
    "neuralcsa_oracle_stage": "Oracle-stage NeuralCSA",
}

JOIN_KEYS = ["dataset_id", "source_row_index", "a_query", "bound_type"]

def attach_kl_reference(
    accuracy: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate E1's selected scalarized curves without extrapolation."""

    required_accuracy = {
        *JOIN_KEYS,
        "sensitivity_model",
        "method",
        "gamma",
        "theta_prediction",
        "theta_reference",
    }
    required_reference = {
        *JOIN_KEYS,
        "gamma_reference",
        "theta_reference",
    }
    assert required_accuracy <= set(accuracy.columns)
    assert required_reference <= set(reference.columns)
    assert set(accuracy["sensitivity_model"]) == {"msm", "kl"}
    assert set(accuracy["method"]) == set(METHOD_LABELS)

    curves: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    query_identity: dict[tuple, pd.Series] = {}
    identity_columns = [
        column
        for column in [
            "p_a1",
            "pi_query",
            "counterfactual_weight",
            *[f"x_{i}" for i in range(10)],
        ]
        if column in reference.columns and column in accuracy.columns
    ]
    for key, group in reference.groupby(JOIN_KEYS, sort=False):
        bound_type = str(key[-1])
        reduction = "max" if bound_type == "upper" else "min"
        ordered = group.groupby("gamma_reference", as_index=False).agg(
            theta_reference=("theta_reference", reduction)
        )
        ordered = ordered.sort_values("gamma_reference")
        gamma = ordered["gamma_reference"].to_numpy(dtype=float)
        theta = ordered["theta_reference"].to_numpy(dtype=float)
        if bound_type == "upper":
            theta = np.maximum.accumulate(theta)
        else:
            theta = np.minimum.accumulate(theta)
        assert len(gamma) >= 2 and np.all(np.diff(gamma) > 0)
        assert np.isfinite(gamma).all() and np.isfinite(theta).all()
        curves[tuple(key)] = (gamma, theta)
        query_identity[tuple(key)] = group.iloc[0][identity_columns]

    output = accuracy.copy()
    output["theta_reference_runner"] = output["theta_reference"]
    output["reference_curve_min_gamma"] = np.nan
    output["reference_curve_max_gamma"] = np.nan
    output["reference_curve_points"] = np.nan
    output["reference_source"] = np.where(
        output["sensitivity_model"].eq("msm"),
        "analytical_msm",
        "unavailable_outside_e1_curve",
    )

    kl_mask = output["sensitivity_model"].eq("kl")
    for index, row in output.loc[kl_mask].iterrows():
        key = tuple(row[column] for column in JOIN_KEYS)
        assert key in curves, f"E1 curve missing for exact E3 query {key}"
        gamma_curve, theta_curve = curves[key]
        for column in identity_columns:
            expected = float(query_identity[key][column])
            actual = float(row[column])
            assert math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=2e-6,
            ), (key, column, actual, expected)
        target = float(row["gamma"])
        output.at[index, "reference_curve_min_gamma"] = gamma_curve[0]
        output.at[index, "reference_curve_max_gamma"] = gamma_curve[-1]
        output.at[index, "reference_curve_points"] = len(gamma_curve)
        if gamma_curve[0] <= target <= gamma_curve[-1]:
            output.at[index, "theta_reference"] = float(
                np.interp(target, gamma_curve, theta_curve)
            )
            output.at[index, "reference_source"] = "e1_linear_interpolation"

    output["reference_available"] = output["theta_reference"].notna()
    output["error"] = output["theta_prediction"] - output["theta_reference"]
    output["abs_error"] = output["error"].abs()

    assert output.loc[
        output["sensitivity_model"].eq("msm"), "reference_available"
    ].all()
    for _, group in output.loc[kl_mask].groupby(
        ["method", *JOIN_KEYS, "gamma"],
        sort=False,
    ):
        assert group["reference_available"].nunique() == 1
    return output

def run(smoke):
    from experiments.shared.observed import observed_run
    from experiments.shared.runtime import fresh, read
    from experiments.shared.inputs import normalized_root, cohort_artifacts
    run=observed_run('e3_smoke' if smoke else 'e3',smoke)
    artifacts=cohort_artifacts('neuralcsa')
    from experiments.shared.inputs import cohort_ids
    ids=cohort_ids('neuralcsa')[:1] if smoke else cohort_ids('neuralcsa')
    first,second,mc=configs_for_mode('accuracy',20260726)
    device=torch.device('cuda:0')
    models={k:load_pifm(p,device)[0] for k,p in [('msm',MSM_CHECKPOINT),('kl',KL_CHECKPOINT)]}
    e1=pd.concat([pd.read_parquet(p) for p in sorted((OBSERVED_RUNS/'e1/fragments').glob('*.parquet'))],ignore_index=True)
    run.manifest.update(stage1=asdict(first),stage2=asdict(second),oracle_mc=mc,timing='accuracy run; batch solver timings are not deployment benchmarks')
    fresh(run.path/'fragments');metrics=[]
    try:
        for pos,i in enumerate(ids):

            attempt=fresh(run.path/f'dgp_{i}');fresh(attempt/'stage1_models');fresh(attempt/'trajectories')
            accuracy,m=evaluate_one(artifact=artifacts[i],dataset_id=i,global_position=pos,mode='retained',base_stage1=first,base_stage2=second,oracle_mc=mc,models=models,device=device,attempt_dir=attempt)
            accuracy=attach_kl_reference(accuracy,e1)
            stats=read(normalized_root()/'scalers'/f'scaler_{i}.json');accuracy['sigma_obs']=stats['y_std']
            accuracy.to_parquet(run.path/'fragments'/f'dgp_{i}.parquet',index=False)
            metrics.append(m);run.manifest['completed_ids'].append(i);run.save()
            print(f'E3 {pos+1}/{len(ids)} elapsed={time.time()-run.started:.1f}s',flush=True)
        pd.DataFrame(metrics).to_csv(run.path/'stage1_metrics.csv',index=False)
        run.done()
    except BaseException:
        run.done('failed');raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.smoke)


if __name__ == '__main__':
    main()
