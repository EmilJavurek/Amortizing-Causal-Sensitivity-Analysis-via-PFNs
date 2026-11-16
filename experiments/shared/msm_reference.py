"""Empirical analytical MSM references from seeded outcome samples."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from setting_standard.causaldataset import (
    build_outcome_model_from_bundle,
    load_outcome_model_bundle,
)

from experiments.comparisons.neuralcsa import OracleOutcomeMap
from experiments.shared.evaluation import propensity_stratified_row_indices

GAMMA_GRID = np.linspace(1.0, 5.0, 21)

QUANTILES = np.array([0.02, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.98])

def load_outcome_map(artifact: dict, device: torch.device) -> OracleOutcomeMap:
    bundle = load_outcome_model_bundle(artifact["outcome_model_path"], map_location="cpu")
    model = build_outcome_model_from_bundle(bundle, device=str(device), freeze=True)
    return OracleOutcomeMap(model)

def seeded_normal(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device.type).manual_seed(seed)
    return torch.randn(*shape, generator=generator, device=device)

def msm_reference_from_samples(
    outcomes: np.ndarray,
    pi_query: np.ndarray,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-threshold empirical MSM bounds on an exact target-Gamma grid."""

    assert outcomes.ndim == 2
    queries, samples = outcomes.shape
    assert pi_query.shape == (queries,)
    upper = np.empty((queries, len(gamma_grid)), dtype=np.float64)
    lower = np.empty_like(upper)
    for query_index in range(queries):
        ascending = np.sort(outcomes[query_index].astype(np.float64))
        for gamma_index, gamma in enumerate(gamma_grid):
            tail_mass = samples / (gamma + 1.0)
            full = int(np.floor(tail_mass))
            fraction = tail_mass - full
            pi = pi_query[query_index]
            inv_high = (1.0 - gamma) * pi + gamma
            inv_low = (1.0 - 1.0 / gamma) * pi + 1.0 / gamma
            raw_sum = tail_mass * inv_high + (samples - tail_mass) * inv_low
            high_weight = inv_high / raw_sum
            low_weight = inv_low / raw_sum
            for destination, ordered in (
                (upper, ascending[::-1]),
                (lower, ascending),
            ):
                value = ordered.sum(dtype=np.float64) * low_weight
                if full > 0:
                    value += (
                        ordered[:full].sum(dtype=np.float64)
                        * (high_weight - low_weight)
                    )
                if fraction > 0.0 and full < samples:
                    value += (
                        fraction
                        * ordered[full]
                        * (high_weight - low_weight)
                    )
                destination[query_index, gamma_index] = value
    assert np.isfinite(upper).all() and np.isfinite(lower).all()
    assert np.all(lower <= upper + 1e-10)
    return lower, upper
