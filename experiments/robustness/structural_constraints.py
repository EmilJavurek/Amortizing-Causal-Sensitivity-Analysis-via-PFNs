"""Euclidean projection of paired sensitivity-bound curves.

For an increasingly ordered Gamma grid, the feasible set is

    lower[0] >= ... >= lower[G - 1],
    upper[0] <= ... <= upper[G - 1],
    lower[g] <= upper[g] for every g.

The component projections are exact: unweighted PAVA for each monotone cone
and pairwise averaging for the ordering halfspaces.  Cyclic Dykstra projection
with one correction vector per component set therefore converges to the
Euclidean projection onto their intersection.
"""

from __future__ import annotations
import sys
from pathlib import Path
from experiments.shared.paths import CODE, ROOT, EXPERIMENTS
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class StructuralViolations:
    """Maximum positive violation of each joint-set constraint family."""

    lower_nonincreasing: float
    upper_nondecreasing: float
    pointwise_order: float

    @property
    def maximum(self) -> float:
        return max(
            self.lower_nonincreasing,
            self.upper_nondecreasing,
            self.pointwise_order,
        )

@dataclass(frozen=True)
class DykstraCycle:
    """One complete three-set Dykstra cycle."""

    cycle: int
    iterate_delta_linf: float
    iterate_delta_l2: float
    correction_delta_linf: float
    objective_squared: float
    objective_delta: float
    feasibility_violation: float
    component_fixed_point_residual: float
    iterate_tolerance: float
    objective_tolerance: float
    feasibility_tolerance: float
    consecutive_stable_cycles: int

@dataclass(frozen=True)
class JointProjectionResult:
    """Projected curve and the strict convergence ledger."""

    lower: FloatArray
    upper: FloatArray
    cycles: int
    converged: bool
    ledger: tuple[DykstraCycle, ...]

def _as_finite_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    assert vector.ndim == 1, f"{name} must be one-dimensional, got {vector.shape}"
    assert vector.size > 0, f"{name} must be nonempty"
    assert np.isfinite(vector).all(), f"{name} contains a nonfinite value"
    return vector.copy()

def pava_nondecreasing(values: ArrayLike) -> FloatArray:
    """Return the exact unweighted L2 projection onto nondecreasing vectors."""

    vector = _as_finite_vector(values, "values")
    size = vector.size
    levels = np.empty(size, dtype=np.float64)
    weights = np.empty(size, dtype=np.int64)
    block_count = 0

    for value in vector:
        levels[block_count] = value
        weights[block_count] = 1
        block_count += 1
        while block_count >= 2 and levels[block_count - 2] > levels[block_count - 1]:
            left = block_count - 2
            merged_weight = weights[left] + weights[left + 1]
            levels[left] = (
                levels[left] * weights[left] + levels[left + 1] * weights[left + 1]
            ) / merged_weight
            weights[left] = merged_weight
            block_count -= 1

    projected = np.empty(size, dtype=np.float64)
    offset = 0
    for block in range(block_count):
        block_size = int(weights[block])
        projected[offset : offset + block_size] = levels[block]
        offset += block_size
    assert offset == size
    assert np.all(projected[:-1] <= projected[1:])
    return projected

def pava_nonincreasing(values: ArrayLike) -> FloatArray:
    """Return the exact unweighted L2 projection onto nonincreasing vectors."""

    return -pava_nondecreasing(-_as_finite_vector(values, "values"))

def project_pointwise_order(
    lower: ArrayLike, upper: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Project paired values onto ``lower[g] <= upper[g]`` independently."""

    lower_vector = _as_finite_vector(lower, "lower")
    upper_vector = _as_finite_vector(upper, "upper")
    assert lower_vector.shape == upper_vector.shape
    crossed = lower_vector > upper_vector
    midpoint = 0.5 * (lower_vector[crossed] + upper_vector[crossed])
    lower_vector[crossed] = midpoint
    upper_vector[crossed] = midpoint
    assert np.all(lower_vector <= upper_vector)
    return lower_vector, upper_vector

def structural_violations(lower: ArrayLike, upper: ArrayLike) -> StructuralViolations:
    """Measure positive violations without applying a numerical tolerance."""

    lower_vector = _as_finite_vector(lower, "lower")
    upper_vector = _as_finite_vector(upper, "upper")
    assert lower_vector.shape == upper_vector.shape
    lower_violation = (
        float(np.maximum(lower_vector[1:] - lower_vector[:-1], 0.0).max())
        if lower_vector.size > 1
        else 0.0
    )
    upper_violation = (
        float(np.maximum(upper_vector[:-1] - upper_vector[1:], 0.0).max())
        if upper_vector.size > 1
        else 0.0
    )
    ordering_violation = float(np.maximum(lower_vector - upper_vector, 0.0).max())
    return StructuralViolations(
        lower_nonincreasing=lower_violation,
        upper_nondecreasing=upper_violation,
        pointwise_order=ordering_violation,
    )

def _project_component(
    vector: FloatArray, component: int, grid_size: int
) -> FloatArray:
    projected = vector.copy()
    if component == 0:
        projected[:grid_size] = pava_nonincreasing(projected[:grid_size])
    elif component == 1:
        projected[grid_size:] = pava_nondecreasing(projected[grid_size:])
    else:
        assert component == 2
        lower, upper = project_pointwise_order(
            projected[:grid_size],
            projected[grid_size:],
        )
        projected[:grid_size] = lower
        projected[grid_size:] = upper
    return projected

def _component_fixed_point_residual(vector: FloatArray, grid_size: int) -> float:
    return max(
        float(
            np.linalg.norm(
                _project_component(vector, component, grid_size) - vector,
                ord=np.inf,
            )
        )
        for component in range(3)
    )

def project_joint_bounds(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    feasibility_tolerance: float = 1e-11,
    required_stable_cycles: int = 3,
    max_cycles: int = 100_000,
) -> JointProjectionResult:
    """Project one paired curve onto the joint structural constraint set.

    Convergence is accepted only after every one of the following remains
    below its declared threshold for ``required_stable_cycles`` complete
    cycles: iterate change, Dykstra-correction change, objective change,
    direct feasibility violation, and all component fixed-point residuals.
    The complete per-cycle ledger is returned and exhaustion is a loud error.
    """

    lower_vector = _as_finite_vector(lower, "lower")
    upper_vector = _as_finite_vector(upper, "upper")
    assert lower_vector.shape == upper_vector.shape
    assert absolute_tolerance > 0.0
    assert relative_tolerance >= 0.0
    assert feasibility_tolerance > 0.0
    assert required_stable_cycles >= 1
    assert max_cycles >= required_stable_cycles

    grid_size = lower_vector.size
    observed = np.concatenate([lower_vector, upper_vector])
    iterate = observed.copy()
    corrections = [np.zeros_like(iterate) for _ in range(3)]
    input_scale = max(1.0, float(np.linalg.norm(observed, ord=np.inf)))
    iterate_tolerance = absolute_tolerance + relative_tolerance * input_scale
    effective_feasibility_tolerance = max(
        feasibility_tolerance,
        relative_tolerance * input_scale,
    )
    previous_objective = float("inf")
    stable_cycles = 0
    ledger: list[DykstraCycle] = []

    for cycle in range(1, max_cycles + 1):
        previous_iterate = iterate.copy()
        previous_corrections = [correction.copy() for correction in corrections]

        for component in range(3):
            corrected = iterate + corrections[component]
            projected = _project_component(corrected, component, grid_size)
            corrections[component] = corrected - projected
            iterate = projected

        iterate_delta_linf = float(
            np.linalg.norm(iterate - previous_iterate, ord=np.inf)
        )
        iterate_delta_l2 = float(np.linalg.norm(iterate - previous_iterate))
        correction_delta_linf = max(
            float(np.linalg.norm(current - previous, ord=np.inf))
            for current, previous in zip(
                corrections,
                previous_corrections,
                strict=True,
            )
        )
        objective_squared = float(np.dot(iterate - observed, iterate - observed))
        objective_delta = abs(objective_squared - previous_objective)
        objective_tolerance = absolute_tolerance + relative_tolerance * max(
            1.0,
            objective_squared,
        )
        violations = structural_violations(
            iterate[:grid_size],
            iterate[grid_size:],
        )
        fixed_point_residual = _component_fixed_point_residual(iterate, grid_size)

        stable = (
            iterate_delta_linf <= iterate_tolerance
            and correction_delta_linf <= iterate_tolerance
            and objective_delta <= objective_tolerance
            and violations.maximum <= effective_feasibility_tolerance
            and fixed_point_residual <= effective_feasibility_tolerance
        )
        stable_cycles = stable_cycles + 1 if stable else 0
        ledger.append(
            DykstraCycle(
                cycle=cycle,
                iterate_delta_linf=iterate_delta_linf,
                iterate_delta_l2=iterate_delta_l2,
                correction_delta_linf=correction_delta_linf,
                objective_squared=objective_squared,
                objective_delta=objective_delta,
                feasibility_violation=violations.maximum,
                component_fixed_point_residual=fixed_point_residual,
                iterate_tolerance=iterate_tolerance,
                objective_tolerance=objective_tolerance,
                feasibility_tolerance=effective_feasibility_tolerance,
                consecutive_stable_cycles=stable_cycles,
            )
        )
        previous_objective = objective_squared
        if stable_cycles >= required_stable_cycles:
            result = JointProjectionResult(
                lower=iterate[:grid_size].copy(),
                upper=iterate[grid_size:].copy(),
                cycles=cycle,
                converged=True,
                ledger=tuple(ledger),
            )
            assert structural_violations(result.lower, result.upper).maximum <= (
                effective_feasibility_tolerance
            )
            return result

    final = ledger[-1]
    raise AssertionError(
        "Dykstra failed its strict convergence criterion: "
        f"cycles={max_cycles}, iterate_delta={final.iterate_delta_linf:.3e}, "
        f"correction_delta={final.correction_delta_linf:.3e}, "
        f"objective_delta={final.objective_delta:.3e}, "
        f"feasibility={final.feasibility_violation:.3e}, "
        f"component_residual={final.component_fixed_point_residual:.3e}"
    )
