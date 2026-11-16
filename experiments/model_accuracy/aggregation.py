"""Posterior-sample aggregation of separate CAPO heads."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.shared.evaluation import gmm_arrays

def sample_gmm_numpy(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    uniforms = rng.random((len(pi), samples))
    cumulative = np.cumsum(pi, axis=1)
    components = (uniforms[:, :, None] > cumulative[:, None, :]).sum(axis=2)
    components = np.minimum(components, pi.shape[1] - 1)
    selected_mu = np.take_along_axis(mu, components, axis=1)
    selected_sigma = np.take_along_axis(sigma, components, axis=1)
    return selected_mu + selected_sigma * rng.standard_normal((len(pi), samples))

def aggregation_table(
    predictions: pd.DataFrame,
    seed: int,
    samples: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for dataset_id, dgp in predictions.groupby("dataset_id"):
        rng = np.random.default_rng(seed + int(dataset_id))
        lower_pi, lower_mu, lower_sigma = gmm_arrays(dgp, "lower")
        upper_pi, upper_mu, upper_sigma = gmm_arrays(dgp, "upper")
        lower_samples = sample_gmm_numpy(
            lower_pi, lower_mu, lower_sigma, samples, rng
        )
        upper_samples = sample_gmm_numpy(
            upper_pi, upper_mu, upper_sigma, samples, rng
        )
        local = dgp[
            [
                "source_row_index",
                "a_query",
                "gamma",
                "theta_lower",
                "theta_upper",
            ]
        ].reset_index(drop=True)
        local["row_position"] = np.arange(len(local))

        for gamma, gamma_frame in local.groupby("gamma"):
            for arm, arm_frame in gamma_frame.groupby("a_query"):
                positions = arm_frame["row_position"].to_numpy()
                for side, draw_matrix, truth_column in (
                    ("lower", lower_samples, "theta_lower"),
                    ("upper", upper_samples, "theta_upper"),
                ):
                    draws = draw_matrix[positions].mean(axis=0)
                    truth = float(arm_frame[truth_column].mean())
                    records.append(
                        aggregate_record(
                            dataset_id,
                            "APO",
                            f"arm={int(arm)}",
                            gamma,
                            side,
                            truth,
                            draws,
                        )
                    )

            arms = gamma_frame.pivot(
                index="source_row_index",
                columns="a_query",
                values=["row_position", "theta_lower", "theta_upper"],
            )
            assert set(arms["row_position"].columns) == {0, 1}
            for source_row_index, row in arms.iterrows():
                pos0 = int(row[("row_position", 0)])
                pos1 = int(row[("row_position", 1)])
                lower_draws = lower_samples[pos1] - upper_samples[pos0]
                upper_draws = upper_samples[pos1] - lower_samples[pos0]
                truth_lower = float(
                    row[("theta_lower", 1)] - row[("theta_upper", 0)]
                )
                truth_upper = float(
                    row[("theta_upper", 1)] - row[("theta_lower", 0)]
                )
                records.append(
                    aggregate_record(
                        dataset_id,
                        "CATE",
                        f"row={int(source_row_index)}",
                        gamma,
                        "lower",
                        truth_lower,
                        lower_draws,
                    )
                )
                records.append(
                    aggregate_record(
                        dataset_id,
                        "CATE",
                        f"row={int(source_row_index)}",
                        gamma,
                        "upper",
                        truth_upper,
                        upper_draws,
                    )
                )

            pos0 = arms[("row_position", 0)].astype(int).to_numpy()
            pos1 = arms[("row_position", 1)].astype(int).to_numpy()
            ate_lower_draws = (
                lower_samples[pos1] - upper_samples[pos0]
            ).mean(axis=0)
            ate_upper_draws = (
                upper_samples[pos1] - lower_samples[pos0]
            ).mean(axis=0)
            truth_ate_lower = float(
                (
                    arms[("theta_lower", 1)]
                    - arms[("theta_upper", 0)]
                ).mean()
            )
            truth_ate_upper = float(
                (
                    arms[("theta_upper", 1)]
                    - arms[("theta_lower", 0)]
                ).mean()
            )
            records.append(
                aggregate_record(
                    dataset_id,
                    "ATE",
                    "all_rows",
                    gamma,
                    "lower",
                    truth_ate_lower,
                    ate_lower_draws,
                )
            )
            records.append(
                aggregate_record(
                    dataset_id,
                    "ATE",
                    "all_rows",
                    gamma,
                    "upper",
                    truth_ate_upper,
                    ate_upper_draws,
                )
            )
    return pd.DataFrame(records)

def aggregate_record(
    dataset_id: int,
    estimand: str,
    unit: str,
    gamma: float,
    side: str,
    truth: float,
    draws: np.ndarray,
) -> dict:
    quantiles = np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "dataset_id": dataset_id,
        "estimand": estimand,
        "unit": unit,
        "gamma": gamma,
        "side": side,
        "truth": truth,
        "mean": float(draws.mean()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "mc_samples": len(draws),
    }
