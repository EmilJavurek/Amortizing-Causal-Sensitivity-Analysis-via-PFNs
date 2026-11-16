"""Shared evaluation computation; callers own result paths and serialization."""

from __future__ import annotations
import sys
from pathlib import Path
from experiments.shared.paths import CODE, ROOT, EXPERIMENTS, PRIORS, CHECKPOINTS
import json
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

CODE_DIR = CODE
PROJECT_ROOT = ROOT
MSM_CHECKPOINT = CHECKPOINTS / 'msm/model.pth'
KL_CHECKPOINT = CHECKPOINTS / 'kl/model.pth'
ROSENBAUM_CHECKPOINT = CHECKPOINTS / 'rosenbaum/model.pth'

def load_json(path: Path) -> dict:
    assert path.is_file(), f"Missing JSON file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))

def load_pifm(checkpoint_path: Path, device: torch.device):
    from tabpfn.model.causalFM import PerFeatureTransformerPIFM

    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model_state_dict"]
    n_components = int(state_dict["gmm_head_upper.fc_pi.weight"].shape[0])
    assert n_components == int(state_dict["gmm_head_lower.fc_pi.weight"].shape[0])

    model = PerFeatureTransformerPIFM(
        gmm_n_components=n_components,
        gmm_min_sigma=1e-3,
        gmm_pi_temp=1.0,
    ).to(device)
    model_keys = set(model.state_dict())
    unexpected = sorted(set(state_dict) - model_keys)
    allowed = [key for key in unexpected if key.startswith("decoder_dict.cate.")]
    assert unexpected == allowed, f"Unexpected checkpoint keys: {unexpected}"
    filtered = {key: value for key, value in state_dict.items() if key in model_keys}
    result = model.load_state_dict(filtered, strict=False)
    assert not result.missing_keys, f"Missing model keys: {result.missing_keys}"
    assert not result.unexpected_keys, f"Unexpected model keys: {result.unexpected_keys}"
    model.eval()
    return model, checkpoint

def normalized_dataset(artifact: dict) -> tuple[pd.DataFrame, list[str]]:
    from setting_standard.causaldataset import _get_x_columns, _normalize_x_columns

    frame = _normalize_x_columns(pd.read_csv(artifact["dataset_path"]))
    x_columns = _get_x_columns(frame)
    required = {"treatment", "outcome", "propensity"}
    assert required.issubset(frame.columns), (
        f"Dataset {artifact['dataset_id']} missing {required - set(frame.columns)}"
    )
    return frame, x_columns

def propensity_stratified_row_indices(
    frame: pd.DataFrame,
    count: int,
    quantiles: Iterable[float] | None = None,
) -> list[int]:
    assert count >= 1
    if quantiles is None:
        quantiles = np.linspace(0.02, 0.98, count)
    quantiles = np.asarray(list(quantiles), dtype=float)
    assert len(quantiles) == count
    assert np.all((quantiles >= 0.0) & (quantiles <= 1.0))

    order = np.argsort(frame["propensity"].to_numpy(), kind="stable")
    positions = np.rint(quantiles * (len(order) - 1)).astype(int)
    selected: list[int] = []
    for position in positions:
        index = int(order[position])
        if index not in selected:
            selected.append(index)
    if len(selected) < count:
        for index in order:
            value = int(index)
            if value not in selected:
                selected.append(value)
            if len(selected) == count:
                break
    assert len(selected) == count
    return selected

def make_query_grid(
    frame: pd.DataFrame,
    x_columns: list[str],
    row_indices: list[int],
    gamma_values: Iterable[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for source_row_index in row_indices:
        source = frame.iloc[source_row_index]
        p_a1 = float(source["propensity"])
        for arm in (0, 1):
            pi_query = p_a1 if arm == 1 else 1.0 - p_a1
            for gamma in gamma_values:
                row = {
                    "source_row_index": int(source_row_index),
                    "a_query": arm,
                    "gamma": float(gamma),
                    "p_a1": p_a1,
                    "pi_query": pi_query,
                    "counterfactual_weight": 1.0 - pi_query,
                }
                row.update({column: float(source[column]) for column in x_columns})
                rows.append(row)
    result = pd.DataFrame(rows)
    assert np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()
    return result

@torch.no_grad()
def pifm_predict_frame(
    model,
    frame: pd.DataFrame,
    x_columns: list[str],
    query_frame: pd.DataFrame,
    device: torch.device,
    chunk_size: int = 640,
) -> pd.DataFrame:
    X = torch.tensor(frame[x_columns].to_numpy(), dtype=torch.float32)
    A = torch.tensor(frame[["treatment"]].to_numpy(), dtype=torch.float32)
    Y = torch.tensor(frame[["outcome"]].to_numpy(), dtype=torch.float32)

    output_chunks: list[pd.DataFrame] = []
    for start in range(0, len(query_frame), chunk_size):
        chunk = query_frame.iloc[start : start + chunk_size].copy()
        prediction = model.estimate_capo_bounds(
            X_context=X,
            A_context=A,
            Y_context=Y,
            X_query=torch.tensor(chunk[x_columns].to_numpy(), dtype=torch.float32),
            A_query=torch.tensor(chunk[["a_query"]].to_numpy(), dtype=torch.float32),
            Gamma_query=torch.tensor(chunk[["gamma"]].to_numpy(), dtype=torch.float32),
        )
        for side in ("upper", "lower"):
            for parameter in ("pi", "mu", "sigma"):
                values = prediction[f"{side}_{parameter}"].detach().cpu().numpy()
                for component in range(values.shape[1]):
                    chunk[f"{side}_{parameter}_{component}"] = values[:, component]
            chunk[f"{side}_mean"] = (
                prediction[f"{side}_pi"] * prediction[f"{side}_mu"]
            ).sum(dim=-1).detach().cpu().numpy()
        output_chunks.append(chunk)
    result = pd.concat(output_chunks, ignore_index=True)
    assert len(result) == len(query_frame)
    return result

def gmm_arrays(frame: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = sorted(
        int(column.rsplit("_", 1)[1])
        for column in frame.columns
        if column.startswith(f"{side}_pi_")
    )
    assert components == list(range(len(components)))
    pi = frame[[f"{side}_pi_{k}" for k in components]].to_numpy()
    mu = frame[[f"{side}_mu_{k}" for k in components]].to_numpy()
    sigma = frame[[f"{side}_sigma_{k}" for k in components]].to_numpy()
    assert pi.shape == mu.shape == sigma.shape
    assert np.all(sigma > 0)
    return pi, mu, sigma

def gmm_cdf(
    values: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return np.sum(pi * ndtr((values - mu) / sigma), axis=1)

def gmm_quantiles(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    probabilities: Iterable[float],
    iterations: int = 64,
) -> np.ndarray:
    probabilities = np.asarray(list(probabilities), dtype=float)
    assert np.all((probabilities > 0) & (probabilities < 1))
    lower = np.min(mu - 12.0 * sigma, axis=1)
    upper = np.max(mu + 12.0 * sigma, axis=1)
    outputs: list[np.ndarray] = []
    for probability in probabilities:
        lo = lower.copy()
        hi = upper.copy()
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            cdf = gmm_cdf(mid, pi, mu, sigma)
            lo = np.where(cdf < probability, mid, lo)
            hi = np.where(cdf >= probability, mid, hi)
        outputs.append(0.5 * (lo + hi))
    return np.stack(outputs, axis=1)

def add_gmm_diagnostics(frame: pd.DataFrame, truth_column: str) -> pd.DataFrame:
    result = frame.copy()
    probability_labels = [
        (0.05, "q05"),
        (0.10, "q10"),
        (0.25, "q25"),
        (0.50, "q50"),
        (0.75, "q75"),
        (0.90, "q90"),
        (0.95, "q95"),
    ]
    for side in ("upper", "lower"):
        pi, mu, sigma = gmm_arrays(result, side)
        quantiles = gmm_quantiles(
            pi, mu, sigma, [probability for probability, _ in probability_labels]
        )
        for index, (_, label) in enumerate(probability_labels):
            result[f"{side}_{label}"] = quantiles[:, index]
        result[f"{side}_pit"] = gmm_cdf(
            result[truth_column].to_numpy(), pi, mu, sigma
        )
    return result
