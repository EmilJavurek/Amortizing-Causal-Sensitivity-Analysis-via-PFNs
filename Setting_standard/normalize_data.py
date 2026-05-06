import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from Setting_standard.causaldataset import (
    build_outcome_model_from_bundle,
    build_propensity_model_from_bundle,
    discover_causal_artifacts,
    load_outcome_model_bundle,
    load_propensity_model_bundle,
    resolve_code_relative_path,
)
from Setting_standard.gen_standard_syn import NormalizedOutcomeMLPWrapper, NormalizedPropensityMLPWrapper


def compute_dataset_normalization_stats(
    dataframe: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    eps: float = 1e-6,
) -> Dict:
    x_values = dataframe[x_columns].values.astype(np.float32)
    y_values = dataframe[y_columns].values.astype(np.float32).reshape(-1)

    x_mean = x_values.mean(axis=0)
    x_std = x_values.std(axis=0)
    x_std = np.where(x_std < eps, 1.0, x_std)

    y_mean = float(y_values.mean())
    y_std = float(y_values.std())
    if y_std < eps:
        y_std = 1.0

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_dataset_dataframe(
    dataframe: pd.DataFrame,
    x_columns: List[str],
    stats: Dict,
) -> pd.DataFrame:
    df_normalized = dataframe.copy()

    df_normalized[x_columns] = (df_normalized[x_columns].values - stats["x_mean"]) / stats["x_std"]

    for column in ["outcome", "y0", "y1"]:
        df_normalized[column] = (df_normalized[column].values - stats["y_mean"]) / stats["y_std"]

    df_normalized["ite"] = df_normalized["y1"] - df_normalized["y0"]

    return df_normalized


def create_normalized_model_bundle(raw_bundle: Dict, stats: Dict, raw_model_path: str, scaler_path: str) -> Dict:
    raw_model = build_outcome_model_from_bundle(raw_bundle, device="cpu", freeze=True)
    normalized_model = NormalizedOutcomeMLPWrapper(
        raw_model=raw_model,
        x_mean=torch.tensor(stats["x_mean"], dtype=torch.float32),
        x_std=torch.tensor(stats["x_std"], dtype=torch.float32),
        y_mean=torch.tensor([stats["y_mean"]], dtype=torch.float32),
        y_std=torch.tensor([stats["y_std"]], dtype=torch.float32),
    )
    normalized_model.eval()

    for parameter in normalized_model.parameters():
        parameter.requires_grad_(False)

    return {
        "model_class": "NormalizedOutcomeMLPWrapper",
        "input_dim_x": raw_bundle["input_dim_x"],
        "input_dim_a": raw_bundle["input_dim_a"],
        "latent_dim_u": raw_bundle["latent_dim_u"],
        "output_dim_y": raw_bundle["output_dim_y"],
        "num_layers": raw_bundle["num_layers"],
        "hidden_size": raw_bundle["hidden_size"],
        "activation_name": raw_bundle["activation_name"],
        "x_mean": stats["x_mean"].tolist(),
        "x_std": stats["x_std"].tolist(),
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "raw_model_path": raw_model_path,
        "scaler_path": scaler_path,
        "state_dict": normalized_model.state_dict(),
    }


def create_normalized_propensity_model_bundle(
    raw_bundle: Dict,
    stats: Dict,
    raw_model_path: str,
    scaler_path: str,
) -> Dict:
    raw_model = build_propensity_model_from_bundle(raw_bundle, device="cpu", freeze=True)
    normalized_propensity_model = NormalizedPropensityMLPWrapper(
        raw_model=raw_model,
        x_mean=torch.tensor(stats["x_mean"], dtype=torch.float32),
        x_std=torch.tensor(stats["x_std"], dtype=torch.float32),
    )
    normalized_propensity_model.eval()

    for parameter in normalized_propensity_model.parameters():
        parameter.requires_grad_(False)

    return {
        "model_class": "NormalizedPropensityMLPWrapper",
        "input_dim_x": raw_bundle["input_dim_x"],
        "output_dim_a": raw_bundle["output_dim_a"],
        "num_layers": raw_bundle["num_layers"],
        "hidden_size": raw_bundle["hidden_size"],
        "activation_name": raw_bundle["activation_name"],
        "x_mean": stats["x_mean"].tolist(),
        "x_std": stats["x_std"].tolist(),
        "raw_model_path": raw_model_path,
        "scaler_path": scaler_path,
        "state_dict": normalized_propensity_model.state_dict(),
    }


def save_normalization_stats(stats: Dict, dataset_id: int, filename: str) -> None:
    serializable = {
        "dataset_id": dataset_id,
        "x_mean": stats["x_mean"].tolist(),
        "x_std": stats["x_std"].tolist(),
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "y_fit_columns": ["y0", "y1"],
    }

    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def save_normalized_metadata(
    raw_artifact: Dict,
    normalized_dataset_path: str,
    normalized_outcome_model_path: str,
    normalized_propensity_model_path: str,
    normalized_metadata_path: str,
    scaler_path: str,
    stats: Dict,
) -> None:
    raw_metadata = raw_artifact["metadata"]

    normalized_metadata = {
        "dataset_id": raw_metadata["dataset_id"],
        "seed": raw_metadata["seed"],
        "num_samples": raw_metadata["num_samples"],
        "num_features": raw_metadata["num_features"],
        "latent_dim_u": raw_metadata["latent_dim_u"],
        "output_dim_y": raw_metadata["output_dim_y"],
        "dataset_path": normalized_dataset_path,
        "outcome_model_path": normalized_outcome_model_path,
        "propensity_model_path": normalized_propensity_model_path,
        "raw_dataset_path": raw_artifact["dataset_path"],
        "raw_outcome_model_path": raw_artifact["outcome_model_path"],
        "raw_propensity_model_path": raw_artifact["propensity_model_path"],
        "raw_metadata_path": raw_artifact["metadata_path"],
        "scaler_path": scaler_path,
        "representation": "normalized",
        "normalization": {
            "x_mean": stats["x_mean"].tolist(),
            "x_std": stats["x_std"].tolist(),
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "y_fit_columns": ["y0", "y1"],
            "normalized_columns": ["x*", "outcome", "y0", "y1", "ite"],
            "untouched_columns": ["treatment", "propensity", "u*"],
        },
        "covariate_generator": raw_metadata["covariate_generator"],
        "treatment_generator": raw_metadata["treatment_generator"],
        "outcome_generator": raw_metadata["outcome_generator"],
    }

    with open(normalized_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(normalized_metadata, handle, indent=2)


def normalize_artifact_dataset(
    raw_artifact: Dict,
    output_root_dir: str,
    eps: float = 1e-6,
) -> Dict:
    dataset_id = raw_artifact["dataset_id"]
    resolved_output_root_dir = resolve_code_relative_path(output_root_dir)
    dataframe = pd.read_csv(raw_artifact["dataset_path"])

    x_columns = sorted([column for column in dataframe.columns if column.startswith("x")], key=lambda name: int(name[1:]))
    stats = compute_dataset_normalization_stats(
        dataframe=dataframe,
        x_columns=x_columns,
        y_columns=["y0", "y1"],
        eps=eps,
    )

    df_normalized = normalize_dataset_dataframe(
        dataframe=dataframe,
        x_columns=x_columns,
        stats=stats,
    )

    datasets_dir = os.path.join(resolved_output_root_dir, "datasets")
    models_dir = os.path.join(resolved_output_root_dir, "models")
    metadata_dir = os.path.join(resolved_output_root_dir, "metadata")
    scalers_dir = os.path.join(resolved_output_root_dir, "scalers")

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)

    normalized_dataset_path = os.path.join(datasets_dir, f"synthetic_continous_dataset_{dataset_id}.csv")
    normalized_outcome_model_path = os.path.join(models_dir, f"outcome_bnn_{dataset_id}.pt")
    normalized_propensity_model_path = os.path.join(models_dir, f"propensity_bnn_{dataset_id}.pt")
    normalized_metadata_path = os.path.join(metadata_dir, f"dgp_{dataset_id}.json")
    scaler_path = os.path.join(scalers_dir, f"scaler_{dataset_id}.json")

    df_normalized.to_csv(normalized_dataset_path, index=False)
    save_normalization_stats(stats=stats, dataset_id=dataset_id, filename=scaler_path)

    raw_bundle = load_outcome_model_bundle(raw_artifact["outcome_model_path"], map_location="cpu")
    normalized_bundle = create_normalized_model_bundle(
        raw_bundle=raw_bundle,
        stats=stats,
        raw_model_path=raw_artifact["outcome_model_path"],
        scaler_path=scaler_path,
    )
    torch.save(normalized_bundle, normalized_outcome_model_path)

    raw_propensity_bundle = load_propensity_model_bundle(raw_artifact["propensity_model_path"], map_location="cpu")
    normalized_propensity_bundle = create_normalized_propensity_model_bundle(
        raw_bundle=raw_propensity_bundle,
        stats=stats,
        raw_model_path=raw_artifact["propensity_model_path"],
        scaler_path=scaler_path,
    )
    torch.save(normalized_propensity_bundle, normalized_propensity_model_path)

    save_normalized_metadata(
        raw_artifact=raw_artifact,
        normalized_dataset_path=normalized_dataset_path,
        normalized_outcome_model_path=normalized_outcome_model_path,
        normalized_propensity_model_path=normalized_propensity_model_path,
        normalized_metadata_path=normalized_metadata_path,
        scaler_path=scaler_path,
        stats=stats,
    )

    return {
        "dataset_id": dataset_id,
        "normalized_dataset_path": normalized_dataset_path,
        "normalized_outcome_model_path": normalized_outcome_model_path,
        "normalized_propensity_model_path": normalized_propensity_model_path,
        "normalized_metadata_path": normalized_metadata_path,
        "scaler_path": scaler_path,
    }


def normalize_dataset_collection(
    raw_root_dir: str,
    output_root_dir: str,
    eps: float = 1e-6,
) -> List[Dict]:
    resolved_raw_root_dir = resolve_code_relative_path(raw_root_dir)
    resolved_output_root_dir = resolve_code_relative_path(output_root_dir)
    raw_artifacts = discover_causal_artifacts(resolved_raw_root_dir)
    print(f"Found {len(raw_artifacts)} raw artifacts to normalize from {resolved_raw_root_dir}")

    os.makedirs(resolved_output_root_dir, exist_ok=False)

    normalized_artifacts = []
    for raw_artifact in tqdm(raw_artifacts, desc="Normalizing datasets"):
        normalized_artifact = normalize_artifact_dataset(
            raw_artifact=raw_artifact,
            output_root_dir=resolved_output_root_dir,
            eps=eps,
        )
        normalized_artifacts.append(normalized_artifact)

    print(f"Normalization complete. Normalized artifacts saved to {resolved_output_root_dir}")
    return normalized_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset normalization.
    
    """
    parser = argparse.ArgumentParser(description="Normalize synthetic dataset collections and wrap outcome models in normalized coordinates.")
    parser.add_argument("--raw-root-dir", type=str, required=True,
        help="Input artifact root containing raw datasets, models, and metadata.")
    parser.add_argument("--output-root-dir", type=str, required=True,
        help="Output artifact root for normalized datasets, models, metadata, and scalers.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum standard deviation floor used during normalization.")
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "raw_root_dir": args.raw_root_dir,
        "output_root_dir": args.output_root_dir,
        "eps": args.eps,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    normalize_dataset_collection(**config)


if __name__ == "__main__":
    main()
