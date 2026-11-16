import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


from setting_standard.fork_run_dir import fork_upstream
from setting_standard.run_manifest import record_pipeline_step
from setting_standard.causaldataset import (
    build_outcome_model_from_bundle,
    build_propensity_model_from_bundle,
    discover_raw_causal_artifacts,
    load_outcome_model_bundle,
    load_propensity_model_bundle,
    resolve_code_relative_path,
)
from setting_standard.gen_standard_syn import NormalizedOutcomeMLPWrapper, NormalizedPropensityMLPWrapper


def compute_dataset_normalization_stats(
    dataframe: pd.DataFrame,
    x_columns: List[str],
    eps: float = 1e-6,
    y_fit_column: str = "outcome",
) -> Dict:
    """Fit X statistics and the explicitly selected per-DGP outcome scale."""
    assert y_fit_column in {"outcome", "pooled_potential_outcomes"}, y_fit_column
    x_values = dataframe[x_columns].values.astype(np.float32)
    # Evaluation fits only observed outcomes; pooling is an explicit synthetic-prior choice.
    fit_values = (
        dataframe["outcome"].to_numpy(dtype=np.float32)
        if y_fit_column == "outcome"
        else dataframe[["y0", "y1"]].to_numpy(dtype=np.float32).reshape(-1)
    )

    x_mean = x_values.mean(axis=0)
    x_std = x_values.std(axis=0)
    x_std = np.where(x_std < eps, 1.0, x_std)

    y_mean = float(fit_values.mean())
    y_std = float(fit_values.std())
    if y_std < eps:
        y_std = 1.0

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_fit_column": y_fit_column,
    }


def normalize_dataset_dataframe(
    dataframe: pd.DataFrame,
    x_columns: List[str],
    stats: Dict,
) -> pd.DataFrame:
    df_normalized = dataframe.copy()

    df_normalized[x_columns] = (df_normalized[x_columns].values - stats["x_mean"]) / stats["x_std"]

    for column in ("outcome", "y0", "y1"):
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
        "y_fit_column": stats["y_fit_column"],
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
            "y_fit_column": stats["y_fit_column"],
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
    y_fit_column: str = "outcome",
) -> Dict:
    dataset_id = raw_artifact["dataset_id"]
    resolved_output_root_dir = resolve_code_relative_path(output_root_dir)
    dataframe = pd.read_csv(raw_artifact["dataset_path"])

    x_columns = sorted([column for column in dataframe.columns if column.startswith("x")], key=lambda name: int(name[1:]))
    stats = compute_dataset_normalization_stats(dataframe=dataframe, x_columns=x_columns, eps=eps, y_fit_column=y_fit_column)

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

    for target in (normalized_dataset_path, normalized_outcome_model_path,
                   normalized_propensity_model_path, normalized_metadata_path, scaler_path):
        assert not os.path.lexists(target), f"Refusing to overwrite {target}"

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


def _normalize_dataset_collection(
    run_dir: str,
    eps: float,
    y_fit_column: str,
) -> List[Dict]:
    resolved = resolve_code_relative_path(run_dir)

    # Input guards: raw subdirs must exist (written by step 1)
    assert os.path.isdir(os.path.join(resolved, "datasets_raw")), \
        f"datasets_raw not found under {resolved} — run gen_standard_syn.py first"
    assert os.path.isdir(os.path.join(resolved, "models_raw")), \
        f"models_raw not found under {resolved} — run gen_standard_syn.py first"
    assert os.path.isdir(os.path.join(resolved, "metadata_raw")), \
        f"metadata_raw not found under {resolved} — run gen_standard_syn.py first"

    # Output guards: normalized subdirs must not yet exist
    assert not os.path.exists(os.path.join(resolved, "datasets")), \
        f"datasets already exists under {resolved} — normalization already ran or run_dir is dirty"
    assert not os.path.exists(os.path.join(resolved, "models")), \
        f"models already exists under {resolved} — normalization already ran or run_dir is dirty"
    assert not os.path.exists(os.path.join(resolved, "metadata")), \
        f"metadata already exists under {resolved} — normalization already ran or run_dir is dirty"
    assert not os.path.exists(os.path.join(resolved, "scalers")), \
        f"scalers already exists under {resolved} — normalization already ran or run_dir is dirty"

    raw_artifacts = discover_raw_causal_artifacts(resolved)
    print(f"Found {len(raw_artifacts)} raw artifacts to normalize from {resolved}")

    normalized_artifacts = []
    for raw_artifact in tqdm(raw_artifacts, desc="Normalizing datasets"):
        normalized_artifact = normalize_artifact_dataset(
            raw_artifact=raw_artifact,
            output_root_dir=resolved,
            eps=eps,
            y_fit_column=y_fit_column,
        )
        normalized_artifacts.append(normalized_artifact)

    print(f"Normalization complete. Normalized artifacts saved to {resolved}")
    return normalized_artifacts



def normalize_dataset_collection(
    run_dir: str,
    eps: float = 1e-6,
    source_run_dir: Optional[str] = None,
    y_fit_column: str = "outcome",
) -> List[Dict]:
    """Prepare the destination run and record one complete normalization attempt."""
    assert y_fit_column in {"outcome", "pooled_potential_outcomes"}, y_fit_column
    resolved = resolve_code_relative_path(run_dir)
    if source_run_dir is not None:
        fork_upstream(
            source_run_dir=source_run_dir,
            dest_run_dir=resolved,
            step=2,
            dataset_ids=None,
        )

    resolved_arguments = {
        "run_dir": run_dir,
        "eps": eps,
        "source_run_dir": source_run_dir,
        "y_fit_column": y_fit_column,
    }
    with record_pipeline_step(
        run_dir=resolved,
        step="02_normalize",
        implementation="src/setting_standard/normalize_data.py",
        resolved_arguments=resolved_arguments,
        seeds={"uses_randomness": False},
    ) as attempt:
        normalized_artifacts = _normalize_dataset_collection(run_dir=resolved, eps=eps, y_fit_column=y_fit_column)
        attempt.set_completion(
            {
                "dataset_ids": [artifact["dataset_id"] for artifact in normalized_artifacts],
                "requested_datasets": len(normalized_artifacts),
                "completed_datasets": len(normalized_artifacts),
                "failed_datasets": 0,
            }
        )
        return normalized_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset normalization.

    --run_dir artifacts/priors/run_1 `
    --eps 1e-6 `

    """
    parser = argparse.ArgumentParser(description="Normalize raw datasets in-place within a run directory.")
    parser.add_argument("--run_dir", type=str, required=True,
        help="Run directory produced by gen_standard_syn.py. Reads *_raw subdirs, writes datasets/, models/, metadata/, scalers/.")
    parser.add_argument("--y_fit_column", choices=("outcome", "pooled_potential_outcomes"),
        default="outcome", help="Observed evaluation scale or synthetic pooled-prior scale.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum standard deviation floor used during normalization.")
    parser.add_argument("--source_run_dir", type=str, default=None,
        help="Optional existing run directory to fork from. When supplied, upstream subdirs "
             "(datasets_raw/, models_raw/, metadata_raw/) are hardlinked into run_dir before normalization runs.")
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "run_dir": args.run_dir,
        "eps": args.eps,
        "source_run_dir": args.source_run_dir,
        "y_fit_column": args.y_fit_column,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    normalize_dataset_collection(**config)


if __name__ == "__main__":
    main()
