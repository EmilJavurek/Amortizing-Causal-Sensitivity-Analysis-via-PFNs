import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Callable


import numpy as np  
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from Setting_standard.gen_standard_syn import (
    NormalizedOutcomeMLPWrapper,
    NormalizedPropensityMLPWrapper,
    OutcomeMLP,
    PropensityMLP,
)


def resolve_code_relative_path(path: str) -> str:
    """
    Resolve artifact paths relative to the repository's code directory.

    Example:
    - `Setting_standard/training_data` -> `<repo>/code/Setting_standard/training_data`
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(CODE_ROOT, path))


def _extract_dataset_id(path: str, prefix: str) -> int:
    filename = os.path.basename(path)
    match = re.match(rf"{re.escape(prefix)}(\d+)", filename)
    assert match is not None, f"Could not extract dataset id from {filename}"
    return int(match.group(1))


def _sorted_paths_with_ids(paths: List[str], prefix: str) -> Dict[int, str]:
    mapping = {}
    for path in paths:
        dataset_id = _extract_dataset_id(path, prefix)
        mapping[dataset_id] = path
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


def discover_causal_artifacts(root_dir: str) -> List[Dict]:
    """
    Discover aligned dataset / metadata / model artifacts produced by
    gen_standard_syn.py and return them as a sorted artifact index.
    """
    resolved_root_dir = resolve_code_relative_path(root_dir)
    datasets_dir = os.path.join(resolved_root_dir, "datasets")
    metadata_dir = os.path.join(resolved_root_dir, "metadata")
    models_dir = os.path.join(resolved_root_dir, "models")

    dataset_paths = glob.glob(os.path.join(datasets_dir, "*.csv"))
    metadata_paths = glob.glob(os.path.join(metadata_dir, "*.json"))
    outcome_model_paths = glob.glob(os.path.join(models_dir, "outcome_bnn_*.pt"))
    propensity_model_paths = glob.glob(os.path.join(models_dir, "propensity_bnn_*.pt"))

    dataset_map = _sorted_paths_with_ids(dataset_paths, "synthetic_continous_dataset_")
    metadata_map = _sorted_paths_with_ids(metadata_paths, "dgp_")
    outcome_model_map = _sorted_paths_with_ids(outcome_model_paths, "outcome_bnn_")
    propensity_model_map = _sorted_paths_with_ids(propensity_model_paths, "propensity_bnn_")

    assert len(dataset_map) > 0, f"No dataset CSVs found under {datasets_dir}"
    assert set(dataset_map.keys()) == set(metadata_map.keys()), "Dataset and metadata ids do not match"
    assert set(dataset_map.keys()) == set(outcome_model_map.keys()), "Dataset and outcome model ids do not match"
    assert set(dataset_map.keys()) == set(propensity_model_map.keys()), "Dataset and propensity model ids do not match"

    artifacts = []
    for dataset_id in dataset_map.keys():
        metadata_path = metadata_map[dataset_id]
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        dataset_path = dataset_map[dataset_id]
        outcome_model_path = outcome_model_map[dataset_id]
        propensity_model_path = propensity_model_map[dataset_id]

        metadata_dataset_path = os.path.basename(metadata["dataset_path"])
        metadata_outcome_model_path = os.path.basename(metadata["outcome_model_path"])
        metadata_propensity_model_path = os.path.basename(metadata["propensity_model_path"])

        assert metadata["dataset_id"] == dataset_id, f"Metadata id mismatch for {metadata_path}"
        assert metadata_dataset_path == os.path.basename(dataset_path), f"Dataset path mismatch for {metadata_path}"
        assert metadata_outcome_model_path == os.path.basename(outcome_model_path), f"Outcome model path mismatch for {metadata_path}"
        assert metadata_propensity_model_path == os.path.basename(propensity_model_path), f"Propensity model path mismatch for {metadata_path}"

        artifacts.append(
            {
                "dataset_id": dataset_id,
                "dataset_path": dataset_path,
                "metadata_path": metadata_path,
                "outcome_model_path": outcome_model_path,
                "propensity_model_path": propensity_model_path,
                "metadata": metadata,
            }
        )

    return artifacts


def load_outcome_model_bundle(model_path: str, map_location: str = "cpu") -> Dict:
    resolved_model_path = resolve_code_relative_path(model_path)
    return torch.load(resolved_model_path, map_location=map_location, weights_only=False)


def load_propensity_model_bundle(model_path: str, map_location: str = "cpu") -> Dict:
    resolved_model_path = resolve_code_relative_path(model_path)
    return torch.load(resolved_model_path, map_location=map_location, weights_only=False)


def build_outcome_model_from_bundle(
    bundle: Dict,
    device: str = "cpu",
    freeze: bool = True,
) -> torch.nn.Module:
    model_class = bundle["model_class"]

    if model_class == "OutcomeMLP":
        input_size = bundle["input_dim_x"] + bundle["input_dim_a"] + bundle["latent_dim_u"]
        model = OutcomeMLP(
            input_size=input_size,
            hidden_size=bundle["hidden_size"],
            num_layers=bundle["num_layers"],
            activation_name=bundle["activation_name"],
        ).to(device)
        model.load_state_dict(bundle["state_dict"])

    elif model_class == "NormalizedOutcomeMLPWrapper":
        raw_model = OutcomeMLP(
            input_size=bundle["input_dim_x"] + bundle["input_dim_a"] + bundle["latent_dim_u"],
            hidden_size=bundle["hidden_size"],
            num_layers=bundle["num_layers"],
            activation_name=bundle["activation_name"],
        ).to(device)

        model = NormalizedOutcomeMLPWrapper(
            raw_model=raw_model,
            x_mean=torch.tensor(bundle["x_mean"], dtype=torch.float32, device=device),
            x_std=torch.tensor(bundle["x_std"], dtype=torch.float32, device=device),
            y_mean=torch.tensor([bundle["y_mean"]], dtype=torch.float32, device=device),
            y_std=torch.tensor([bundle["y_std"]], dtype=torch.float32, device=device),
        ).to(device)
        model.load_state_dict(bundle["state_dict"])

    else:
        assert False, f"Unsupported model class: {model_class}"

    model.eval()

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return model


def build_propensity_model_from_bundle(
    bundle: Dict,
    device: str = "cpu",
    freeze: bool = True,
) -> torch.nn.Module:
    model_class = bundle["model_class"]

    if model_class == "PropensityMLP":
        model = PropensityMLP(
            input_size=bundle["input_dim_x"],
            hidden_size=bundle["hidden_size"],
            num_layers=bundle["num_layers"],
            activation_name=bundle["activation_name"],
        ).to(device)
        model.load_state_dict(bundle["state_dict"])

    elif model_class == "NormalizedPropensityMLPWrapper":
        raw_model = PropensityMLP(
            input_size=bundle["input_dim_x"],
            hidden_size=bundle["hidden_size"],
            num_layers=bundle["num_layers"],
            activation_name=bundle["activation_name"],
        ).to(device)

        model = NormalizedPropensityMLPWrapper(
            raw_model=raw_model,
            x_mean=torch.tensor(bundle["x_mean"], dtype=torch.float32, device=device),
            x_std=torch.tensor(bundle["x_std"], dtype=torch.float32, device=device),
        ).to(device)
        model.load_state_dict(bundle["state_dict"])

    else:
        assert False, f"Unsupported model class: {model_class}"

    model.eval()

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return model


class _BaseCausalDataset(Dataset):
    """
    Shared artifact-index base class for -style generated datasets.
    """

    def __init__(self, root_dir: str, transform=None, log_fn: Callable[[str], None] = lambda msg: None):
        self.root_dir = resolve_code_relative_path(root_dir)
        self.transform = transform
        self.artifacts = discover_causal_artifacts(self.root_dir)
        self.log_fn = log_fn

        self.log_fn(f"Found {len(self.artifacts)} aligned DGP artifacts under {self.root_dir}")
        for artifact in self.artifacts:
            self.log_fn(f" - dataset_id={artifact['dataset_id']} | {artifact['dataset_path']}")

    def __len__(self) -> int:
        return len(self.artifacts)

    def _load_dataframe(self, idx: int) -> pd.DataFrame:
        artifact = self.artifacts[idx]
        dataframe = pd.read_csv(artifact["dataset_path"])
        return dataframe

    def _get_feature_columns(self, dataframe: pd.DataFrame) -> List[str]:
        return sorted([column for column in dataframe.columns if column.startswith("x")], key=lambda name: int(name[1:]))

    def _apply_transform(self, sample: Dict) -> Dict:
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CausalTrainDataset(_BaseCausalDataset):
    """
    Training-facing dataset. Only exposes X, a, y tensors.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataframe = self._load_dataframe(idx)
        x_cols = self._get_feature_columns(dataframe)

        sample = {
            "X": torch.FloatTensor(dataframe[x_cols].values),
            "a": torch.FloatTensor(dataframe["treatment"].values).unsqueeze(1),
            "y": torch.FloatTensor(dataframe["outcome"].values).unsqueeze(1),
        }

        return self._apply_transform(sample)


class CausalFrontierDataset(_BaseCausalDataset):
    """
    Frontier-facing dataset. Exposes X, a, y tensors and frozen SCM modules.
    """

    def __init__(self, root_dir: str, transform=None, log_fn: Callable[[str], None] = lambda msg: None, model_device: str = "cpu", ):
        super().__init__(root_dir=root_dir, transform=transform, log_fn=log_fn)
        self.model_device = model_device
        self.outcome_model_cache: Dict[int, torch.nn.Module] = {}
        self.propensity_model_cache: Dict[int, torch.nn.Module] = {}

    def _load_outcome_model(self, idx: int) -> torch.nn.Module:
        artifact = self.artifacts[idx]
        dataset_id = artifact["dataset_id"]

        if dataset_id not in self.outcome_model_cache:
            bundle = load_outcome_model_bundle(artifact["outcome_model_path"], map_location=self.model_device)
            model = build_outcome_model_from_bundle(
                bundle=bundle,
                device=self.model_device,
                freeze=True,
            )
            self.outcome_model_cache[dataset_id] = model

        return self.outcome_model_cache[dataset_id]

    def _load_propensity_model(self, idx: int) -> torch.nn.Module:
        artifact = self.artifacts[idx]
        dataset_id = artifact["dataset_id"]

        if dataset_id not in self.propensity_model_cache:
            bundle = load_propensity_model_bundle(artifact["propensity_model_path"], map_location=self.model_device)
            model = build_propensity_model_from_bundle(
                bundle=bundle,
                device=self.model_device,
                freeze=True,
            )
            self.propensity_model_cache[dataset_id] = model

        return self.propensity_model_cache[dataset_id]

    def __getitem__(self, idx: int) -> Dict:
        dataframe = self._load_dataframe(idx)
        x_cols = self._get_feature_columns(dataframe)
        artifact = self.artifacts[idx]

        sample = {
            "X": torch.FloatTensor(dataframe[x_cols].values),
            "a": torch.FloatTensor(dataframe["treatment"].values).unsqueeze(1),
            "y": torch.FloatTensor(dataframe["outcome"].values).unsqueeze(1),
            "f_BNN": self._load_outcome_model(idx),
            "f_A": self._load_propensity_model(idx),
            "dataset_id": artifact["dataset_id"],
            "metadata": artifact["metadata"],
            "metadata_path": artifact["metadata_path"],
            "outcome_model_path": artifact["outcome_model_path"],
            "propensity_model_path": artifact["propensity_model_path"],
        }

        return self._apply_transform(sample)


class CausalDiagnosticDataset(_BaseCausalDataset):
    """
    Diagnostic-facing dataset. Exposes all saved tensors, metadata, and model path.
    """

    def __getitem__(self, idx: int) -> Dict:
        dataframe = self._load_dataframe(idx)
        x_cols = self._get_feature_columns(dataframe)
        u_cols = sorted(
            [column for column in dataframe.columns if re.fullmatch(r"u\d+", column)],
            key=lambda name: int(name[1:]),
        )
        artifact = self.artifacts[idx]

        sample = {
            "X": torch.FloatTensor(dataframe[x_cols].values),
            "a": torch.FloatTensor(dataframe["treatment"].values).unsqueeze(1),
            "y": torch.FloatTensor(dataframe["outcome"].values).unsqueeze(1),
            "y0": torch.FloatTensor(dataframe["y0"].values).unsqueeze(1),
            "y1": torch.FloatTensor(dataframe["y1"].values).unsqueeze(1),
            "ite": torch.FloatTensor(dataframe["ite"].values).unsqueeze(1),
            "u": torch.FloatTensor(dataframe[u_cols].values),
            "dataset_id": artifact["dataset_id"],
            "metadata": artifact["metadata"],
            "metadata_path": artifact["metadata_path"],
            "outcome_model_path": artifact["outcome_model_path"],
            "propensity_model_path": artifact["propensity_model_path"],
        }

        if "normalization" in artifact["metadata"]:
            sample["normalization"] = artifact["metadata"]["normalization"]

        return self._apply_transform(sample)


def collate_train(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    X = torch.stack([item["X"] for item in batch], dim=1)
    a = torch.stack([item["a"] for item in batch], dim=1)
    y = torch.stack([item["y"] for item in batch], dim=1)

    return {
        "X": X,
        "a": a,
        "y": y,
    }


def collate_frontier(batch: List[Dict]) -> Dict:
    X = torch.stack([item["X"] for item in batch], dim=1)
    a = torch.stack([item["a"] for item in batch], dim=1)
    y = torch.stack([item["y"] for item in batch], dim=1)

    return {
        "X": X,
        "a": a,
        "y": y,
        "f_BNN": [item["f_BNN"] for item in batch],
        "f_A": [item["f_A"] for item in batch],
        "dataset_id": [item["dataset_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "metadata_path": [item["metadata_path"] for item in batch],
        "outcome_model_path": [item["outcome_model_path"] for item in batch],
        "propensity_model_path": [item["propensity_model_path"] for item in batch],
    }


def collate_diagnostic(batch: List[Dict]) -> Dict:
    X = torch.stack([item["X"] for item in batch], dim=1)
    a = torch.stack([item["a"] for item in batch], dim=1)
    y = torch.stack([item["y"] for item in batch], dim=1)
    y0 = torch.stack([item["y0"] for item in batch], dim=1)
    y1 = torch.stack([item["y1"] for item in batch], dim=1)
    ite = torch.stack([item["ite"] for item in batch], dim=1)
    u = torch.stack([item["u"] for item in batch], dim=1)

    batch_dict = {
        "X": X,
        "a": a,
        "y": y,
        "y0": y0,
        "y1": y1,
        "ite": ite,
        "u": u,
        "dataset_id": [item["dataset_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "metadata_path": [item["metadata_path"] for item in batch],
        "outcome_model_path": [item["outcome_model_path"] for item in batch],
        "propensity_model_path": [item["propensity_model_path"] for item in batch],
    }

    if "normalization" in batch[0]:
        batch_dict["normalization"] = [item["normalization"] for item in batch]

    return batch_dict


def create_causal_train_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: Optional[int] = None,
):
    dataset = CausalTrainDataset(root_dir)

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_train,
        worker_init_fn=_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_train,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


def create_causal_frontier_loader(
    root_dir: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    model_device: str = "cpu",
):
    dataset = CausalFrontierDataset(root_dir=root_dir, model_device=model_device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_frontier,
    )

    return loader


def create_causal_diagnostic_loader(
    root_dir: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
):
    dataset = CausalDiagnosticDataset(root_dir=root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_diagnostic,
    )

    return loader



# ------------------------------------------
# Additions for PI-FM Training specifically.
# ------------------------------------------


# ── Artifact discovery ───────────────────────────────────────────────────────

def discover_pifm_artifacts(root_dir: str) -> List[Dict]:
    """
    Discover aligned dataset / queries / frontier CSVs for PI FM training.

    Expected directory layout under root_dir:
        datasets/   synthetic_continous_dataset_{id}.csv
        queries/    queries_{id}.csv
        frontiers/  frontier_points_{id}.csv

    Returns a list of artifact dicts sorted by dataset_id, each containing:
        dataset_id   : int
        dataset_path : str   (absolute)
        queries_path : str   (absolute)
        frontier_path: str   (absolute)

    Asserts that dataset / queries / frontier IDs form identical sets.
    """
    resolved = resolve_code_relative_path(root_dir)

    dataset_paths  = glob.glob(os.path.join(resolved, "datasets",  "*.csv"))
    queries_paths  = glob.glob(os.path.join(resolved, "queries",   "*.csv"))
    frontier_paths = glob.glob(os.path.join(resolved, "frontiers", "*.csv"))

    dataset_map  = _sorted_paths_with_ids(dataset_paths,  "synthetic_continous_dataset_")
    queries_map  = _sorted_paths_with_ids(queries_paths,  "queries_")
    frontier_map = _sorted_paths_with_ids(frontier_paths, "frontier_points_")

    assert len(dataset_map)  > 0, f"No dataset CSVs found under {resolved}/datasets"
    assert set(dataset_map.keys()) == set(queries_map.keys()),  \
        "Dataset and queries IDs do not match"
    assert set(dataset_map.keys()) == set(frontier_map.keys()), \
        "Dataset and frontier IDs do not match"

    return [
        {
            "dataset_id":    did,
            "dataset_path":  dataset_map[did],
            "queries_path":  queries_map[did],
            "frontier_path": frontier_map[did],
        }
        for did in dataset_map
    ]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_joined_frontier(queries_path: str, frontier_path: str) -> pd.DataFrame:
    """
    Load queries_{id}.csv and frontier_points_{id}.csv, inner-join on query_id.

    Input schemas:
        queries:   query_id, source_row_index, a_query, x_0 .. x_p
        frontier:  query_id, bound_type, gamma_star, theta_star

    Output columns (in order):
        query_id, source_row_index, a_query, x_0 .. x_p,
        bound_type, gamma_star, theta_star

    bound_type is stored as str ("upper"/"lower") in the CSV; kept as-is here,
    converted to int (0=upper, 1=lower) in __getitem__.

    Asserts no NaNs in gamma_star or theta_star after join.
    Asserts that every query_id in frontier appears in queries.
    """
    queries  = pd.read_csv(queries_path)
    queries  = _normalize_x_columns(queries)
    frontier = pd.read_csv(frontier_path)

    assert set(frontier["query_id"].unique()).issubset(set(queries["query_id"].unique())), \
        f"frontier contains query_ids absent from queries: {queries_path}"

    joined = frontier.merge(queries, on="query_id", how="inner")

    assert not joined["gamma_star"].isna().any(),  "NaN in gamma_star after join"
    assert not joined["theta_star"].isna().any(),  "NaN in theta_star after join"

    return joined

def _normalize_x_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename bare x-columns (x0, x1, ...) to underscored form (x_0, x_1, ...).
    
    Handles mixed-convention CSVs from different subsystems.
    Columns already in x_N form are left untouched.
    """
    rename_map = {
        col: f"x_{col[1:]}"
        for col in df.columns
        if re.fullmatch(r"x\d+", col)
    }
    return df.rename(columns=rename_map) if rename_map else df

def _get_x_columns(df: pd.DataFrame) -> List[str]:
    """Return sorted x_0..x_p column names from a DataFrame."""
    return sorted(
        [c for c in df.columns if re.fullmatch(r"x_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )


# ── Dataset ──────────────────────────────────────────────────────────────────

class PIFMDataset(Dataset):
    """
    Dataset for PI Foundation Model training.

    Each item corresponds to one DGP M* and yields:
      - the full observational context D  (X, a_context, y_context)
      - m = m_prime * g query rows sampled from the Pareto frontier

    Sampling procedure in __getitem__:
      1. Group joined frontier rows by (query_id).
         Each group has exactly G rows (G gamma points), split across
         bound_type values "upper" and "lower".
      2. Sample m_prime groups without replacement.
      3. Within each selected group, sample g rows without replacement
         (across both bound types jointly — do NOT stratify by bound_type
         at this stage; imbalance within a DGP is fine in expectation).
      4. Concatenate → m = m_prime * g rows, order randomised.

    Shapes returned by __getitem__:
        X                : (n, d_x)    float32
        a_context        : (n, 1)      float32
        y_context        : (n, 1)      float32
        x_query          : (m, d_x)   float32
        a_query          : (m, 1)      float32
        gamma            : (m, 1)      float32
        theta_star       : (m, 1)      float32
        bound_type       : (m,)        int64   0=upper 1=lower
        query_id         : (m,)        int64   for diagnostics
        source_row_index : (m,)        int64   for diagnostics
    """

    BOUND_TYPE_MAP: Dict[str, int] = {"upper": 0, "lower": 1}

    def __init__(
        self,
        root_dir: str,
        m_prime: int,
        g: int,
        log_fn: Callable[[str], None] = lambda msg: None,
    ):
        """
        Args:
            root_dir:  path to the PI FM data root (resolved relative to CODE_ROOT
                       if not absolute, following existing convention).
            m_prime:   number of (query_id) groups to subsample per DGP per step.
            g:         number of gamma-frontier rows to subsample per group.
            log_fn:    optional logging callback.
        """
        self.root_dir = resolve_code_relative_path(root_dir)
        self.m_prime  = m_prime
        self.g        = g
        self.log_fn   = log_fn

        self.artifacts: List[Dict] = discover_pifm_artifacts(self.root_dir)

        # Lazy caches: keyed by dataset_id. Intentionally not pre-loaded — with O(10k)
        # DGPs the full set would be tens of GB. Each DataLoader worker gets a forked
        # copy and populates it independently (no shared state, but re-reads per worker
        # for the same DGP). If I/O is a bottleneck, pre-join/pre-read to parquet.
        self._frontier_cache: Dict[int, pd.DataFrame] = {}
        self._dataset_cache:  Dict[int, pd.DataFrame] = {}

        self.log_fn(f"PIFMDataset: {len(self.artifacts)} DGPs under {self.root_dir}")
        self.log_fn(f"  m_prime={m_prime}, g={g}, m={m_prime * g} query rows per item")

        # Assert all DGPs have the same number of context rows so collate_pifm can stack.
        _row_counts = [len(pd.read_csv(a["dataset_path"])) for a in self.artifacts]
        assert len(set(_row_counts)) == 1, f"DGPs have inconsistent row counts: {set(_row_counts)}"

    def __len__(self) -> int:
        return len(self.artifacts)

    def _get_frontier(self, idx: int) -> pd.DataFrame:
        """Return cached joined frontier DataFrame for artifact at idx."""
        artifact   = self.artifacts[idx]
        dataset_id = artifact["dataset_id"]
        if dataset_id not in self._frontier_cache:
            self._frontier_cache[dataset_id] = _load_joined_frontier(
                artifact["queries_path"],
                artifact["frontier_path"],
            )
        return self._frontier_cache[dataset_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        artifact = self.artifacts[idx]

        # ── Context: full observational dataset D ─────────────────────────
        dataset_id = artifact["dataset_id"]
        if dataset_id not in self._dataset_cache:
            self._dataset_cache[dataset_id] = _normalize_x_columns(
                pd.read_csv(artifact["dataset_path"])
            )
        dataset_df = self._dataset_cache[dataset_id]
        x_cols = _get_x_columns(dataset_df)

        assert "treatment" in dataset_df.columns, \
            f"Missing 'treatment' column in {artifact['dataset_path']}"
        assert "outcome" in dataset_df.columns, \
            f"Missing 'outcome' column in {artifact['dataset_path']}"

        X         = torch.FloatTensor(dataset_df[x_cols].values)            # (n, d_x)
        a_context = torch.FloatTensor(dataset_df["treatment"].values).unsqueeze(1)  # (n,1)
        y_context = torch.FloatTensor(dataset_df["outcome"].values).unsqueeze(1)    # (n,1)

        # ── Query rows: sample from Pareto frontier ────────────────────────
        frontier = self._get_frontier(idx)   # full joined DataFrame, G rows per query_id

        all_query_ids = frontier["query_id"].unique()
        assert len(all_query_ids) >= self.m_prime, (
            f"DGP {artifact['dataset_id']}: only {len(all_query_ids)} unique query_ids "
            f"but m_prime={self.m_prime} requested"
        )

        # Step 1: sample m_prime query_ids without replacement
        selected_ids = np.random.choice(all_query_ids, size=self.m_prime, replace=False)

        # Step 2: for each selected query_id, sample g rows without replacement
        sampled_rows: List[pd.DataFrame] = []
        for qid in selected_ids:
            group = frontier[frontier["query_id"] == qid]
            assert len(group) >= self.g, (
                f"DGP {artifact['dataset_id']}, query_id {qid}: "
                f"only {len(group)} frontier rows but g={self.g} requested"
            )
            sampled_rows.append(group.sample(n=self.g, replace=False))

        selected = pd.concat(sampled_rows, ignore_index=True)
        # Shuffle so upper/lower/gamma ordering is not structured within the item.
        # REMOVED: destroys (m_prime, g) block structure required by monotonicity_penalty
        # in training_standard.py. Re-enable if monotonicity penalty is dropped.
        # selected = selected.sample(frac=1).reset_index(drop=True)

        assert len(selected) == self.m_prime * self.g, \
            f"Expected {self.m_prime * self.g} query rows, got {len(selected)}"

        qx_cols = _get_x_columns(selected)
        assert qx_cols == x_cols, \
            "x columns in frontier do not match x columns in dataset — schema mismatch"

        x_query    = torch.FloatTensor(selected[qx_cols].values)                   # (m, d_x)
        a_query    = torch.FloatTensor(selected["a_query"].values).unsqueeze(1)     # (m, 1)
        gamma      = torch.FloatTensor(selected["gamma_star"].values).unsqueeze(1)  # (m, 1)
        theta_star = torch.FloatTensor(selected["theta_star"].values).unsqueeze(1)  # (m, 1)

        bound_type_int = selected["bound_type"].map(self.BOUND_TYPE_MAP).values
        assert not pd.isna(bound_type_int).any(), \
            f"Unknown bound_type value in {artifact['frontier_path']}"
        bound_type       = torch.LongTensor(bound_type_int.astype(int))             # (m,)
        query_id_t       = torch.LongTensor(selected["query_id"].values)            # (m,)
        source_row_index = torch.LongTensor(selected["source_row_index"].values)    # (m,)

        return {
            "X":                X,
            "a_context":        a_context,
            "y_context":        y_context,
            "x_query":          x_query,
            "a_query":          a_query,
            "gamma":            gamma,
            "theta_star":       theta_star,
            "bound_type":       bound_type,
            "query_id":         query_id_t,
            "source_row_index": source_row_index,
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_pifm(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate B PIFMDataset items into a single batch dict.

    torch.stack(dim=1) inserts the batch dimension, preserving the
    sequence dimension as dim=0. This matches the (seq_len, B, features)
    convention used throughout the existing codebase.

    Output shapes (B = batch size, n = context rows, m = query rows):
        X                : (n, B, d_x)
        a_context        : (n, B, 1)
        y_context        : (n, B, 1)
        x_query          : (m, B, d_x)
        a_query          : (m, B, 1)
        gamma            : (m, B, 1)
        theta_star       : (m, B, 1)
        bound_type       : (m, B)
        query_id         : (m, B)       diagnostic — not used in loss
        source_row_index : (m, B)       diagnostic — not used in loss
    """
    return {
        "X":                torch.stack([item["X"]                for item in batch], dim=1),
        "a_context":        torch.stack([item["a_context"]        for item in batch], dim=1),
        "y_context":        torch.stack([item["y_context"]        for item in batch], dim=1),
        "x_query":          torch.stack([item["x_query"]          for item in batch], dim=1),
        "a_query":          torch.stack([item["a_query"]          for item in batch], dim=1),
        "gamma":            torch.stack([item["gamma"]            for item in batch], dim=1),
        "theta_star":       torch.stack([item["theta_star"]       for item in batch], dim=1),
        "bound_type":       torch.stack([item["bound_type"]       for item in batch], dim=1),
        "query_id":         torch.stack([item["query_id"]         for item in batch], dim=1),
        "source_row_index": torch.stack([item["source_row_index"] for item in batch], dim=1),
    }


# ── DataLoader factory ────────────────────────────────────────────────────────
def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % (2**32) + worker_id)


def create_pifm_data_loaders(
    root_dir: str,
    m_prime: int,
    g: int,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4,
    log_fn: Callable[[str], None] = lambda msg: None,
    seed: Optional[int] = None,
) -> tuple:
    """
    Create train and validation DataLoaders for PI FM training.

    Val split is performed at the DGP level (not the frontier row level):
    the model never sees D_{M*} from the validation DGPs during training,
    which is the correct generalisation test.

    Returns:
        train_loader, val_loader
    """
    dataset = PIFMDataset(root_dir=root_dir, m_prime=m_prime, g=g, log_fn=log_fn)

    n_total = len(dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    assert n_train > 0 and n_val > 0, \
        f"val_split={val_split} leaves empty train or val set for {n_total} DGPs"

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pifm,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pifm,
        worker_init_fn=_worker_init_fn,
    )

    log_fn(f"PIFMDataset split: {n_train} train DGPs / {n_val} val DGPs")
    return train_loader, val_loader
