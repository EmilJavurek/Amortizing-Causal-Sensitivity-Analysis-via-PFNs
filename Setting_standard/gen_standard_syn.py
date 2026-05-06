import argparse
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)

def resolve_code_relative_path(path: str) -> str:
    """
    Resolve artifact paths relative to the repository's code directory.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(CODE_ROOT, path))


class DAGStructuredSCM:
    """
    Implementation of DAG-structured Structural Causal Models (SCMs) sampler
    based on a modified MLP architecture.
    """

    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 6),
        prior_hidden_size: Callable = lambda: np.random.randint(10, 50),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.4,
        activation: Callable = lambda x: np.tanh(x),
    ):
        """
        Initialize the DAG-structured SCM sampler.

        Args:
            prior_layers: Distribution for number of layers
            prior_hidden_size: Distribution for hidden size
            prior_weight: Distribution for edge weights
            edge_drop_prob: Probability of dropping an edge to create DAG
            activation: Activation function to use (default: tanh)
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.dag = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.feature_nodes = []
        self.node_values = {}
        self.topological_order = []
        self.sampled_num_layers = None
        self.sampled_hidden_size = None

    def sample_noise_distribution(self) -> Callable:
        """
        Sample a noise distribution from a meta-distribution.
        """
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])

        if dist_type == "uniform":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.uniform(-scale, scale, size)

        if dist_type == "laplace":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.laplace(0, scale, size)

        if dist_type == "logistic":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.logistic(0, scale, size)

        # Default to normal if none of the above
        scale = np.random.uniform(0.1, 2.0)
        return lambda size: np.random.normal(0, scale, size)

    def construct_mlp_graph(self, num_layers: int, hidden_size: int) -> nx.DiGraph:
        """
        Construct an MLP-like directed graph structure.

        Args:
            num_layers: Number of layers in the MLP
            hidden_size: Size of hidden layers

        Returns:
            A directed graph representing the MLP structure
        """
        graph = nx.DiGraph()

        node_id = 0
        layer_sizes = [hidden_size] * num_layers

        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                graph.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)

        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]

            for src in src_nodes:
                for dst in dst_nodes:
                    graph.add_edge(src, dst)

        return graph

    def transform_to_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        dag = graph.copy()

        edges = list(graph.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)

        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(len(edges), size=num_edges_to_drop, replace=False)

            for idx in edges_to_drop:
                u, v = edges[idx]
                dag.remove_edge(u, v)

        assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG after edge removal"

        return dag

    def sample_structural_equation_parameters(self) -> None:
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))

            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}

            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def evaluate_node(self, node: int) -> float:
        parents = list(self.dag.predecessors(node))

        if not parents:
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(self.biases[node] + noise)
        else:
            weighted_sum = sum(self.weights[node][parent] * self.node_values[parent] for parent in parents)
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(weighted_sum + self.biases[node] + noise)

        return value

    def sample_observation(self) -> np.ndarray:
        self.node_values = {}

        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)

        features = np.array([self.node_values[node] for node in self.feature_nodes])

        return features

    def generate_dataset(self, num_features: int, num_samples: int) -> np.ndarray:
        # Sample prior for DAG structure
        self.sampled_num_layers = self.prior_layers()
        self.sampled_hidden_size = self.prior_hidden_size()

        mlp_graph = self.construct_mlp_graph(self.sampled_num_layers, self.sampled_hidden_size)
        self.dag = self.transform_to_dag(mlp_graph)
        self.topological_order = list(nx.topological_sort(self.dag))

        # Sample structural equation parameters (weights, biases, noise) for each node
        self.sample_structural_equation_parameters()

        # Randomly select feature nodes from the DAG 
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes), "Requested more features than available nodes" # NOTE: How often is this an issue? Can I prevent?
        self.feature_nodes = np.random.choice(all_nodes, size=num_features, replace=False)

        # Sample observations
        dataset = np.zeros((num_samples, num_features))
        for i in range(num_samples):
            dataset[i] = self.sample_observation()

        return dataset


class OutcomeMLP(nn.Module):
    """
    PyTorch outcome network used as f_BNN(x, a, u).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        activation_name: str = "tanh",
    ):
        super().__init__()

        assert num_layers >= 3, "OutcomeMLP requires at least 3 layers (input, hidden, output)"
        assert activation_name in ["tanh"], "Unsupported activation function"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation_name = activation_name

        if activation_name == "tanh":
            self.activation = nn.Tanh()

        layers: List[nn.Module] = []
        current_size = input_size

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(self.activation)
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D tensors 
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if a.ndim == 0:
            a = a.unsqueeze(0)
        if a.ndim == 1:
            a = a.unsqueeze(1)
        if u.ndim == 1:
            u = u.unsqueeze(1)

        inputs = torch.cat([x, a, u], dim=1)
        return self.network(inputs)


class PropensityMLP(nn.Module):
    """
    PyTorch propensity network used as f_A(x) = P(A = 1 | X = x).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        activation_name: str = "tanh",
    ):
        super().__init__()

        assert num_layers >= 3, "PropensityMLP requires at least 3 layers (input, hidden, output)"
        assert activation_name in ["tanh"], "Unsupported activation function"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation_name = activation_name

        if activation_name == "tanh":
            self.activation = nn.Tanh()

        layers: List[nn.Module] = []
        current_size = input_size

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(self.activation)
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return torch.sigmoid(self.network(x))


class NormalizedOutcomeMLPWrapper(nn.Module):
    """
    Wrapper around the raw outcome model that exposes normalized input/output space.

    It expects normalized x as input, un-normalizes it before calling the raw
    f_BNN(x, a, u), and then re-normalizes the resulting y.
    """

    def __init__(
        self,
        raw_model: OutcomeMLP,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ):
        super().__init__()

        self.raw_model = raw_model
        self.register_buffer("x_mean", x_mean.float().view(1, -1))
        self.register_buffer("x_std", x_std.float().view(1, -1))
        self.register_buffer("y_mean", y_mean.float().view(1, -1))
        self.register_buffer("y_std", y_std.float().view(1, -1))

    def normalize_x(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.x_mean) / self.x_std

    def unnormalize_x(self, x_normalized: torch.Tensor) -> torch.Tensor:
        return x_normalized * self.x_std + self.x_mean

    def normalize_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.y_mean) / self.y_std

    def unnormalize_y(self, y_normalized: torch.Tensor) -> torch.Tensor:
        return y_normalized * self.y_std + self.y_mean

    def forward(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_raw = self.unnormalize_x(x)
        y_raw = self.raw_model(x_raw, a, u)
        y_normalized = self.normalize_y(y_raw)
        return y_normalized


class NormalizedPropensityMLPWrapper(nn.Module):
    """
    Wrapper around the raw propensity model that exposes normalized x-space.

    It expects normalized x as input, un-normalizes it before calling the raw
    f_A(x), and leaves the probability output unchanged.
    """

    def __init__(
        self,
        raw_model: PropensityMLP,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ):
        super().__init__()

        self.raw_model = raw_model
        self.register_buffer("x_mean", x_mean.float().view(1, -1))
        self.register_buffer("x_std", x_std.float().view(1, -1))

    def unnormalize_x(self, x_normalized: torch.Tensor) -> torch.Tensor:
        return x_normalized * self.x_std + self.x_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = self.unnormalize_x(x)
        return self.raw_model(x_raw)


class OutcomeGenerator:
    """
    Outcome generator built around a sampled PyTorch MLP.
    """

    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation_name: str = "tanh",
        latent_dim: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize the outcome generator.

        Args:
            prior_layers: Distribution for number of layers
            prior_hidden_size: Distribution for hidden size
            prior_weight: Distribution for edge weights
            activation_name: Activation function used inside the PyTorch MLP
            latent_dim: Dimension of U ~ N(0, I)
            device: Device used for the outcome network
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation_name = activation_name
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.outcome_model: Optional[OutcomeMLP] = None
        self.sampled_num_layers = None
        self.sampled_hidden_size = None

    # Sample the architecture 
    def build_outcome_model(self, n_features: int) -> OutcomeMLP:
        self.sampled_num_layers = max(3, self.prior_layers()) # enforce min 3 layers
        self.sampled_hidden_size = self.prior_hidden_size()

        input_size = n_features + 1 + self.latent_dim # dim(x) + dim(a) + dim(u)
        self.outcome_model = OutcomeMLP(
            input_size=input_size,
            hidden_size=self.sampled_hidden_size,
            num_layers=self.sampled_num_layers,
            activation_name=self.activation_name,
        ).to(self.device)

        self.sample_outcome_network_parameters()

        return self.outcome_model

    # Sample the parameters 
    def sample_outcome_network_parameters(self) -> None:
        assert self.outcome_model is not None, "Outcome model must be built before sampling parameters"

        with torch.no_grad():
            for module in self.outcome_model.modules():
                if isinstance(module, nn.Linear):
                    weight = np.array(
                        [
                            [self.prior_weight() for _ in range(module.weight.shape[1])]
                            for _ in range(module.weight.shape[0])
                        ]
                    )
                    bias = np.array([self.prior_weight() for _ in range(module.bias.shape[0])])

                    module.weight.copy_(torch.tensor(weight, dtype=torch.float32, device=self.device))
                    module.bias.copy_(torch.tensor(bias, dtype=torch.float32, device=self.device))

    def sample_latent_confounders(self, num_samples: int) -> np.ndarray:
        return np.random.normal(0.0, 1.0, size=(num_samples, self.latent_dim))

    def generate_outcomes(
        self,
        X: np.ndarray,
        A: np.ndarray,
        U: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape

        # TODO: Replace this generic MLP with an architecture that guarantees
        # invertibility in U for fixed (X, A), as required by the CSA-PFN theory.
        # NOTE: For Subsystem A we intentionally keep a standard differentiable MLP
        # here and treat invertibility in U as an approximation/assumption for now.
        assert self.outcome_model is not None, "Outcome model must be built before generating outcomes"

        if U is None:
            U = self.sample_latent_confounders(n_samples)

        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        a_tensor = torch.tensor(A, dtype=torch.float32, device=self.device)
        u_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y0_tensor = self.outcome_model(
                x_tensor,
                torch.zeros_like(a_tensor),
                u_tensor,
            )
            y1_tensor = self.outcome_model(
                x_tensor,
                torch.ones_like(a_tensor),
                u_tensor,
            )
            y_tensor = torch.where(a_tensor.unsqueeze(1) > 0.5, y1_tensor, y0_tensor)

        Y = y_tensor.cpu().numpy().reshape(-1)
        Y0 = y0_tensor.cpu().numpy().reshape(-1)
        Y1 = y1_tensor.cpu().numpy().reshape(-1)

        return Y, Y0, Y1, U

    def get_model_bundle(self, n_features: int) -> Dict:
        assert self.outcome_model is not None, "Outcome model has not been built yet"

        return {
            "model_class": "OutcomeMLP",
            "input_dim_x": n_features,
            "input_dim_a": 1,
            "latent_dim_u": self.latent_dim,
            "output_dim_y": 1,
            "num_layers": self.sampled_num_layers,
            "hidden_size": self.sampled_hidden_size,
            "activation_name": self.activation_name,
            "state_dict": self.outcome_model.state_dict(),
        }


class TreatmentGenerator:
    """
    Treatment generator built around a sampled deterministic propensity MLP.
    """

    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation_name: str = "tanh",
        device: str = "cpu",
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation_name = activation_name
        self.device = torch.device(device)
        self.propensity_model: Optional[PropensityMLP] = None
        self.sampled_num_layers = None
        self.sampled_hidden_size = None

    def build_propensity_model(self, n_features: int) -> PropensityMLP:
        self.sampled_num_layers = max(3, self.prior_layers())
        self.sampled_hidden_size = self.prior_hidden_size()

        self.propensity_model = PropensityMLP(
            input_size=n_features,
            hidden_size=self.sampled_hidden_size,
            num_layers=self.sampled_num_layers,
            activation_name=self.activation_name,
        ).to(self.device)

        self.sample_propensity_network_parameters()

        return self.propensity_model

    def sample_propensity_network_parameters(self) -> None:
        assert self.propensity_model is not None, "Propensity model must be built before sampling parameters"

        with torch.no_grad():
            for module in self.propensity_model.modules():
                if isinstance(module, nn.Linear):
                    weight = np.array(
                        [
                            [self.prior_weight() for _ in range(module.weight.shape[1])]
                            for _ in range(module.weight.shape[0])
                        ]
                    )
                    bias = np.array([self.prior_weight() for _ in range(module.bias.shape[0])])

                    module.weight.copy_(torch.tensor(weight, dtype=torch.float32, device=self.device))
                    module.bias.copy_(torch.tensor(bias, dtype=torch.float32, device=self.device))

    def generate_treatments(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.propensity_model is not None, "Propensity model must be built before generating treatments"

        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            propensity = self.propensity_model(x_tensor).cpu().numpy().reshape(-1)

        treatment = np.random.binomial(1, propensity).astype(int)
        return treatment, propensity

    def get_model_bundle(self, n_features: int) -> Dict:
        assert self.propensity_model is not None, "Propensity model has not been built yet"

        return {
            "model_class": "PropensityMLP",
            "input_dim_x": n_features,
            "output_dim_a": 1,
            "num_layers": self.sampled_num_layers,
            "hidden_size": self.sampled_hidden_size,
            "activation_name": self.activation_name,
            "state_dict": self.propensity_model.state_dict(),
        }

def save_dataset(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    Y0: Optional[np.ndarray] = None,
    Y1: Optional[np.ndarray] = None,
    U: Optional[np.ndarray] = None,
    propensity: Optional[np.ndarray] = None,
    filename: str = "synthetic_continous_dataset.csv",
    log_fn: Callable[[str], None] = print,
) -> pd.DataFrame:
    n_features = X.shape[1]
    feature_names = [f"x{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df["treatment"] = A
    if propensity is not None:
        df["propensity"] = propensity
    df["outcome"] = Y

    if Y0 is not None:
        df["y0"] = Y0
    if Y1 is not None:
        df["y1"] = Y1

    if Y0 is not None and Y1 is not None:
        df["ite"] = Y1 - Y0

    if U is not None:
        if U.ndim == 1:
            df["u0"] = U
        else:
            for idx in range(U.shape[1]):
                df[f"u{idx}"] = U[:, idx]

    df.to_csv(filename, index=False)
    log_fn(f"Dataset saved to {filename}")

    return df


def save_model_bundle(bundle: Dict, filename: str, log_fn: Callable[[str], None] = print) -> None:
    torch.save(bundle, filename)
    log_fn(f"Model bundle saved to {filename}")


def save_dgp_metadata(metadata: Dict, filename: str, log_fn: Callable[[str], None] = print) -> None:
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    log_fn(f"DGP metadata saved to {filename}")


def generate_single_dataset(
    dataset_id: int,
    num_samples: int = 1024,
    num_features: int = 10,
    output_dir: str = "data/continuous",
    seed_offset: int = 0,
    latent_dim: int = 1,
    device: str = "cpu",
    log_fn: Callable[[str], None] = print,
):
    """
    Generate a single synthetic causal dataset together with separately saved
    outcome and propensity model artifacts for later frontier optimization.

    Args:
        dataset_id: ID for this dataset
        num_samples: Number of samples to generate
        num_features: Number of features to generate
        output_dir: Directory to save all artifacts
        seed_offset: Offset for random seed to ensure different datasets
        latent_dim: Dimension of the latent confounder U ~ N(0, I)
        device: Device used for the outcome model
        log_fn: Function to use for logging messages
    """
    resolved_output_dir = resolve_code_relative_path(output_dir)
    dataset_seed = dataset_id + seed_offset
    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)

    log_fn(f"\n=== Generating Dataset {dataset_id} ===")

    def prior_layers_x():
        return np.random.randint(3, 7)

    def prior_hidden_size_x():
        return np.random.randint(15, 40)

    def prior_weight_x():
        return np.random.normal(0, 1)

    dag_scm = DAGStructuredSCM(
        prior_layers=prior_layers_x,
        prior_hidden_size=prior_hidden_size_x,
        prior_weight=prior_weight_x,
        edge_drop_prob=0.5,
        activation=lambda x: np.tanh(x),
    )

    X = dag_scm.generate_dataset(num_features, num_samples)

    def prior_layers_a():
        return np.random.randint(3, 5)

    def prior_hidden_size_a():
        return np.random.randint(8, 20)

    def prior_weight_a():
        return np.random.normal(0, 0.8)

    treatment_generator = TreatmentGenerator(
        prior_layers=prior_layers_a,
        prior_hidden_size=prior_hidden_size_a,
        prior_weight=prior_weight_a,
        activation_name="tanh",
        device=device,
    )

    treatment_generator.build_propensity_model(num_features)
    A, propensity = treatment_generator.generate_treatments(X)

    def prior_layers_y():
        return np.random.randint(3, 6)

    def prior_hidden_size_y():
        return np.random.randint(10, 25)

    def prior_weight_y():
        return np.random.normal(0, 1.0)

    outcome_generator = OutcomeGenerator(
        prior_layers=prior_layers_y,
        prior_hidden_size=prior_hidden_size_y,
        prior_weight=prior_weight_y,
        activation_name="tanh",
        latent_dim=latent_dim,
        device=device,
    )

    outcome_generator.build_outcome_model(num_features)
    Y, Y0, Y1, U = outcome_generator.generate_outcomes(X, A)

    datasets_dir = os.path.join(resolved_output_dir, "datasets")
    models_dir = os.path.join(resolved_output_dir, "models")
    metadata_dir = os.path.join(resolved_output_dir, "metadata")

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    dataset_filename = os.path.join(datasets_dir, f"synthetic_continous_dataset_{dataset_id}.csv")
    outcome_model_filename = os.path.join(models_dir, f"outcome_bnn_{dataset_id}.pt")
    propensity_model_filename = os.path.join(models_dir, f"propensity_bnn_{dataset_id}.pt")
    metadata_filename = os.path.join(metadata_dir, f"dgp_{dataset_id}.json")

    dataset = save_dataset(
        X,
        A,
        Y,
        Y0,
        Y1,
        U,
        propensity=propensity,
        filename=dataset_filename,
        log_fn=log_fn,
    )

    outcome_model_bundle = outcome_generator.get_model_bundle(num_features)
    save_model_bundle(outcome_model_bundle, outcome_model_filename, log_fn)

    propensity_model_bundle = treatment_generator.get_model_bundle(num_features)
    save_model_bundle(propensity_model_bundle, propensity_model_filename, log_fn)

    metadata = {
        "dataset_id": dataset_id,
        "seed": dataset_seed,
        "num_samples": num_samples,
        "num_features": num_features,
        "latent_dim_u": latent_dim,
        "output_dim_y": 1,
        "dataset_path": dataset_filename,
        "outcome_model_path": outcome_model_filename,
        "propensity_model_path": propensity_model_filename,
        "covariate_generator": {
            "sampled_num_layers": dag_scm.sampled_num_layers,
            "sampled_hidden_size": dag_scm.sampled_hidden_size,
            "edge_drop_prob": dag_scm.edge_drop_prob,
            "num_graph_nodes": dag_scm.dag.number_of_nodes() if dag_scm.dag is not None else None,
            "num_graph_edges": dag_scm.dag.number_of_edges() if dag_scm.dag is not None else None,
        },
        "treatment_generator": {
            "model_class": "PropensityMLP",
            "sampled_num_layers": treatment_generator.sampled_num_layers,
            "sampled_hidden_size": treatment_generator.sampled_hidden_size,
            "activation_name": treatment_generator.activation_name,
            "device": str(treatment_generator.device),
        },
        "outcome_generator": {
            "model_class": "OutcomeMLP",
            "sampled_num_layers": outcome_generator.sampled_num_layers,
            "sampled_hidden_size": outcome_generator.sampled_hidden_size,
            "activation_name": outcome_generator.activation_name,
            "device": str(outcome_generator.device),
        },
    }

    save_dgp_metadata(metadata, metadata_filename, log_fn)

    return {
        "dataset": dataset,
        "X": X,
        "A": A,
        "U": U,
        "Y": Y,
        "Y0": Y0,
        "Y1": Y1,
        "propensity": propensity,
        "dataset_path": dataset_filename,
        "outcome_model_path": outcome_model_filename,
        "propensity_model_path": propensity_model_filename,
        "metadata_path": metadata_filename,
    }


def generate_multiple_datasets(
    num_datasets: int = 10,
    num_samples: int = 1024,
    num_features: int = 10,
    output_dir: str = "data/continuous",
    base_seed: int = 42,
    latent_dim: int = 1,
    device: str = "cpu",
    log_fn: Callable[[str], None] = tqdm.write,
):
    """
    Generate multiple synthetic causal datasets and save the outcome and
    propensity model artifacts needed by Subsystem B alongside each dataset.

    Args:
        num_datasets: Number of datasets to generate
        num_samples: Number of samples per dataset
        num_features: Number of features per dataset
        output_dir: Directory to save dataset/model/metadata artifacts
        base_seed: Base seed for reproducibility
        latent_dim: Dimension of U ~ N(0, I)
        device: Device used for the outcome model
        log_fn: Function for logging messages (default: tqdm.write for progress bar compatibility)
    """
    resolved_output_dir = resolve_code_relative_path(output_dir)
    print(f"Generating {num_datasets} datasets with {num_samples} samples each...")

    os.makedirs(resolved_output_dir, exist_ok=False)

    for i in tqdm(range(1, num_datasets + 1), desc="Generating Datasets"):
        try:
            generate_single_dataset(
                dataset_id=i,
                num_samples=num_samples,
                num_features=num_features,
                output_dir=resolved_output_dir,
                seed_offset=base_seed,
                latent_dim=latent_dim,
                device=device,
                log_fn=log_fn,
            )

        except Exception as exc:
            log_fn(f"Error generating dataset {i}: {str(exc)}")
            continue



def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for synthetic dataset generation.
    
    """
    parser = argparse.ArgumentParser(description="Generate standard synthetic datasets and associated model artifacts.")
    parser.add_argument("--output-dir", type=str, required=True,
        help="Output directory relative to the code root unless an absolute path is provided.")
    parser.add_argument("--num-datasets", type=int, default=10, help="Number of datasets to generate.")
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of samples per dataset.")
    parser.add_argument("--num-features", type=int, default=10, help="Number of covariate features per dataset.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed offset used to derive per-dataset seeds.")
    parser.add_argument("--latent-dim", type=int, default=1, help="Dimension of the latent confounder U.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device used for the outcome model.")
    parser.add_argument("--log-fn", type=callable, default=lambda _: None, help="Function for logging messages.")
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    """Construct generate_multiple_datasets kwargs from parsed CLI arguments."""
    return {
        "num_datasets": args.num_datasets,
        "num_samples": args.num_samples,
        "num_features": args.num_features,
        "output_dir": args.output_dir,
        "base_seed": args.base_seed,
        "latent_dim": args.latent_dim,
        "device": args.device,
        "log_fn": args.log_fn,
    }

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    generate_multiple_datasets(**config)


if __name__ == "__main__":
    main()
