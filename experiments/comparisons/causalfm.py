"""CausalFM CATE architecture, checkpoint loading and matched PI-FM comparison.

Architecture from yccm/CausalFM-toolkit, commit bb74ef7 (Apache-2.0).
"""

from __future__ import annotations

from experiments.shared.paths import CHECKPOINTS
import torch
import numpy as np
import pandas as pd
from experiments.shared.evaluation import gmm_arrays
from typing import Any, Literal
from collections.abc import Iterable
import networkx as nx
import random
import einops
from torch import nn
from tabpfn.model.encoders import LinearInputEncoderStep, NanHandlingEncoderStep, SequentialEncoder
from tabpfn.model.layer import PerFeatureEncoderLayer
from tabpfn.model.causalFM import DEFAULT_EMSIZE, GMMHead, LayerStack, isolate_torch_rng
from experiments.shared.runtime import sha


UPSTREAM_REPOSITORY = "https://github.com/yccm/CausalFM-toolkit"

UPSTREAM_COMMIT = "bb74ef729c70d274fe2bb422c3b0edff4754fa37"

UPSTREAM_LICENSE = "Apache-2.0"

CHECKPOINT_SHA256 = "9b75ebf02016a61b3a6e8fc075f52f9c56189ea5d7e7633137a5c9b70f25774e"

CHECKPOINT_PATH = (
    CHECKPOINTS / "causalfm" / UPSTREAM_COMMIT / "best_model.pth"
)

class PerFeatureTransformerCATE(nn.Module):

    def __init__(
        self,
        *,
        use_gmm_head: bool = True,
        gmm_n_components: int =5,
        gmm_min_sigma: float = 1e-3,
        gmm_pi_temp: float = 1.0,
        x_encoder: nn.Module | None = None,  # For covariates X
        a_encoder: nn.Module | None = None,  # For treatment A
        y_encoder: nn.Module | None = None,  # For factual outcome Y
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        decoder_dict: dict[str, tuple[type[nn.Module] | None, int]] | None = None,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: int | None = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        **layer_kwargs: Any,
    ):


        decoder_dict = {"cate": (None, 1)}


        super().__init__()

        # Set up encoders for X, A, Y
        if x_encoder is None:
            x_encoder = SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=1,
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            )
        if a_encoder is None:
            a_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # A + nan indicators
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )

        if y_encoder is None:
            y_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # Y + nan indicators
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )


        self.x_encoder = x_encoder  # Covariate encoder
        self.a_encoder = a_encoder  # Treatment encoder
        self.y_encoder = y_encoder  # Outcome encoder

        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            **layer_kwargs,
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            assert nlayers_decoder is not None
            self.transformer_decoder = LayerStack(
                layer_creator=layer_creator,
                num_layers=nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert use_separate_decoder
            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression,
                ninp,
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator=layer_creator,
                num_layers=2,
            )

        initialized_decoder_dict = {}
        for decoder_key in decoder_dict:
            decoder_model, decoder_n_out = decoder_dict[decoder_key]
            if decoder_model is None:
                initialized_decoder_dict[decoder_key] = nn.Sequential(
                    nn.Linear(ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, decoder_n_out),
                )
            else:
                initialized_decoder_dict[decoder_key] = decoder_model(
                    ninp,
                    nhid,
                    decoder_n_out,
                )
        self.decoder_dict = nn.ModuleDict(initialized_decoder_dict)

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)


        self.use_gmm_head = use_gmm_head
        if self.use_gmm_head:
            self.gmm_head = GMMHead(
                z_dim = self.ninp,
                n_components = gmm_n_components,
                min_sigma = gmm_min_sigma,
                pi_temp = gmm_pi_temp,
            )

    def _pack_eval_io(
        self,
        X_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Expected input shapes:
            X_train: [N_tr, Dx],  A_train: [N_tr, 1],  Y_train: [N_tr, 1],  X_test: [N_te, Dx]
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = torch.float32

        X_train = X_train.to(device=device, dtype=dtype)
        A_train = A_train.to(device=device, dtype=dtype)
        Y_train = Y_train.to(device=device, dtype=dtype)
        X_test  = X_test.to(device=device,  dtype=dtype)

        n_tr = X_train.shape[0]
        n_te = X_test.shape[0]
        single_eval_pos = int(n_tr)

        # concat along seq (support + query)
        X_all = torch.cat([X_train, X_test], dim=0)
        A_all = torch.cat([A_train, torch.full((n_te, A_train.shape[1]), torch.nan, device=device, dtype=dtype)], dim=0)
        Y_all = torch.cat([Y_train, torch.full((n_te, Y_train.shape[1]), torch.nan, device=device, dtype=dtype)], dim=0)

        # add batch dim = 1 -> [S, 1, D]
        X_all = X_all.unsqueeze(1)
        A_all = A_all.unsqueeze(1)
        Y_all = Y_all.unsqueeze(1)

        x = {"main": X_all}
        a = {"main": A_all}
        y = {"main": Y_all}
        return x, a, y, single_eval_pos

    @torch.no_grad()
    def estimate_cate(
        self,
        X_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        *,
        return_gmm: bool = False,
        **forward_kwargs,
    ):
        """
            inputs: X_train, A_train, Y_train, X_test
            returns: dict with predicted CATE on X_test (and optional GMM params)
        """
        x, a, y, single_eval_pos = self._pack_eval_io(X_train, A_train, Y_train, X_test)
        out = self.forward(x, a, y, single_eval_pos=single_eval_pos, **forward_kwargs)

        cate = out["cate_mean"][:, single_eval_pos:, :]
        cate = cate.squeeze()
        result = {"cate": cate}

        if return_gmm and ("gmm_pi" in out):
            result.update({
                "gmm_pi":    out["gmm_pi"][0, single_eval_pos:, :],
                "gmm_mu":    out["gmm_mu"][0, single_eval_pos:, :],
                "gmm_sigma": out["gmm_sigma"][0, single_eval_pos:, :],
            })
        return result

    def forward(
        self,
        x: torch.Tensor | dict,
        a: torch.Tensor | dict | None = None,
        y: torch.Tensor | dict | None = None,
        single_eval_pos: int | None = None,
        *,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """Core forward pass for CATE estimation

        Args:
            x: Covariates. Shape: (seq_len, batch_size, num_features) # seq_len = n_train_rows + n_test_rows
            a: Treatment assignment. Shape: (seq_len, batch_size, 1)
            y: Factual outcomes. Shape: (seq_len, batch_size, 1)
            single_eval_pos: Position to evaluate at
        Returns:
            Dictionary containing CATE distribution predictions and optionally other outputs
        """

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_x_features = x["main"].shape # seq_len = n_train_rows + n_test_rows

        if isinstance(a, dict):
            assert "main" in set(a.keys()), f"Main must be in input keys: {a.keys()}."
        else:
            a = {"main": a}
        _, _, num_a_features = a["main"].shape

        if isinstance(y, dict):
            assert "main" in set(y.keys())
        else:
            y = {"main": y}
        _, _, num_y_features = y["main"].shape

        training_targets_provided = y["main"] is not None and y["main"].shape[0] > 0

        # Handle y (can be None during inference)
        if y is None:
            y = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )
        # Handle a (can be None during inference)
        if a is None:
            a = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )


        # The model will make predictions from the single_eval_pos'th row onwards.

        single_eval_pos_ = int(single_eval_pos)

        # Pad features to multiple of features_per_group
        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos:] = torch.nan
        a["main"][single_eval_pos:] = torch.nan

        # Encode factual outcome Y
        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y

        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

                # Encode treatment A

        embedded_a = self.a_encoder(
            a,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del a
        if torch.isnan(embedded_a).any():
            raise ValueError(
                f"{torch.isnan(embedded_a).any()=}, make sure to add nan handlers"
                " to the as that are not fully provided (test set missing)",
            )

        # Encode covariates X
        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.x_encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,

            ),
            "s (b f) e -> b s f e",
            b=batch_size,
        )
        del x


        embedded_a_grouped = embedded_a.unsqueeze(2)  # (b, s, e) -> (b, s, 1, e)
        embedded_y_grouped = embedded_y.unsqueeze(2)  # (b, s, e) -> (b, s, 1, e)


        embedded_input = torch.cat(
            (embedded_x, embedded_a_grouped, embedded_y_grouped), dim=2
        )  # (b, s, f_x + 1 + 1, e)

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x, a, and y."
                f"Your embedded inputs returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_a).any()=} | {torch.isnan(embedded_y).any()=}",
            )

        # Apply positional embeddings
        embedded_input = self.add_embeddings_cate(
            embedded_input,
            data_dags=data_dags,
            num_features=num_x_features + num_a_features,
            seq_len=seq_len,
        )

        # Pass through transformer
        encoder_out = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos_,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )


        if self.transformer_decoder:
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)


        cate_representation = encoder_out.mean(dim=2)

        gmm_out = None
        if self.use_gmm_head:
            B, S, E = cate_representation.shape
            z_rep = cate_representation.reshape(B * S, E)
            pi, mu, sigma = self.gmm_head(z_rep)
            K = pi.shape[-1]
            pi = pi.reshape(B, S, K)
            mu = mu.reshape(B, S, K)
            sigma = sigma.reshape(B, S, K)
            gmm_out = {"gmm_pi": pi, "gmm_mu": mu, "gmm_sigma": sigma}




        out = {}

        if gmm_out is not None:
            out.update(gmm_out)  # add gmm params if enabled
            out["cate_mean"] = (pi * mu).sum(dim=-1, keepdim=True)
        return out


    def add_embeddings_cate(
        self,
        embedded_input: torch.Tensor,
        *,
        data_dags: Iterable[nx.DiGraph] | None,
        num_features: int,
        seq_len: int,
    ) -> torch.Tensor:
        """embedding addition for causality"""

        with isolate_torch_rng(self.seed, device=embedded_input.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3]),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (embedded_input.shape[2], embedded_input.shape[3]),
                        device=embedded_input.device,
                        dtype=embedded_input.dtype,
                    )
                    * 2
                    - 1
                )
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (embedded_input.shape[2],),
                    )
                ]
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3] // 4),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                embedded_input += embs[None, None]

        return embedded_input



    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers."""
        for layer in self.transformer_encoder.layers:
            assert hasattr(layer, "save_peak_mem_factor")
            layer.save_peak_mem_factor = factor

    def empty_trainset_representation_cache(self) -> None:
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()

def checkpoint_provenance() -> dict:
    assert CHECKPOINT_PATH.is_file(), f"Missing CausalFM checkpoint: {CHECKPOINT_PATH}"
    assert sha(CHECKPOINT_PATH) == CHECKPOINT_SHA256, 'Wrong CausalFM checkpoint'
    return dict(path=str(CHECKPOINT_PATH), sha256=CHECKPOINT_SHA256,
                upstream_repository=UPSTREAM_REPOSITORY, upstream_commit=UPSTREAM_COMMIT,
                license=UPSTREAM_LICENSE)

def load_causalfm(device: torch.device):
    provenance = checkpoint_provenance()
    model_class = PerFeatureTransformerCATE
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(checkpoint, dict)
    assert set(checkpoint) == {
        "epoch",
        "epoch_times",
        "model_state_dict",
        "optimizer_state_dict",
        "training_time",
        "val_loss",
    }
    state_dict = checkpoint["model_state_dict"]
    assert isinstance(state_dict, dict) and len(state_dict) == 76
    model = model_class()
    result = model.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance["checkpoint_metadata"] = {
        "epoch": int(checkpoint["epoch"]),
        "val_loss": float(checkpoint["val_loss"]),
        "state_dict_keys": len(state_dict),
        "load": "torch.load(weights_only=True); load_state_dict(strict=True)",
    }
    return model, provenance

def assert_gmm_output(output: dict, query_count: int) -> None:
    assert set(output) == {"cate", "gmm_pi", "gmm_mu", "gmm_sigma"}
    assert output["cate"].shape == (query_count,)
    assert output["gmm_pi"].shape == (query_count, 5)
    assert output["gmm_mu"].shape == (query_count, 5)
    assert output["gmm_sigma"].shape == (query_count, 5)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.all(output["gmm_pi"] >= 0)
    assert torch.allclose(
        output["gmm_pi"].sum(dim=-1),
        torch.ones(query_count, device=output["gmm_pi"].device),
        atol=2e-6,
        rtol=0,
    )
    assert torch.all(output["gmm_sigma"] >= 1e-3)
    reconstructed = (output["gmm_pi"] * output["gmm_mu"]).sum(dim=-1)
    assert torch.allclose(output["cate"], reconstructed, atol=2e-6, rtol=0)

SEED = 20260727

X_COLUMNS = [f"x_{index}" for index in range(10)]


def paired_e4_instances(collapse: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_id",
        "source_row_index",
        "base_query_index",
        "a_query",
        "q0",
        "theta_lower",
        "theta_upper",
        "upper_mean",
        "lower_mean",
        *X_COLUMNS,
    }
    assert required <= set(collapse.columns)
    records: list[dict] = []
    for dataset_id, dgp in collapse.groupby("dataset_id", sort=False):
        assert len(dgp) == 16
        dgp = dgp.sort_values("base_query_index")
        for query_position, (_, pair) in enumerate(
            dgp.groupby("source_row_index", sort=False)
        ):
            pair = pair.sort_values("a_query")
            assert pair["a_query"].tolist() == [0, 1]
            assert np.allclose(
                pair.iloc[0][X_COLUMNS].to_numpy(dtype=float),
                pair.iloc[1][X_COLUMNS].to_numpy(dtype=float),
                atol=2e-7,
                rtol=0,
            )
            assert np.allclose(pair["theta_lower"], pair["theta_upper"], atol=1e-6, rtol=1e-6)
            arm0, arm1 = pair.iloc[0], pair.iloc[1]
            pifm_lower = float(arm1["lower_mean"] - arm0["upper_mean"])
            pifm_upper = float(arm1["upper_mean"] - arm0["lower_mean"])
            record = {
                "dataset_id": int(dataset_id),
                "query_position": query_position,
                "source_row_index": int(arm0["source_row_index"]),
                "oracle_cate": float(arm1["q0"] - arm0["q0"]),
                "pifm_cate_lower": pifm_lower,
                "pifm_cate_upper": pifm_upper,
                "pifm_cate": 0.5 * (pifm_lower + pifm_upper),
                "pifm_cate_collapse_gap": pifm_upper - pifm_lower,
            }
            record.update({column: float(arm0[column]) for column in X_COLUMNS})
            records.append(record)
    result = pd.DataFrame(records)
    assert len(result) == collapse["dataset_id"].nunique() * 8
    assert np.isfinite(result.select_dtypes(include=[np.number])).all().all()
    return result

def sample_gmm(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assert pi.shape == mu.shape == sigma.shape
    uniforms = rng.random((len(pi), samples))
    cumulative = np.cumsum(pi, axis=1)
    components = (uniforms[:, :, None] > cumulative[:, None, :]).sum(axis=2)
    components = np.minimum(components, pi.shape[1] - 1)
    selected_mu = np.take_along_axis(mu, components, axis=1)
    selected_sigma = np.take_along_axis(sigma, components, axis=1)
    return selected_mu + selected_sigma * rng.standard_normal((len(pi), samples))

def attach_pifm_predictive_intervals(
    instances: pd.DataFrame,
    collapse: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    output = instances.copy()
    for column in (
        "pifm_lower_q05",
        "pifm_lower_q95",
        "pifm_upper_q05",
        "pifm_upper_q95",
    ):
        output[column] = np.nan

    for dataset_id, dgp in collapse.groupby("dataset_id", sort=False):
        dgp = dgp.sort_values("base_query_index")
        arm0 = dgp.loc[dgp["a_query"].eq(0)].sort_values("base_query_index")
        arm1 = dgp.loc[dgp["a_query"].eq(1)].sort_values("base_query_index")
        assert len(arm0) == len(arm1) == 8
        rng = np.random.default_rng(seed + int(dataset_id))
        lower1 = sample_gmm(*gmm_arrays(arm1, "lower"), samples, rng)
        upper0 = sample_gmm(*gmm_arrays(arm0, "upper"), samples, rng)
        upper1 = sample_gmm(*gmm_arrays(arm1, "upper"), samples, rng)
        lower0 = sample_gmm(*gmm_arrays(arm0, "lower"), samples, rng)
        lower_cate = lower1 - upper0
        upper_cate = upper1 - lower0
        local_index = output.index[output["dataset_id"].eq(dataset_id)]
        assert len(local_index) == 8
        output.loc[local_index, ["pifm_lower_q05", "pifm_lower_q95"]] = np.quantile(
            lower_cate, [0.05, 0.95], axis=1
        ).T
        output.loc[local_index, ["pifm_upper_q05", "pifm_upper_q95"]] = np.quantile(
            upper_cate, [0.05, 0.95], axis=1
        ).T
    assert output.filter(regex=r"_q(05|95)$").notna().all().all()
    return output


def point_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    truth = predictions["oracle_cate"].to_numpy()
    for method, column in (
        ("causalfm", "causalfm_cate"),
        ("pifm_midpoint", "pifm_cate"),
    ):
        estimate = predictions[column].to_numpy()
        error = estimate - truth
        records.append(
            {
                "method": method,
                "n_dgps": predictions["dataset_id"].nunique(),
                "n_queries": len(predictions),
                "bias": error.mean(),
                "mae": np.abs(error).mean(),
                "rmse": np.sqrt(np.square(error).mean()),
                "pehe": np.sqrt(np.square(error).mean()),
                "correlation": np.corrcoef(estimate, truth)[0, 1],
            }
        )
    return pd.DataFrame(records)

def predictive_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    truth = predictions["oracle_cate"]
    records: list[dict] = []
    for method, low, high in (
        ("causalfm_direct", "causalfm_q05", "causalfm_q95"),
        ("pifm_lower_derived", "pifm_lower_q05", "pifm_lower_q95"),
        ("pifm_upper_derived", "pifm_upper_q05", "pifm_upper_q95"),
    ):
        covered = (truth >= predictions[low]) & (truth <= predictions[high])
        width = predictions[high] - predictions[low]
        assert (width >= 0).all()
        records.append(
            {
                "method": method,
                "n_dgps": predictions["dataset_id"].nunique(),
                "n_queries": len(predictions),
                "coverage_90": covered.mean(),
                "mean_width_90": width.mean(),
                "median_width_90": width.median(),
            }
        )
    return pd.DataFrame(records)

def paired_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions[
        ["dataset_id", "query_position", "source_row_index", "oracle_cate"]
    ].copy()
    output["causalfm_error"] = predictions["causalfm_cate"] - predictions["oracle_cate"]
    output["pifm_error"] = predictions["pifm_cate"] - predictions["oracle_cate"]
    output["absolute_error_difference_causalfm_minus_pifm"] = (
        output["causalfm_error"].abs() - output["pifm_error"].abs()
    )
    output["squared_error_difference_causalfm_minus_pifm"] = output[
        "causalfm_error"
    ].pow(2) - output["pifm_error"].pow(2)
    output["causalfm_lower_absolute_error"] = (
        output["causalfm_error"].abs() < output["pifm_error"].abs()
    )
    return output

def ci_record(
    family: str,
    method: str,
    metric: str,
    point: float,
    values: np.ndarray,
    repetitions: int,
) -> dict:
    assert len(values) == repetitions and np.isfinite(values).all()
    return {
        "family": family,
        "method": method,
        "metric": metric,
        "point": point,
        "lower_95": np.quantile(values, 0.025),
        "median": np.quantile(values, 0.5),
        "upper_95": np.quantile(values, 0.975),
        "repetitions": repetitions,
    }

def clustered_bootstrap(
    predictions: pd.DataFrame,
    point: pd.DataFrame,
    predictive: pd.DataFrame,
    seed: int,
    repetitions: int,
) -> pd.DataFrame:
    dataset_ids = predictions["dataset_id"].drop_duplicates().to_numpy()
    n_clusters = len(dataset_ids)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(repetitions, n_clusters))
    records: list[dict] = []
    truth_column = "oracle_cate"

    for method, prediction_column in (
        ("causalfm", "causalfm_cate"),
        ("pifm_midpoint", "pifm_cate"),
    ):
        rows: list[dict] = []
        for dataset_id in dataset_ids:
            group = predictions.loc[predictions["dataset_id"].eq(dataset_id)]
            truth = group[truth_column].to_numpy()
            estimate = group[prediction_column].to_numpy()
            error = estimate - truth
            rows.append(
                {
                    "n": len(group),
                    "sum_truth": truth.sum(),
                    "sum_estimate": estimate.sum(),
                    "sum_truth2": np.square(truth).sum(),
                    "sum_estimate2": np.square(estimate).sum(),
                    "sum_cross": (truth * estimate).sum(),
                    "sum_error": error.sum(),
                    "sum_sq_error": np.square(error).sum(),
                    "sum_abs_error": np.abs(error).sum(),
                }
            )
        aggregate = pd.DataFrame(rows)
        sampled = {
            column: aggregate[column].to_numpy()[draws].sum(axis=1)
            for column in aggregate
        }
        n = sampled["n"]
        covariance = sampled["sum_cross"] - (
            sampled["sum_truth"] * sampled["sum_estimate"] / n
        )
        truth_ss = sampled["sum_truth2"] - np.square(sampled["sum_truth"]) / n
        estimate_ss = sampled["sum_estimate2"] - np.square(sampled["sum_estimate"]) / n
        values = {
            "bias": sampled["sum_error"] / n,
            "mae": sampled["sum_abs_error"] / n,
            "rmse": np.sqrt(sampled["sum_sq_error"] / n),
            "pehe": np.sqrt(sampled["sum_sq_error"] / n),
            "correlation": covariance / np.sqrt(truth_ss * estimate_ss),
        }
        indexed = point.set_index("method").loc[method]
        for metric, bootstrap_values in values.items():
            records.append(
                ci_record(
                    "point_accuracy",
                    method,
                    metric,
                    float(indexed[metric]),
                    bootstrap_values,
                    repetitions,
                )
            )

    for method, low, high in (
        ("causalfm_direct", "causalfm_q05", "causalfm_q95"),
        ("pifm_lower_derived", "pifm_lower_q05", "pifm_lower_q95"),
        ("pifm_upper_derived", "pifm_upper_q05", "pifm_upper_q95"),
    ):
        rows = []
        for dataset_id in dataset_ids:
            group = predictions.loc[predictions["dataset_id"].eq(dataset_id)]
            covered = (group[truth_column] >= group[low]) & (
                group[truth_column] <= group[high]
            )
            width = group[high] - group[low]
            rows.append(
                {
                    "n": len(group),
                    "covered": covered.sum(),
                    "width": width.sum(),
                }
            )
        aggregate = pd.DataFrame(rows)
        n = aggregate["n"].to_numpy()[draws].sum(axis=1)
        coverage = aggregate["covered"].to_numpy()[draws].sum(axis=1) / n
        width = aggregate["width"].to_numpy()[draws].sum(axis=1) / n
        indexed = predictive.set_index("method").loc[method]
        records.extend(
            [
                ci_record(
                    "predictive",
                    method,
                    "coverage_90",
                    float(indexed["coverage_90"]),
                    coverage,
                    repetitions,
                ),
                ci_record(
                    "predictive",
                    method,
                    "mean_width_90",
                    float(indexed["mean_width_90"]),
                    width,
                    repetitions,
                ),
            ]
        )

    paired = paired_errors(predictions)
    rows = []
    for dataset_id in dataset_ids:
        group = paired.loc[paired["dataset_id"].eq(dataset_id)]
        rows.append(
            {
                "n": len(group),
                "absolute_difference": group[
                    "absolute_error_difference_causalfm_minus_pifm"
                ].sum(),
                "squared_difference": group[
                    "squared_error_difference_causalfm_minus_pifm"
                ].sum(),
                "causalfm_better": group["causalfm_lower_absolute_error"].sum(),
            }
        )
    aggregate = pd.DataFrame(rows)
    n = aggregate["n"].to_numpy()[draws].sum(axis=1)
    pair_values = {
        "mean_absolute_error_difference": (
            aggregate["absolute_difference"].to_numpy()[draws].sum(axis=1) / n
        ),
        "mean_squared_error_difference": (
            aggregate["squared_difference"].to_numpy()[draws].sum(axis=1) / n
        ),
        "fraction_causalfm_lower_absolute_error": (
            aggregate["causalfm_better"].to_numpy()[draws].sum(axis=1) / n
        ),
    }
    point_values = {
        "mean_absolute_error_difference": paired[
            "absolute_error_difference_causalfm_minus_pifm"
        ].mean(),
        "mean_squared_error_difference": paired[
            "squared_error_difference_causalfm_minus_pifm"
        ].mean(),
        "fraction_causalfm_lower_absolute_error": paired[
            "causalfm_lower_absolute_error"
        ].mean(),
    }
    for metric, values in pair_values.items():
        records.append(
            ci_record(
                "paired",
                "causalfm_minus_pifm",
                metric,
                float(point_values[metric]),
                values,
                repetitions,
            )
        )
    return pd.DataFrame(records)
