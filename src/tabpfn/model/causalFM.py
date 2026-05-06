# Model for partial identification foundation modeling


from __future__ import annotations

import random
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabpfn.model.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from tabpfn.model.layer import PerFeatureEncoderLayer

DEFAULT_EMSIZE = 128


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)

        return x



class GMMHead(nn.Module):

    def __init__(self, z_dim: int, n_components: int, min_sigma: float = 1e-3, pi_temp: float = 1.0):
        super().__init__()
        self.K = n_components
        self.min_sigma = min_sigma
        self.pi_temp = pi_temp
        self.fc_pi = nn.Linear(z_dim, n_components) 
        self.fc_mu = nn.Linear(z_dim, n_components) 
        self.fc_sigma = nn.Linear(z_dim, n_components) 

    def forward(self, z: torch.Tensor):

        logits = self.fc_pi(z) / self.pi_temp # temperature scaling
        pi = F.softmax(logits, dim=-1) # mixture weight 
        mu = self.fc_mu(z)
        sigma = F.softplus(self.fc_sigma(z)) + self.min_sigma
        return pi, mu, sigma


class PerFeatureTransformerPIFM(nn.Module):

    def __init__(
        self,
        *,
        gmm_n_components: int = 5,
        gmm_min_sigma: float = 1e-3,
        gmm_pi_temp: float = 1.0,
        x_encoder: nn.Module | None = None,   # For covariates X
        a_encoder: nn.Module | None = None,   # For treatment A
        y_encoder: nn.Module | None = None,   # For factual outcome Y
        gamma_encoder: nn.Module | None = None,  # For sensitivity parameter Gamma
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
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
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        **layer_kwargs: Any,
    ):
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


        if gamma_encoder is None:
            gamma_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # Gamma scalar + nan indicator
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )

        self.x_encoder = x_encoder      # Covariate encoder
        self.a_encoder = a_encoder      # Treatment encoder
        self.y_encoder = y_encoder      # Outcome encoder
        self.gamma_encoder = gamma_encoder  # Sensitivity parameter encoder
        
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation

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

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.seed = seed if seed is not None else random.randint(0, 1_000_000)


        self.gmm_head_upper = GMMHead(
            z_dim=self.ninp,
            n_components=gmm_n_components,
            min_sigma=gmm_min_sigma,
            pi_temp=gmm_pi_temp,
        )
        self.gmm_head_lower = GMMHead(
            z_dim=self.ninp,
            n_components=gmm_n_components,
            min_sigma=gmm_min_sigma,
            pi_temp=gmm_pi_temp,
        )

    def _pack_pifm_io(
        self,
        X_context: torch.Tensor,
        A_context: torch.Tensor,
        Y_context: torch.Tensor,
        X_query: torch.Tensor,
        A_query: torch.Tensor,
        Gamma_query: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Pack context and query arrays into the (seq_len, 1, ...) format
        expected by forward().

        Input shapes:
            X_context:   (n, d_x)
            A_context:   (n, 1)
            Y_context:   (n, 1)
            X_query:     (m, d_x)
            A_query:     (m, 1)
            Gamma_query: (m, 1)

        Returns:
            x, a, y, gamma - each a dict with key "main" and shape (n+m, 1, ...)
            single_eval_pos = n
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = torch.float32

        n = X_context.shape[0]
        m = X_query.shape[0]

        def _to(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=dtype)

        X_context = _to(X_context)
        A_context = _to(A_context)
        Y_context = _to(Y_context)
        X_query = _to(X_query)
        A_query = _to(A_query)
        Gamma_query = _to(Gamma_query)

        def nan_col(rows: int, cols: int) -> torch.Tensor:
            return torch.full((rows, cols), torch.nan, device=device, dtype=dtype)

        x_full = torch.cat([X_context, X_query], dim=0)
        a_full = torch.cat([A_context, A_query], dim=0)
        y_full = torch.cat([Y_context, nan_col(m, 1)], dim=0)
        gamma_full = torch.cat([nan_col(n, 1), Gamma_query], dim=0)

        x = {"main": x_full.unsqueeze(1)}
        a = {"main": a_full.unsqueeze(1)}
        y = {"main": y_full.unsqueeze(1)}
        gamma = {"main": gamma_full.unsqueeze(1)}

        return x, a, y, gamma, n

    @torch.no_grad()
    def estimate_capo_bounds(
        self,
        X_context: torch.Tensor,
        A_context: torch.Tensor,
        Y_context: torch.Tensor,
        X_query: torch.Tensor,
        A_query: torch.Tensor,
        Gamma_query: torch.Tensor,
        **forward_kwargs,
    ) -> dict:
        """
        Estimate CAPO bound PPDs for a set of query points.

        Args:
            X_context:   (n, d_x)   observational covariates
            A_context:   (n, 1)     observed treatments
            Y_context:   (n, 1)     observed outcomes
            X_query:     (m, d_x)   query covariates
            A_query:     (m, 1)     query treatment arms
            Gamma_query: (m, 1)     sensitivity parameters

        Returns dict with keys:
            upper_mean:  (m,)  GMM mean of upper bound PPD per query
            lower_mean:  (m,)  GMM mean of lower bound PPD per query
            upper_pi, upper_mu, upper_sigma:  (m, K)  full GMM params, upper head
            lower_pi, lower_mu, lower_sigma:  (m, K)  full GMM params, lower head
        """
        x, a, y, gamma, n = self._pack_pifm_io(
            X_context,
            A_context,
            Y_context,
            X_query,
            A_query,
            Gamma_query,
        )
        out = self.forward(x, a, y, gamma, single_eval_pos=n, **forward_kwargs)

        def _q(key: str) -> torch.Tensor:
            return out[key][0, n:, :]

        upper_pi = _q("gmm_upper_pi")
        upper_mu = _q("gmm_upper_mu")
        upper_sigma = _q("gmm_upper_sigma")
        lower_pi = _q("gmm_lower_pi")
        lower_mu = _q("gmm_lower_mu")
        lower_sigma = _q("gmm_lower_sigma")

        return {
            "upper_mean": (upper_pi * upper_mu).sum(dim=-1),
            "lower_mean": (lower_pi * lower_mu).sum(dim=-1),
            "upper_pi": upper_pi,
            "upper_mu": upper_mu,
            "upper_sigma": upper_sigma,
            "lower_pi": lower_pi,
            "lower_mu": lower_mu,
            "lower_sigma": lower_sigma,
        }

    def forward(
        self,
        x: torch.Tensor | dict,
        a: torch.Tensor | dict | None = None,
        y: torch.Tensor | dict | None = None,
        gamma: torch.Tensor | dict | None = None,
        single_eval_pos: int | None = None,
        *,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """Core forward pass for partial identification bound estimation.
        
        Args:
            x: Covariates. Shape: (seq_len, batch_size, num_features)
            a: Treatment assignment. Shape: (seq_len, batch_size, 1)
            y: Factual outcomes. Shape: (seq_len, batch_size, 1)
            gamma: Sensitivity parameter. Shape: (seq_len, batch_size, 1)
            single_eval_pos: Position to evaluate at
        Returns:
            Dictionary containing upper/lower bound GMM parameters.
        """

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
            # Avoid mutating caller-owned tensors during internal rearranges.
            x = {k: v for k, v in x.items()}
        else:
            x = {"main": x}
        seq_len, batch_size, num_x_features = x["main"].shape

        if a is None:
            a = {"main": torch.zeros(0, batch_size, 1, device=x["main"].device, dtype=x["main"].dtype)}
        elif isinstance(a, dict):
            assert "main" in set(a.keys()), f"Main must be in input keys: {a.keys()}."
            a = {k: v for k, v in a.items()}
        else:
            a = {"main": a}
        _, _, num_a_features = a["main"].shape

        if y is None:
            y = {"main": torch.zeros(0, batch_size, 1, device=x["main"].device, dtype=x["main"].dtype)}
        elif isinstance(y, dict):
            assert "main" in set(y.keys())
            y = {k: v for k, v in y.items()}
        else:
            y = {"main": y}

        if gamma is None:
            gamma = {
                "main": torch.full(
                    (seq_len, batch_size, 1),
                    torch.nan,
                    device=x["main"].device,
                    dtype=x["main"].dtype,
                )
            }
        elif isinstance(gamma, dict):
            assert "main" in gamma.keys(), (
                f"gamma dict must contain 'main' key, got: {gamma.keys()}"
            )
            gamma = {k: v for k, v in gamma.items()}
        else:
            gamma = {"main": gamma}

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

        # Prevent outcome leakage at query positions - y is never observed there.
        # NOTE: a is NOT zeroed here. For CAPO, a_query is an explicit input
        # specifying which treatment arm to bound. Masking it would be wrong.
        y["main"][single_eval_pos:] = torch.nan

        # Encode factual outcome Y
        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y

        assert not torch.isnan(embedded_y).any(), (
            "NaN in encoded y. Make sure to add nan handlers to the ys that are "
            "not fully provided (test set missing)."
        )
        
        # Encode treatment A  

        embedded_a = self.a_encoder(
            a,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del a
        assert not torch.isnan(embedded_a).any(), (
            "NaN in encoded a. Make sure to add nan handlers to the as that are "
            "not fully provided (test set missing)."
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

        # Encode sensitivity parameter Gamma.
        # Context rows have NaN Gamma (no sensitivity parameter); query rows have Gamma*.
        # NanHandlingEncoderStep converts NaN -> (0, indicator=1) in both cases.
        embedded_gamma = self.gamma_encoder(
            gamma,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)
        del gamma

        assert not torch.isnan(embedded_gamma).any(), (
            "NaN in encoded gamma. Check that NanHandlingEncoderStep is present "
            "in gamma_encoder."
        )

        embedded_gamma_grouped = embedded_gamma.unsqueeze(2)  # (b, s, 1, e)

        embedded_input = torch.cat(
            (embedded_x, embedded_a_grouped, embedded_y_grouped, embedded_gamma_grouped),
            dim=2,
        )  # (b, s, f_x + 3, e)

        assert not torch.isnan(embedded_input).any(), (
            "There should be no NaNs in the encoded x, a, y, and gamma."
        )

        # Apply positional embeddings
        embedded_input = self.add_embeddings_pifm(
            embedded_input,
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


        capo_representation = encoder_out.mean(dim=2)

        B, S, E = capo_representation.shape
        z_rep = capo_representation.reshape(B * S, E)

        pi_u, mu_u, sigma_u = self.gmm_head_upper(z_rep)
        pi_l, mu_l, sigma_l = self.gmm_head_lower(z_rep)

        K = pi_u.shape[-1]
        return {
            "gmm_upper_pi": pi_u.reshape(B, S, K),
            "gmm_upper_mu": mu_u.reshape(B, S, K),
            "gmm_upper_sigma": sigma_u.reshape(B, S, K),
            "gmm_lower_pi": pi_l.reshape(B, S, K),
            "gmm_lower_mu": mu_l.reshape(B, S, K),
            "gmm_lower_sigma": sigma_l.reshape(B, S, K),
        }


    def add_embeddings_pifm(
        self,
        embedded_input: torch.Tensor,
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
