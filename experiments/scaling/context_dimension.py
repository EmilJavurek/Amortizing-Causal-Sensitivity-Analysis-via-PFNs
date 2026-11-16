"""Inference-time context-length and covariate-dimension sweeps with observed-context scaling."""


from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.shared.evaluation import MSM_CHECKPOINT, add_gmm_diagnostics, gmm_arrays, load_pifm, make_query_grid, normalized_dataset, pifm_predict_frame, propensity_stratified_row_indices
from experiments.shared.observed import observed_run
from experiments.shared.runtime import fresh, read, write
from experiments.shared.inputs import cohort_ids, input_path, normalized_root, cohort_artifacts
from experiments.shared.paths import OBSERVED_DATA
from setting_standard.causaldataset import build_outcome_model_from_bundle, build_propensity_model_from_bundle, discover_causal_artifacts, discover_raw_causal_artifacts, load_outcome_model_bundle, load_propensity_model_bundle
from setting_standard.gen_standard_syn import DAGStructuredSCM
from setting_standard.normalize_data import normalize_artifact_dataset
from typing import Iterable


# Protocol


GAMMA_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

TRAINED_CONTEXT_LENGTH = 1024

def _covariate_scm() -> DAGStructuredSCM:
    """Construct the exact covariate prior used by ``generate_single_dataset``."""

    def prior_layers_x():
        return np.random.randint(3, 7)

    def prior_hidden_size_x():
        return np.random.randint(15, 40)

    def prior_weight_x():
        return np.random.normal(0, 1)

    return DAGStructuredSCM(
        prior_layers=prior_layers_x,
        prior_hidden_size=prior_hidden_size_x,
        prior_weight=prior_weight_x,
        edge_drop_prob=0.5,
        activation=lambda value: np.tanh(value),
    )

def replay_covariates(
    dataset_seed: int,
    num_features: int,
    num_rows: int,
) -> np.ndarray:
    """Replay and continue one covariate SCM without changing caller RNG state."""

    numpy_state = np.random.get_state()
    try:
        np.random.seed(dataset_seed)
        values = _covariate_scm().generate_dataset(num_features, num_rows)
    finally:
        np.random.set_state(numpy_state)
    assert values.shape == (num_rows, num_features)
    assert np.isfinite(values).all()
    return values

def generate_extra_normalized_context(
    frame: pd.DataFrame,
    x_columns: list[str],
    artifact: dict,
    maximum_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Continue the exact covariate SCM, then draw A and Y from saved mechanisms."""

    assert len(frame) == TRAINED_CONTEXT_LENGTH
    assert maximum_length >= len(frame)
    metadata = artifact["metadata"]
    assert metadata["num_samples"] == TRAINED_CONTEXT_LENGTH
    assert metadata["num_features"] == len(x_columns)
    # Read beside the selected current artifact; recorded manifests keep their original paths.
    raw_path = Path(artifact["metadata_path"]).parent.parent / "datasets_raw" / Path(metadata["raw_dataset_path"]).name
    assert raw_path.is_file(), raw_path
    raw_frame = pd.read_csv(raw_path)
    raw_x_columns = [f"x{index}" for index in range(len(x_columns))]
    assert set(raw_x_columns).issubset(raw_frame.columns)

    replay = replay_covariates(
        int(metadata["seed"]),
        len(x_columns),
        maximum_length,
    )
    saved_raw = raw_frame[raw_x_columns].to_numpy(dtype=np.float64)
    replay_error = np.abs(replay[: len(saved_raw)] - saved_raw)
    maximum_replay_error = float(replay_error.max(initial=0.0))
    assert np.allclose(replay[:len(saved_raw)],saved_raw,atol=1e-6,rtol=1e-6), (
        "Covariate SCM replay did not reproduce the immutable source; "
        f"maximum absolute error={maximum_replay_error}"
    )

    normalization = metadata["normalization"]
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float64)
    x_std = np.asarray(normalization["x_std"], dtype=np.float64)
    assert np.all(x_std > 0)
    extra_raw = replay[len(frame) :]
    extra_x = (extra_raw - x_mean) / x_std
    extra = pd.DataFrame(extra_x, columns=x_columns)

    propensity_bundle = load_propensity_model_bundle(
        artifact["propensity_model_path"], map_location="cpu"
    )
    propensity_model = build_propensity_model_from_bundle(
        propensity_bundle, device="cpu", freeze=True
    )
    outcome_bundle = load_outcome_model_bundle(
        artifact["outcome_model_path"], map_location="cpu"
    )
    outcome_model = build_outcome_model_from_bundle(
        outcome_bundle, device="cpu", freeze=True
    )
    x_tensor = torch.tensor(extra_x, dtype=torch.float32)
    with torch.no_grad():
        propensity = propensity_model(x_tensor).reshape(-1).numpy()
    assert np.all((propensity >= 0.0) & (propensity <= 1.0))
    rng = np.random.default_rng(seed)
    treatment = (rng.random(len(extra)) < propensity).astype(np.float32)
    latent_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    latent = torch.randn(
        len(extra),
        int(metadata["latent_dim_u"]),
        generator=latent_generator,
    )
    with torch.no_grad():
        outcome = outcome_model(
            x_tensor,
            torch.tensor(treatment, dtype=torch.float32),
            latent,
        ).reshape(-1)
    extra["treatment"] = treatment
    extra["outcome"] = outcome.numpy()
    extra["propensity"] = propensity
    assert np.isfinite(extra.to_numpy()).all()
    return extra, {
        "dataset_id": int(artifact["dataset_id"]),
        "dataset_seed": int(metadata["seed"]),
        "maximum_length": maximum_length,
        "extra_rows": len(extra),
        "covariate_method": "exact seed replay and continuation of saved covariate SCM",
        "treatment_method": "fresh Bernoulli draws from saved normalized propensity model",
        "outcome_method": "fresh U~N(0,1) draws through saved normalized outcome model",
        "maximum_raw_covariate_replay_error": maximum_replay_error,
    }

def nested_contexts(
    frame: pd.DataFrame,
    x_columns: list[str],
    artifact: dict,
    lengths: Iterable[int],
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict]:
    """Build deterministic nested contexts around the immutable 1,024-row context."""

    lengths = tuple(sorted(int(value) for value in lengths))
    assert lengths == tuple(sorted(set(lengths)))
    assert lengths[0] > 0
    maximum_length = lengths[-1]
    assert len(frame) == TRAINED_CONTEXT_LENGTH
    extra, diagnostics = generate_extra_normalized_context(
        frame, x_columns, artifact, maximum_length, seed
    )
    rng = np.random.default_rng(seed + 2)
    permutation = rng.permutation(len(frame))
    contexts: dict[int, pd.DataFrame] = {}
    columns = x_columns + ["treatment", "outcome", "propensity"]
    base = frame[columns].reset_index(drop=True)
    for length in lengths:
        if length < len(base):
            positions = np.sort(permutation[:length])
            context = base.iloc[positions].reset_index(drop=True)
        elif length == len(base):
            context = base.copy()
        else:
            context = pd.concat(
                [base, extra.iloc[: length - len(base)][columns]],
                ignore_index=True,
            )
        assert len(context) == length
        assert np.isfinite(context.to_numpy()).all()
        contexts[length] = context
    diagnostics["subsample_method"] = (
        "one seeded permutation; shorter contexts use nested prefixes without replacement"
    )
    diagnostics["contexts_nested_below_1024"] = True
    diagnostics["contexts_nested_above_1024"] = True
    return contexts, diagnostics

def msm_reference_from_samples(
    outcomes: np.ndarray,
    pi_query: np.ndarray,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-threshold empirical MSM bounds on an exact Gamma grid."""

    assert outcomes.ndim == 2
    queries, samples = outcomes.shape
    assert pi_query.shape == (queries,)
    lower = np.empty((queries, len(gamma_grid)), dtype=np.float64)
    upper = np.empty_like(lower)
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
                    value += ordered[:full].sum(dtype=np.float64) * (
                        high_weight - low_weight
                    )
                if fraction > 0.0 and full < samples:
                    value += fraction * ordered[full] * (high_weight - low_weight)
                destination[query_index, gamma_index] = value
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(lower <= upper + 1e-10)
    return lower, upper

def reference_queries(
    artifact: dict,
    frame: pd.DataFrame,
    x_columns: list[str],
    row_count: int,
    oracle_mc_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Construct fixed queries and high-MC closed-form MSM reference bounds."""

    quantiles = np.linspace(0.1, 0.9, row_count)
    row_indices = propensity_stratified_row_indices(frame, row_count, quantiles)
    query = make_query_grid(frame, x_columns, row_indices, GAMMA_GRID)
    base_query = (
        query.groupby(["source_row_index", "a_query"], sort=False).first().reset_index()
    )
    assert len(base_query) == row_count * 2

    outcome_bundle = load_outcome_model_bundle(
        artifact["outcome_model_path"], map_location="cpu"
    )
    outcome_model = build_outcome_model_from_bundle(
        outcome_bundle, device="cpu", freeze=True
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    outcome_rows: list[np.ndarray] = []
    for _, base_row in base_query.iterrows():
        latent = torch.randn(oracle_mc_samples, 1, generator=generator)
        x = (
            torch.tensor(
                base_row[x_columns].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            .reshape(1, -1)
            .expand(oracle_mc_samples, -1)
        )
        arm = torch.full(
            (oracle_mc_samples,),
            float(base_row["a_query"]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            values = outcome_model(x, arm, latent).reshape(-1).numpy()
        outcome_rows.append(values)
    outcomes = np.stack(outcome_rows)
    lower, upper = msm_reference_from_samples(
        outcomes,
        base_query["pi_query"].to_numpy(),
        GAMMA_GRID,
    )
    query["theta_lower"] = lower.reshape(-1)
    query["theta_upper"] = upper.reshape(-1)
    query["oracle_mc_samples"] = oracle_mc_samples
    assert np.all(query["theta_lower"] <= query["theta_upper"] + 1e-10)
    return query

def add_prediction_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_gmm_diagnostics(frame, "theta_upper")
    from experiments.shared.evaluation import gmm_cdf

    lower_pi, lower_mu, lower_sigma = gmm_arrays(result, "lower")
    result["lower_pit"] = gmm_cdf(
        result["theta_lower"].to_numpy(),
        lower_pi,
        lower_mu,
        lower_sigma,
    )
    result["predicted_bound_width"] = result["upper_mean"] - result["lower_mean"]
    result["oracle_bound_width"] = result["theta_upper"] - result["theta_lower"]
    return result


# Run


LENGTHS=(128,256,512,1024,2048,4096)

def fit(values):
    values=np.asarray(values,dtype=np.float32);mean=float(values.mean());std=float(values.std())
    return mean,1. if std<1e-6 else std

def run(study,smoke):
    name='e7_'+study+('_smoke' if smoke else '')
    run=observed_run(name,smoke);data=fresh(OBSERVED_DATA/name);fresh(run.path/'fragments')

    device=torch.device('cuda:0');model,_=load_pifm(MSM_CHECKPOINT,device)
    try:
        if study=='context':
            arts=cohort_artifacts('context')
            ids=cohort_ids('context')[:1] if smoke else cohort_ids('context')

            for pos,i in enumerate(ids):
                artifact=arts[i];frame,xcols=normalized_dataset(artifact)
                contexts,replay=nested_contexts(frame,xcols,artifact,LENGTHS,20260726+1000003*i)
                stats=read(normalized_root()/'scalers'/f'scaler_{i}.json')
                assert stats['y_fit_column']=='outcome'
                # Same query selection, MC count and latent seed as the retained context study.
                query=reference_queries(artifact,frame,xcols,4,16384,20260726+1009*i)
                dest=fresh(data/f'dgp_{i}');write(dest/'replay.json',replay)
                outputs=[]
                for length,context in contexts.items():
                    # Recover raw outcomes before fitting this context length; potential outcomes never enter fit.
                    raw_y=context.outcome.to_numpy()*stats['y_std']+stats['y_mean']
                    mean,std=fit(raw_y)
                    local=context.copy();local['outcome']=(raw_y-mean)/std
                    local.to_csv(dest/f'context_{length}.csv',index=False)
                    write(dest/f'scaler_{length}.json',dict(y_mean=mean,y_std=std,y_fit_column='outcome',context_length=length,raw_outcome_recovery='inverse saved affine wrapper',x_mean=stats['x_mean'],x_std=stats['x_std']))
                    q=query.copy()
                    for side in ['lower','upper']:q[f'theta_{side}']=(q[f'theta_{side}']*stats['y_std']+stats['y_mean']-mean)/std
                    pred=pifm_predict_frame(model,local,xcols,q,device,chunk_size=len(q));pred=add_prediction_diagnostics(pred)
                    pred['dataset_id']=i;pred['context_length']=length;pred['sigma_obs']=std;outputs.append(pred)
                pd.concat(outputs).to_parquet(run.path/'fragments'/f'dgp_{i}.parquet',index=False)
                run.manifest['completed_ids'].append(i);run.save();print(f'E7 context {pos+1}/{len(ids)}',flush=True)
        else:
            source=input_path('dimension_raw_root')
            for dimension in [2,5,10,20,40]:

                family=fresh(data/f'dimension_{dimension}')
                raw=discover_raw_causal_artifacts(str(source/f'dimension_{dimension}'))
                raw=raw[:1] if smoke else raw
                for artifact in raw:
                    normalize_artifact_dataset(artifact,str(family),y_fit_column='outcome')
                arts=discover_causal_artifacts(str(family))
                for artifact in arts:
                    i=artifact['dataset_id'];frame,xcols=normalized_dataset(artifact)
                    query=reference_queries(artifact,frame,xcols,4,16384,20260726+7000000+i)
                    pred=pifm_predict_frame(model,frame,xcols,query,device,chunk_size=len(query));pred=add_prediction_diagnostics(pred)
                    pred['dataset_id']=dimension*1000000+i;pred['replicate_index']=i;pred['covariate_dimension']=dimension
                    pred['sigma_obs']=read(family/'scalers'/f'scaler_{i}.json')['y_std']
                    pred.to_parquet(run.path/'fragments'/f'dim_{dimension}_dgp_{i}.parquet',index=False)
                    run.manifest['completed_ids'].append(int(pred.dataset_id.iloc[0]))
                run.save();print(f'E7 dimension {dimension} completed',flush=True)
        run.done()
    except BaseException:
        run.done('failed');raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['context', 'dimension'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.study, args.smoke)


if __name__ == '__main__':
    main()
