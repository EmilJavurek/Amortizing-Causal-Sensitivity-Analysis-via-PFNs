"""Out-of-prior mechanisms and factual-context evaluation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, ndtr, ndtri
from scipy.stats import t as student_t
from setting_standard.gen_standard_syn import (
    DAGStructuredSCM,
    OutcomeGenerator,
    TreatmentGenerator,
)
from experiments.shared.evaluation import add_gmm_diagnostics, gmm_arrays, gmm_cdf, make_query_grid, pifm_predict_frame, propensity_stratified_row_indices
from experiments.shared.paths import ROOT, OBSERVED_DATA
import argparse
from experiments.shared.observed import observed_run
from experiments.shared.runtime import write, sha, fresh
from experiments.shared.evaluation import load_pifm, MSM_CHECKPOINT, make_query_grid, propensity_stratified_row_indices


FAMILIES = (
    "oscillatory_outcome",
    "heavy_tailed_outcome",
    "extreme_overlap",
    "gamma_extrapolation",
)

TRAINED_GAMMA_GRID = np.linspace(1.0, 5.0, 21)

EXTRAPOLATION_GAMMA_GRID = np.linspace(1.0, 9.0, 41)

@dataclass
class GeneratedDGP:
    """A normalized observational context and its conditional outcome evaluator."""

    family: str
    replicate: int
    frame: pd.DataFrame
    evaluator: "ConditionalOutcomeEvaluator"
    metadata: dict

class ConditionalOutcomeEvaluator:
    """Evaluate a generated potential outcome on normalized x and y scales."""

    def __init__(
        self,
        family: str,
        production_outcome: torch.nn.Module,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ):
        assert family in FAMILIES
        self.family = family
        self.production_outcome = production_outcome.cpu().eval()
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.asarray(x_std, dtype=np.float64)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        assert self.y_std > 0.0

    def evaluate_raw(
        self,
        x_normalized: np.ndarray,
        arm: np.ndarray,
        latent: np.ndarray,
    ) -> np.ndarray:
        x_normalized = np.asarray(x_normalized, dtype=np.float64)
        arm = np.asarray(arm, dtype=np.float64).reshape(-1)
        latent = np.asarray(latent, dtype=np.float64)
        assert x_normalized.ndim == 2
        assert latent.ndim == 2
        assert len(x_normalized) == len(arm) == len(latent)

        if self.family == "oscillatory_outcome":
            return oscillatory_outcome(x_normalized, arm, latent)
        if self.family == "heavy_tailed_outcome":
            return heavy_tailed_outcome(x_normalized, arm, latent)
        return self._evaluate_production(x_normalized, arm, latent)

    def evaluate_normalized(
        self,
        x_normalized: np.ndarray,
        arm: np.ndarray,
        latent: np.ndarray,
    ) -> np.ndarray:
        raw = self.evaluate_raw(x_normalized, arm, latent)
        return (raw - self.y_mean) / self.y_std

    def _evaluate_production(
        self,
        x_normalized: np.ndarray,
        arm: np.ndarray,
        latent: np.ndarray,
        chunk_size: int = 131_072,
    ) -> np.ndarray:
        queries, samples = latent.shape
        x_raw = x_normalized * self.x_std[None, :] + self.x_mean[None, :]
        x_flat = np.repeat(x_raw, samples, axis=0)
        a_flat = np.repeat(arm, samples)
        u_flat = latent.reshape(-1)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(u_flat), chunk_size):
                stop = min(start + chunk_size, len(u_flat))
                values = self.production_outcome(
                    torch.tensor(x_flat[start:stop], dtype=torch.float32),
                    torch.tensor(a_flat[start:stop], dtype=torch.float32),
                    torch.tensor(u_flat[start:stop], dtype=torch.float32),
                )
                outputs.append(values.reshape(-1).cpu().numpy())
        result = np.concatenate(outputs).reshape(queries, samples)
        assert np.isfinite(result).all()
        return result.astype(np.float64)

def production_components(
    seed: int,
    context_rows: int,
    dimension: int = 10,
) -> dict:
    """Sample an in-prior SCM in memory using the production prior definitions."""

    assert dimension == 10, "E9 holds trained covariate dimension fixed"
    np.random.seed(seed)
    torch.manual_seed(seed)

    dag = DAGStructuredSCM(
        prior_layers=lambda: np.random.randint(3, 7),
        prior_hidden_size=lambda: np.random.randint(15, 40),
        prior_weight=lambda: np.random.normal(0, 1),
        edge_drop_prob=0.5,
        activation=lambda value: np.tanh(value),
    )
    x_raw = dag.generate_dataset(dimension, context_rows)

    treatment_generator = TreatmentGenerator(
        prior_layers=lambda: np.random.randint(3, 5),
        prior_hidden_size=lambda: np.random.randint(8, 20),
        prior_weight=lambda: np.random.normal(0, 0.8),
        activation_name="tanh",
        device="cpu",
    )
    treatment_generator.build_propensity_model(dimension)
    treatment, propensity = treatment_generator.generate_treatments(x_raw)

    outcome_generator = OutcomeGenerator(
        prior_layers=lambda: np.random.randint(3, 6),
        prior_hidden_size=lambda: np.random.randint(10, 25),
        prior_weight=lambda: np.random.normal(0, 1.0),
        activation_name="tanh",
        latent_dim=1,
        device="cpu",
    )
    production_outcome = outcome_generator.build_outcome_model(dimension)
    outcome, y0, y1, latent = outcome_generator.generate_outcomes(x_raw, treatment)
    del outcome
    return {
        "x_raw": x_raw,
        "treatment": treatment,
        "propensity": propensity,
        "propensity_bundle": treatment_generator.get_model_bundle(dimension),
        "y0": y0,
        "y1": y1,
        "latent": latent.reshape(-1, 1),
        "production_outcome": production_outcome,
        "architecture": {
            "covariate_layers": int(dag.sampled_num_layers),
            "covariate_hidden_size": int(dag.sampled_hidden_size),
            "treatment_layers": int(treatment_generator.sampled_num_layers),
            "treatment_hidden_size": int(
                treatment_generator.sampled_hidden_size
            ),
            "outcome_layers": int(outcome_generator.sampled_num_layers),
            "outcome_hidden_size": int(outcome_generator.sampled_hidden_size),
        },
    }

def oscillatory_outcome(
    x_normalized: np.ndarray,
    arm: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    """Bounded, high-frequency outcome outside the sampled tanh-network prior."""

    arm = arm[:, None]
    location = 0.55 * np.tanh(
        1.10 * x_normalized[:, 0]
        - 0.75 * x_normalized[:, 1]
        + 0.35 * x_normalized[:, 2] * x_normalized[:, 3]
    )
    oscillation = (
        0.70 * np.sin(5.5 * latent + 1.2 * x_normalized[:, 0, None])
        + 0.25 * np.sin(11.0 * latent - 2.0 * x_normalized[:, 1, None])
    )
    effect = arm * (
        0.45
        + 0.25 * np.tanh(x_normalized[:, 4, None])
        + 0.18 * np.cos(3.0 * latent)
    )
    return location[:, None] + oscillation + effect

def heavy_tailed_outcome(
    x_normalized: np.ndarray,
    arm: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    """Student-t(3) conditional tails, unlike the bounded tanh outcome prior."""

    probability = np.clip(ndtr(latent), 1e-10, 1.0 - 1e-10)
    heavy_noise = student_t.ppf(probability, df=3)
    location = 0.55 * np.tanh(
        0.9 * x_normalized[:, 0]
        - 0.6 * x_normalized[:, 1]
        + 0.25 * x_normalized[:, 2] * x_normalized[:, 3]
    )
    scale = 0.38 + 0.14 * expit(x_normalized[:, 4])
    effect = arm[:, None] * (
        0.45
        + 0.20 * np.tanh(x_normalized[:, 5, None])
        + 0.06 * heavy_noise
    )
    result = (
        location[:, None] + scale[:, None] * heavy_noise + effect
    )
    assert np.isfinite(result).all()
    return result

def extreme_propensity(x_normalized: np.ndarray) -> np.ndarray:
    score = (
        1.00 * x_normalized[:, 0]
        - 0.75 * x_normalized[:, 1]
        + 0.35 * x_normalized[:, 2]
    )
    return np.clip(expit(7.0 * score), 0.001, 0.999)

def potential_outcomes_for_family(
    family: str,
    x_normalized: np.ndarray,
    latent: np.ndarray,
    production_y0: np.ndarray,
    production_y1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "oscillatory_outcome":
        zeros = np.zeros(len(x_normalized))
        ones = np.ones(len(x_normalized))
        return (
            oscillatory_outcome(x_normalized, zeros, latent).reshape(-1),
            oscillatory_outcome(x_normalized, ones, latent).reshape(-1),
        )
    if family == "heavy_tailed_outcome":
        zeros = np.zeros(len(x_normalized))
        ones = np.ones(len(x_normalized))
        return (
            heavy_tailed_outcome(x_normalized, zeros, latent).reshape(-1),
            heavy_tailed_outcome(x_normalized, ones, latent).reshape(-1),
        )
    return production_y0.astype(np.float64), production_y1.astype(np.float64)

def generate_dgp(
    family: str,
    replicate: int,
    seed: int,
    context_rows: int,
    components: dict | None = None,
) -> GeneratedDGP:
    assert family in FAMILIES
    if components is None:
        components = production_components(seed, context_rows)
    x_raw = components["x_raw"].astype(np.float32)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_normalized = (x_raw - x_mean) / x_std

    propensity = components["propensity"].astype(np.float64)
    treatment = components["treatment"].astype(int)
    if family == "extreme_overlap":
        propensity = extreme_propensity(x_normalized)
        rng = np.random.default_rng(seed + 90_001)
        treatment = rng.binomial(1, propensity).astype(int)

    y0, y1 = potential_outcomes_for_family(
        family,
        x_normalized,
        components["latent"],
        components["y0"],
        components["y1"],
    )
    # Match production normalization: float32 moments fitted jointly to y0 and y1.
    potential_values = np.concatenate([y0, y1]).astype(np.float32)
    y_mean = float(potential_values.mean())
    y_std = float(potential_values.std())
    if y_std < 1e-6:
        y_std = 1.0
    outcome = np.where(treatment == 1, y1, y0)

    x_columns = [f"x_{index}" for index in range(x_normalized.shape[1])]
    frame = pd.DataFrame(x_normalized, columns=x_columns)
    frame["treatment"] = treatment
    frame["propensity"] = propensity
    frame["outcome"] = (outcome - y_mean) / y_std
    assert np.isfinite(frame.to_numpy()).all()
    assert set(frame["treatment"].unique()) == {0, 1}

    evaluator = ConditionalOutcomeEvaluator(
        family=family,
        production_outcome=components["production_outcome"],
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    metadata = {
        "family": family,
        "replicate": replicate,
        "seed": seed,
        "context_rows": context_rows,
        "dimension": len(x_columns),
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "y_fit_columns": ["y0", "y1"],
        },
        "propensity": {
            "minimum": float(propensity.min()),
            "q01": float(np.quantile(propensity, 0.01)),
            "q50": float(np.quantile(propensity, 0.50)),
            "q99": float(np.quantile(propensity, 0.99)),
            "maximum": float(propensity.max()),
            "outside_002_098_fraction": float(
                ((propensity < 0.02) | (propensity > 0.98)).mean()
            ),
        },
        "architecture": components["architecture"],
    }
    if family == "extreme_overlap":
        assert metadata["propensity"]["outside_002_098_fraction"] >= 0.25, (
            "Extreme-overlap family did not produce sufficiently extreme propensities"
        )
    return GeneratedDGP(family, replicate, frame, evaluator, metadata)

def midpoint_normal_grid(samples: int) -> np.ndarray:
    assert samples >= 128
    probability = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    latent = ndtri(probability)
    assert np.isfinite(latent).all()
    return latent

def msm_reference_from_samples(
    outcomes: np.ndarray,
    pi_query: np.ndarray,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-threshold analytical MSM formula on an empirical quantile grid."""

    assert outcomes.ndim == 2
    queries, samples = outcomes.shape
    assert pi_query.shape == (queries,)
    assert np.all((pi_query > 0.0) & (pi_query < 1.0))
    upper = np.empty((queries, len(gamma_grid)), dtype=np.float64)
    lower = np.empty_like(upper)
    for query_index in range(queries):
        ascending = np.sort(outcomes[query_index].astype(np.float64))
        for gamma_index, gamma in enumerate(gamma_grid):
            assert gamma >= 1.0
            tail_mass = samples / (gamma + 1.0)
            full = int(np.floor(tail_mass))
            fraction = tail_mass - full
            pi = pi_query[query_index]
            inverse_high = (1.0 - gamma) * pi + gamma
            inverse_low = (1.0 - 1.0 / gamma) * pi + 1.0 / gamma
            raw_sum = (
                tail_mass * inverse_high
                + (samples - tail_mass) * inverse_low
            )
            high_weight = inverse_high / raw_sum
            low_weight = inverse_low / raw_sum
            for destination, ordered in (
                (upper, ascending[::-1]),
                (lower, ascending),
            ):
                value = ordered.sum(dtype=np.float64) * low_weight
                if full > 0:
                    value += (
                        ordered[:full].sum(dtype=np.float64)
                        * (high_weight - low_weight)
                    )
                if fraction > 0.0 and full < samples:
                    value += (
                        fraction
                        * ordered[full]
                        * (high_weight - low_weight)
                    )
                destination[query_index, gamma_index] = value
    assert np.isfinite(upper).all() and np.isfinite(lower).all()
    assert np.all(lower <= upper + 1e-10)
    return lower, upper

def evaluate_oracle(
    generated: GeneratedDGP,
    query_frame: pd.DataFrame,
    x_columns: list[str],
    gamma_grid: np.ndarray,
    oracle_samples: int,
) -> tuple[pd.DataFrame, dict]:
    base = query_frame.drop_duplicates(
        ["source_row_index", "a_query"], keep="first"
    ).reset_index(drop=True)
    assert len(query_frame) == len(base) * len(gamma_grid)
    x_query = base[x_columns].to_numpy()
    a_query = base["a_query"].to_numpy()
    pi_query = base["pi_query"].to_numpy()

    latent = midpoint_normal_grid(oracle_samples)
    latent_matrix = np.broadcast_to(latent, (len(base), oracle_samples))
    outcomes = generated.evaluator.evaluate_normalized(
        x_query, a_query, latent_matrix
    )
    lower, upper = msm_reference_from_samples(
        outcomes, pi_query, gamma_grid
    )

    coarse_samples = oracle_samples // 8
    coarse_latent = midpoint_normal_grid(coarse_samples)
    coarse_matrix = np.broadcast_to(
        coarse_latent, (len(base), coarse_samples)
    )
    coarse_outcomes = generated.evaluator.evaluate_normalized(
        x_query, a_query, coarse_matrix
    )
    coarse_lower, coarse_upper = msm_reference_from_samples(
        coarse_outcomes, pi_query, gamma_grid
    )

    result = query_frame.copy()
    result["base_query_index"] = np.repeat(
        np.arange(len(base)), len(gamma_grid)
    )
    result["theta_lower"] = lower.reshape(-1)
    result["theta_upper"] = upper.reshape(-1)
    result["oracle_width"] = result["theta_upper"] - result["theta_lower"]
    assert np.all(result["theta_lower"] <= result["theta_upper"] + 1e-10)

    difference_lower = lower - coarse_lower
    difference_upper = upper - coarse_upper
    convergence = {
        "family": generated.family,
        "replicate": generated.replicate,
        "oracle_samples": oracle_samples,
        "coarse_samples": coarse_samples,
        "lower_max_abs_difference": float(np.abs(difference_lower).max()),
        "upper_max_abs_difference": float(np.abs(difference_upper).max()),
        "lower_rmse_difference": float(
            np.sqrt(np.mean(difference_lower**2))
        ),
        "upper_rmse_difference": float(
            np.sqrt(np.mean(difference_upper**2))
        ),
    }
    return result, convergence

def evaluate_one_dgp(
    generated: GeneratedDGP,
    pifm,
    device: torch.device,
    row_count: int,
    query_quantiles: np.ndarray,
    oracle_samples: int,
) -> tuple[pd.DataFrame, dict]:
    frame = generated.frame
    x_columns = [
        column for column in frame.columns if column.startswith("x_")
    ]
    x_columns.sort(key=lambda value: int(value.split("_")[1]))
    gamma_grid = (
        EXTRAPOLATION_GAMMA_GRID
        if generated.family == "gamma_extrapolation"
        else TRAINED_GAMMA_GRID
    )
    row_indices = propensity_stratified_row_indices(
        frame, row_count, query_quantiles
    )
    query = make_query_grid(
        frame, x_columns, row_indices, gamma_grid
    )
    reference, convergence = evaluate_oracle(
        generated,
        query,
        x_columns,
        gamma_grid,
        oracle_samples,
    )
    prediction = pifm_predict_frame(
        pifm,
        frame,
        x_columns,
        reference,
        device,
    )
    prediction["family"] = generated.family
    prediction["replicate"] = generated.replicate
    prediction["dataset_id"] = (
        f"{generated.family}_{generated.replicate:03d}"
    )
    prediction["gamma_region"] = np.where(
        prediction["gamma"] <= 5.0 + 1e-12,
        "trained_range",
        "extrapolated",
    )
    prediction = add_gmm_diagnostics(prediction, "theta_upper")
    lower_pi, lower_mu, lower_sigma = gmm_arrays(prediction, "lower")
    prediction["lower_pit"] = gmm_cdf(
        prediction["theta_lower"].to_numpy(),
        lower_pi,
        lower_mu,
        lower_sigma,
    )
    return prediction, convergence

# Factual-context evaluation and artifact persistence.



def main(smoke,seed):
    suffix=('_seed'+str(seed) if seed!=20260726 else '')+('_smoke' if smoke else '')
    run=observed_run('e9'+suffix,smoke)
    data=fresh(OBSERVED_DATA/('e9'+suffix))
    model,_=load_pifm(MSM_CHECKPOINT,torch.device('cuda:0'))
    fresh(run.path/'fragments')
    audits=[]
    try:
        for rep in range(1 if smoke else 64):
            replicate_seed=seed+10000*rep
            components=production_components(replicate_seed,1024)
            for family in FAMILIES:
                dest=fresh(data/family/f'dgp_{rep:03d}')
                generated=generate_dgp(family,rep,replicate_seed,1024,components=components)
                stats=generated.metadata['normalization']
                xcols=[f'x_{i}' for i in range(10)]
                y0,y1=potential_outcomes_for_family(family,generated.frame[xcols].to_numpy(),components['latent'],components['y0'],components['y1'])
                raw=pd.DataFrame(components['x_raw'].astype(np.float32),columns=[f'x{i}' for i in range(10)])
                raw['treatment']=generated.frame.treatment.to_numpy();raw['propensity']=generated.frame.propensity.to_numpy()
                raw['u0']=components['latent'].reshape(-1);raw['y0']=y0;raw['y1']=y1
                raw['outcome']=np.where(raw.treatment==1,raw.y1,raw.y0);raw['ite']=raw.y1-raw.y0
                raw.to_csv(dest/'context_raw.csv',index=False)
                torch.save(components['propensity_bundle'],dest/'propensity_bnn_raw.pt')
                architecture=components['architecture']
                torch.save(dict(model_class='OutcomeMLP',input_dim_x=10,input_dim_a=1,latent_dim_u=1,output_dim_y=1,num_layers=architecture['outcome_layers'],hidden_size=architecture['outcome_hidden_size'],activation_name='tanh',state_dict=components['production_outcome'].state_dict()),dest/'outcome_bnn_raw.pt')
                write(dest/'metadata_raw.json',dict(**generated.metadata,raw_context='context_raw.csv',production_outcome='outcome_bnn_raw.pt',production_propensity='propensity_bnn_raw.pt',family_mechanism_source='experiments/robustness/out_of_prior.py',family_mechanism_sha256=sha(ROOT/'experiments/robustness/out_of_prior.py'),family_outcome_uses_normalized_x=family in ['oscillatory_outcome','heavy_tailed_outcome'],extreme_overlap_uses_analytic_propensity=family=='extreme_overlap'))
                audit=dict(family=family,replicate=rep)
                # Only factual context Y determines the replacement scale.
                values=raw.outcome.to_numpy(dtype=np.float32);mean=float(values.mean());std=float(values.std());std=1. if std<1e-6 else std
                generated.frame['outcome']=(raw.outcome.to_numpy()-mean)/std
                generated.evaluator.y_mean=mean;generated.evaluator.y_std=std
                generated.metadata['normalization']={**stats,'y_mean':mean,'y_std':std,'y_fit_column':'outcome'}
                generated.metadata['normalization'].pop('y_fit_columns',None)
                generated.frame.to_csv(dest/'context_observed.csv',index=False)
                write(dest/'scaler_observed.json',generated.metadata['normalization'])
                pred,convergence=evaluate_one_dgp(generated,model,torch.device('cuda:0'),8,np.array([.02,.10,.25,.40,.60,.75,.90,.98]),65536)
                pred['sigma_obs']=std
                pred.to_parquet(run.path/'fragments'/f'{family}_{rep:03d}.parquet',index=False)
                audit.update(convergence);audits.append(audit)
                run.manifest['completed_ids'].append(f'{family}_{rep:03d}')
            run.save();print(f'E9 {rep+1}/{1 if smoke else 64}',flush=True)
        pd.DataFrame(audits).to_csv(run.path/'oracle_convergence.csv',index=False)
        run.manifest.update(master_seed=seed,checkpoint_sha256=sha(MSM_CHECKPOINT),oracle_samples=65536)
        run.done()
    except BaseException:
        pd.DataFrame(audits).to_csv(run.path/'oracle_convergence.csv',index=False)
        run.done('failed');raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=20260726)
    args = parser.parse_args()
    main(args.smoke, args.seed)
