"""ACIC semi-synthetic mechanisms, frozen PI-FM evaluation and reports."""


from __future__ import annotations

from experiments.shared.paths import REPORTS
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from experiments.shared.paths import DATA
from experiments.shared.runtime import Run, cluster_bootstrap, completed, dataset_root, read, sha, write
from scipy.special import expit


# Acic dgp


FACTORS = ('x_2', 'x_21', 'x_24')
# Declared 10-covariate variant: the ten numeric covariates with the most distinct values
# (ties broken by column order). A data property only; no outcome information is used.
D10_COLUMNS = ['x_36', 'x_27', 'x_6', 'x_37', 'x_44', 'x_34', 'x_26', 'x_28', 'x_3', 'x_20']
GAMMA_CONFOUNDING = 0.25   # NeuralCSA semi-synthetic config value
NOISE_SD = 0.1
PROPENSITY_SEED = 20261107
PROPENSITY_TERMS = 8

def encode(raw: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Full one-hot encoding (variant 'full', 82 columns) or the declared 10-covariate subset."""
    assert raw.shape == (4802, 58)
    if variant == 'd10':
        return raw[D10_COLUMNS].astype(float)
    assert variant == 'full'
    numeric = raw.drop(columns=list(FACTORS)).astype(float)
    dummies = [pd.get_dummies(raw[f].astype(str), prefix=f, dtype=float) for f in FACTORS]
    out = pd.concat([numeric] + dummies, axis=1)
    assert out.shape[1] == 82 and np.isfinite(out.to_numpy()).all()
    return out

@dataclass(frozen=True)
class Mechanism:
    z: np.ndarray          # (n, d) globally standardized covariates
    pi: np.ndarray         # (n,) observational P(A = 1 | x)
    score: np.ndarray      # (n,) sum(Z) / (d + 1), the true CAPO of arm 1

def mechanism(x: pd.DataFrame) -> Mechanism:
    values = x.to_numpy(dtype=float)
    std = values.std(axis=0)
    assert np.all(std > 0), 'constant covariate column'
    z = (values - values.mean(axis=0)) / std
    rng = np.random.default_rng(PROPENSITY_SEED)
    terms = rng.choice(z.shape[1], size=PROPENSITY_TERMS, replace=False)
    eta = z[:, terms] @ rng.normal(size=PROPENSITY_TERMS)
    eta = eta / eta.std()
    pi = expit(eta)
    d = z.shape[1]
    return Mechanism(z=z, pi=pi, score=z.sum(axis=1) / (d + 1))

def weight_coefficients(pi):
    """NeuralCSA weight w(u) = c0 + c1 u with E[w(U)] = 1 and w pi <= 1."""
    g = GAMMA_CONFOUNDING
    pi = np.asarray(pi, dtype=float)
    first = g >= 2 - 1 / pi
    c0 = np.where(first, g, 2 - 1 / pi)
    c1 = np.where(first, 2 * (1 - g), 2 * (1 / pi - 1))
    return c0, c1

def simulate(mech: Mechanism, seed: int) -> dict:
    """One replication: hidden U, treatment, potential and factual outcomes for all rows."""
    rng = np.random.default_rng(seed)
    n, d = mech.z.shape
    u = rng.uniform(size=n)
    c0, c1 = weight_coefficients(mech.pi)
    p_full = (c0 + c1 * u) * mech.pi
    assert np.all((p_full >= 0) & (p_full <= 1 + 1e-12))
    a = rng.binomial(1, np.clip(p_full, 0, 1))
    base = (mech.z.sum(axis=1) + u - 0.5) / (d + 1)
    y1 = base + rng.normal(0, NOISE_SD, n)
    y0 = -base + rng.normal(0, NOISE_SD, n)
    return dict(u=u, a=a, y0=y0, y1=y1, y=np.where(a == 1, y1, y0), p_full=p_full)

def latent_density_coefficients(pi, arm):
    """Linear density alpha + beta u of U | x, A = arm on [0, 1]."""
    c0, c1 = weight_coefficients(pi)
    if arm == 1:
        return c0, c1
    pi = np.asarray(pi, dtype=float)
    return (1 - pi * c0) / (1 - pi), -pi * c1 / (1 - pi)

def sample_linear_density(alpha, beta, uniforms):
    """Inverse CDF of density alpha + beta u on [0, 1] (alpha + beta / 2 = 1), vectorized by row."""
    alpha = np.asarray(alpha, float)[:, None]
    beta = np.asarray(beta, float)[:, None]
    assert np.allclose(alpha + beta / 2, 1)
    assert np.all(alpha >= -1e-12) and np.all(alpha + beta >= -1e-12)
    flat = np.abs(beta) < 1e-10
    safe = np.where(flat, 1.0, beta)
    root = (-alpha + np.sqrt(np.maximum(alpha ** 2 + 2 * safe * uniforms, 0))) / safe
    return np.clip(np.where(flat, uniforms, root), 0, 1)

def observational_samples(mech: Mechanism, rows, arm, draws, seed):
    """Exact draws of Y | x, A = arm for each requested row: (len(rows), draws)."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    alpha, beta = latent_density_coefficients(mech.pi[rows], arm)
    u = sample_linear_density(alpha, beta, rng.uniform(size=(len(rows), draws)))
    d = mech.z.shape[1]
    sign = 1.0 if arm == 1 else -1.0
    base = (mech.z[rows].sum(axis=1)[:, None] + u - 0.5) / (d + 1)
    return sign * base + rng.normal(0, NOISE_SD, size=(len(rows), draws))

def true_capo(mech: Mechanism, rows, arm):
    return (1.0 if arm == 1 else -1.0) * mech.score[np.asarray(rows)]

def oracle_msm_gamma(pi):
    """Smallest MSM Gamma containing the true model at x: max odds-ratio deviation over u in [0, 1]."""
    c0, c1 = weight_coefficients(pi)
    pi = np.asarray(pi, float)
    out = np.ones_like(pi)
    for u in (0.0, 1.0):
        p = (c0 + c1 * u) * pi
        with np.errstate(divide='ignore'):
            ratio = (pi / (1 - pi)) / (p / (1 - p))
        out = np.maximum(out, np.maximum(ratio, 1 / ratio))
    return out

def oracle_kl_gamma(pi, grid=200001):
    """Corrected binary KL level max(KL(nu||phi), KL(phi||nu)) between the two arms' U laws.

    Symmetric in the queried arm. Infinite where one density vanishes on a set of positive mass
    (possible only at the boundary branch, where it is reported as inf).
    """
    u = (np.arange(grid) + 0.5) / grid
    out = []
    for p in np.atleast_1d(pi):
        a1, b1 = latent_density_coefficients(np.array([p]), 1)
        a0, b0 = latent_density_coefficients(np.array([p]), 0)
        f1 = a1[0] + b1[0] * u
        f0 = a0[0] + b0[0] * u
        with np.errstate(divide='ignore', invalid='ignore'):
            k10 = np.mean(np.where(f1 > 0, f1 * np.log(f1 / f0), 0.0))
            k01 = np.mean(np.where(f0 > 0, f0 * np.log(f0 / f1), 0.0))
        out.append(max(k10, k01))
    return np.asarray(out)


# Run

PACKAGE = 'e05_acic'
VARIANTS = ('full', 'd10')
REPLICATIONS = 5
CONTEXT_ROWS = 1024
QUERY_ROWS = 64
ORACLE_DRAWS = 16384
MSM_GRID = (1.0, 1.4, 2.0, 3.0, 5.0)
KL_GRID = (0.05, 0.1, 0.25, 0.5, 1.0)
MAX_QUERY_ORACLE_MSM = 9.0     # MSM Gamma extrapolation reaches 9
MAX_QUERY_ORACLE_KL = 2.0
OVERLAP = (0.3, 0.7)           # NeuralCSA overlap mask, applied to the true pi here
# Covariates from github.com/vdorie/aciccomp@4ec804cd0f0ddc138ef3cad7543ec0d6cd3b5cc8.
RAW = DATA / 'acic2016' / 'input_2016.parquet'

def standardize_context(x_context, y_context):
    """Pipeline convention: float32 moments, population std, std < 1e-6 replaced by 1."""
    xv = x_context.astype(np.float32)
    x_mean, x_std = xv.mean(axis=0), xv.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    yv = y_context.astype(np.float32)
    y_mean, y_std = float(yv.mean()), float(yv.std())
    assert y_std > 1e-6
    return x_mean.astype(float), x_std.astype(float), y_mean, y_std

def median_oracles(mech):
    mask = (mech.pi > OVERLAP[0]) & (mech.pi < OVERLAP[1])
    msm = oracle_msm_gamma(mech.pi[mask])
    kl = oracle_kl_gamma(mech.pi[mask], grid=20001)
    return float(np.median(msm)), float(np.median(kl)), int(mask.sum())

def build(smoke):
    from experiments.shared.msm_reference import msm_reference_from_samples
    run = Run(PACKAGE, 'build_smoke' if smoke else 'build', smoke=smoke, gpu=False, threads=4, dataset=True,
              raw_sha256=sha(RAW), replications=REPLICATIONS, context_rows=CONTEXT_ROWS,
              query_rows=QUERY_ROWS, oracle_draws=ORACLE_DRAWS, msm_grid=MSM_GRID, kl_grid=KL_GRID,
              simulation_seed=1000, split_seed=2000, reference_seed=3000)
    out_root = run.path
    raw = pd.read_parquet(RAW)
    reps = 1 if smoke else REPLICATIONS
    summary = {}
    try:
        for variant in VARIANTS:
            x = encode(raw, variant)
            mech = mechanism(x)
            med_msm, med_kl, n_overlap = median_oracles(mech)
            summary[variant] = dict(d=x.shape[1], median_oracle_msm=med_msm, median_oracle_kl=med_kl,
                                    overlap_rows=n_overlap, pi_quantiles=np.quantile(mech.pi, [0, .05, .5, .95, 1]).tolist())
            for rep in range(reps):
                sim = simulate(mech, seed=1000 + rep)
                rng = np.random.default_rng(2000 + rep)
                context = np.sort(rng.choice(len(x), CONTEXT_ROWS, replace=False))
                pool = np.setdiff1d(np.arange(len(x)), context)
                queries = np.sort(rng.choice(pool, QUERY_ROWS, replace=False))
                x_mean, x_std, y_mean, y_std = standardize_context(x.to_numpy()[context], sim['y'][context])
                xcols = [f'x_{j}' for j in range(x.shape[1])]
                ctx = pd.DataFrame((x.to_numpy()[context] - x_mean) / x_std, columns=xcols)
                ctx['treatment'] = sim['a'][context]
                ctx['outcome'] = (sim['y'][context] - y_mean) / y_std
                ctx['propensity'] = mech.pi[context]
                ctx['row'] = context
                rows = []
                for arm in (0, 1):
                    pi_q = mech.pi[queries] if arm == 1 else 1 - mech.pi[queries]
                    draws = observational_samples(mech, queries, arm, ORACLE_DRAWS, seed=3000 + 10 * rep + arm)
                    o_msm = oracle_msm_gamma(mech.pi[queries])
                    o_kl = oracle_kl_gamma(mech.pi[queries], grid=20001)
                    gammas = [(g, 'grid') for g in MSM_GRID] + [(med_msm, 'median_oracle')]
                    truth = (true_capo(mech, queries, arm) - y_mean) / y_std
                    for g, kind in gammas:
                        lo, up = msm_reference_from_samples(draws, pi_q, np.array([g]))
                        rows.append(pd.DataFrame(dict(query_row=queries, a_query=arm, gamma=g, gamma_kind=kind,
                                                      theta_lower=(lo[:, 0] - y_mean) / y_std,
                                                      theta_upper=(up[:, 0] - y_mean) / y_std)))
                    finite = np.where(o_msm <= MAX_QUERY_ORACLE_MSM)[0]
                    for k in finite:
                        lo, up = msm_reference_from_samples(draws[k:k + 1], pi_q[k:k + 1], np.array([o_msm[k]]))
                        rows.append(pd.DataFrame(dict(query_row=[queries[k]], a_query=arm, gamma=o_msm[k], gamma_kind='query_oracle',
                                                      theta_lower=(lo[:, 0] - y_mean) / y_std, theta_upper=(up[:, 0] - y_mean) / y_std)))
                    for frame in rows[-(len(gammas) + len(finite)):]:
                        index = np.searchsorted(queries, frame.query_row.to_numpy())
                        frame['true_capo'] = truth[index]
                        frame['pi_query'] = pi_q[index]
                        frame['oracle_msm'] = o_msm[index]
                        frame['oracle_kl'] = o_kl[index]
                        frame['obs_mean'] = (draws[index].mean(axis=1) - y_mean) / y_std
                msm = pd.concat(rows, ignore_index=True)
                msm['p_a1'] = mech.pi[msm.query_row]
                msm['counterfactual_weight'] = 1 - msm.pi_query
                # Sanity: the true model lies in the MSM set at its own oracle Gamma.
                oracle_rows = msm[msm.gamma_kind.eq('query_oracle')]
                tol = 4 * NOISE_SD / np.sqrt(ORACLE_DRAWS) / y_std
                assert ((oracle_rows.theta_lower <= oracle_rows.true_capo + tol) & (oracle_rows.true_capo <= oracle_rows.theta_upper + tol)).all()
                kl = pd.concat([pd.DataFrame(dict(query_row=queries, a_query=arm, gamma=g, gamma_kind=kind))
                                for arm in (0, 1) for g, kind in [(g, 'grid') for g in KL_GRID] + [(med_kl, 'median_oracle')]], ignore_index=True)
                o_kl_all = oracle_kl_gamma(mech.pi[queries], grid=20001)
                extra = [pd.DataFrame(dict(query_row=[q], a_query=arm, gamma=[o], gamma_kind='query_oracle'))
                         for arm in (0, 1) for q, o in zip(queries, o_kl_all) if o <= MAX_QUERY_ORACLE_KL]
                kl = pd.concat([kl] + extra, ignore_index=True)
                kl['true_capo'] = [(true_capo(mech, [q], a)[0] - y_mean) / y_std for q, a in zip(kl.query_row, kl.a_query)]
                kl['p_a1'] = mech.pi[kl.query_row]
                kl['pi_query'] = np.where(kl.a_query.eq(1), kl.p_a1, 1 - kl.p_a1)
                kl['oracle_kl'] = oracle_kl_gamma(kl.p_a1.to_numpy(), grid=20001)
                qx = pd.DataFrame((x.to_numpy()[queries] - x_mean) / x_std, columns=xcols)
                qx['query_row'] = queries
                target = out_root / variant / f'rep{rep}'
                target.mkdir(parents=True)
                ctx.to_parquet(target / 'context.parquet', index=False)
                qx.to_parquet(target / 'query_x.parquet', index=False)
                msm.to_parquet(target / 'msm_reference.parquet', index=False)
                kl.to_parquet(target / 'kl_queries.parquet', index=False)
                write(target / 'scaling.json', dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                                                    treated_fraction=float(sim['a'][context].mean())))
                print(f'{variant} rep{rep}: d={x.shape[1]} treated={sim["a"][context].mean():.3f} '
                      f'y_std={y_std:.3f} msm_rows={len(msm)} kl_rows={len(kl)}', flush=True)
        run.done(**summary)
    except BaseException:
        run.done('failed')
        raise

def pfn(smoke):
    import torch
    from experiments.shared.evaluation import (KL_CHECKPOINT, MSM_CHECKPOINT, add_gmm_diagnostics, load_pifm,
                                pifm_predict_frame)
    root = dataset_root(PACKAGE, smoke)
    run = Run(PACKAGE, 'pfn_smoke' if smoke else 'pfn', smoke=smoke, gpu=True, threads=4,
              msm_checkpoint_sha256=sha(MSM_CHECKPOINT), kl_checkpoint_sha256=sha(KL_CHECKPOINT), data_root=str(root))
    device = torch.device('cuda:0')
    models = {'msm': load_pifm(MSM_CHECKPOINT, device)[0], 'kl': load_pifm(KL_CHECKPOINT, device)[0]}
    out = []
    timing = []
    try:
        for target in sorted(root.glob('*/rep*')):
            variant, rep = target.parent.name, int(target.name[3:])
            ctx = pd.read_parquet(target / 'context.parquet')
            qx = pd.read_parquet(target / 'query_x.parquet')
            xcols = [c for c in qx.columns if c.startswith('x_')]
            for model_name, table in (('msm', 'msm_reference.parquet'), ('kl', 'kl_queries.parquet')):
                query = pd.read_parquet(target / table).merge(qx, on='query_row', validate='many_to_one')
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred = pifm_predict_frame(models[model_name], ctx, xcols, query, device)
                torch.cuda.synchronize()
                timing.append(dict(variant=variant, rep=rep, model=model_name, rows=len(query), seconds=time.perf_counter() - start))
                truth = 'theta_upper' if model_name == 'msm' else 'true_capo'
                pred = add_gmm_diagnostics(pred, truth).drop(columns=xcols)
                pred['variant'], pred['rep'], pred['model'] = variant, rep, model_name
                out.append(pred)
            print(f'{variant} rep{rep} done', flush=True)
        frame = pd.concat(out, ignore_index=True)
        frame.to_parquet(run.path / 'predictions.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(rows=len(frame))
    except BaseException:
        run.done('failed')
        raise


# Report


EXPORT = REPORTS / 'acic'

def endpoint_stats(f):
    out = {}
    for s in ('lower', 'upper'):
        err = f[f'{s}_mean'] - f[f'theta_{s}']
        out[f'rmse_{s}'] = float(np.sqrt(np.mean(err ** 2)))
        out[f'bias_{s}'] = float(err.mean())
        out[f'cov90_{s}'] = float(((f[f'{s}_q05'] <= f[f'theta_{s}']) & (f[f'theta_{s}'] <= f[f'{s}_q95'])).mean())
    out['outward_joint'] = float(((f.lower_q05 <= f.theta_lower) & (f.upper_q95 >= f.theta_upper)).mean())
    out['pred_width'] = float((f.upper_mean - f.lower_mean).mean())
    out['ref_width'] = float((f.theta_upper - f.theta_lower).mean())
    return out

def cate_endpoint_stats(f):
    out = {f'rmse_{s}': float(np.sqrt(np.mean((f[f'{s}_mean'] - f[f'theta_{s}']) ** 2))) for s in ('lower', 'upper')}
    out['outward_joint'] = float(((f.lower_q05 <= f.theta_lower) & (f.upper_q95 >= f.theta_upper)).mean())
    out['pred_width'] = float((f.upper_mean - f.lower_mean).mean())
    out['ref_width'] = float((f.theta_upper - f.theta_lower).mean())
    return out

def truth_stats(f):
    return dict(true_in_means=float(((f.lower_mean <= f.true_capo) & (f.true_capo <= f.upper_mean)).mean()),
                true_in_outward=float(((f.lower_q05 <= f.true_capo) & (f.true_capo <= f.upper_q95)).mean()),
                width_means=float((f.upper_mean - f.lower_mean).mean()),
                width_outward=float((f.upper_q95 - f.lower_q05).mean()))

def boot(frame, by, stat):
    rows = []
    for key, part in frame.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), 'rows': len(part), **cluster_bootstrap(part, 'rep', stat, replicates=2000)})
    return pd.DataFrame(rows)

def cate_frame(p):
    """Per (variant, rep, query_row, gamma_kind, gamma): CATE bounds from arm-specific heads."""
    keys = ['variant', 'rep', 'query_row', 'gamma_kind', 'gamma']
    p = p[p.gamma_kind.isin(['grid', 'median_oracle'])]
    one, zero = p[p.a_query.eq(1)].set_index(keys), p[p.a_query.eq(0)].set_index(keys)
    out = pd.DataFrame(dict(lower_mean=one.lower_mean - zero.upper_mean, upper_mean=one.upper_mean - zero.lower_mean,
                            lower_q05=one.lower_q05 - zero.upper_q95, upper_q95=one.upper_q95 - zero.lower_q05,
                            true_capo=one.true_capo - zero.true_capo))
    if 'theta_lower' in one:
        out['theta_lower'] = one.theta_lower - zero.theta_upper
        out['theta_upper'] = one.theta_upper - zero.theta_lower
    return out.reset_index()

def report():
    run = completed('e05_acic', 'pfn')
    build = dataset_root(PACKAGE)
    pred = pd.read_parquet(run / 'predictions.parquet')
    EXPORT.mkdir(parents=True, exist_ok=True)
    msm, kl = pred[pred.model.eq('msm')], pred[pred.model.eq('kl')]
    grid = msm[msm.gamma_kind.eq('grid')]
    tables = {
        'msm_endpoint_by_variant': boot(grid, ['variant'], endpoint_stats),
        'msm_endpoint_by_variant_gamma': boot(grid, ['variant', 'gamma'], endpoint_stats),
        'msm_endpoint_by_variant_rep': grid.groupby(['variant', 'rep']).apply(lambda f: pd.Series(endpoint_stats(f)), include_groups=False).reset_index(),
        'msm_truth_by_kind': boot(msm[msm.gamma_kind.ne('grid') | msm.gamma.gt(1)], ['variant', 'gamma_kind'], truth_stats),
        'msm_endpoint_oracle': boot(msm[msm.gamma_kind.ne('grid')], ['variant', 'gamma_kind'], endpoint_stats),
        'kl_truth_by_kind_gamma': boot(kl.assign(g=np.where(kl.gamma_kind.eq('grid'), kl.gamma, np.nan)), ['variant', 'gamma_kind'], truth_stats),
        'kl_truth_by_grid_gamma': boot(kl[kl.gamma_kind.eq('grid')], ['variant', 'gamma'], truth_stats),
        'msm_cate_truth': boot(cate_frame(msm), ['variant', 'gamma_kind'], truth_stats),
        'msm_cate_endpoint': boot(cate_frame(grid), ['variant'], cate_endpoint_stats),
    }
    for name, table in tables.items():
        table.to_csv(EXPORT / f'{name}.csv', index=False)
    timing = pd.read_csv(run / 'timing.csv')
    timing.to_csv(EXPORT / 'pfn_timing.csv', index=False)
    summary = read(build / 'manifest.json')['summary']
    oracle = msm.drop_duplicates(['variant', 'rep', 'query_row']).groupby('variant').oracle_msm.describe(percentiles=[.25, .5, .75])
    oracle.to_csv(EXPORT / 'oracle_msm_gamma_queries.csv')
    write(EXPORT / 'design_summary.json', dict(build=summary, predictions=len(pred), pfn_run=str(run.relative_to(run.parents[2])),
                                                inference_seconds_median=float(timing.seconds.median())))
    # Figures: RMSE and coverage by Gamma; example curves.
    t = tables['msm_endpoint_by_variant_gamma']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for variant, color in (('d10', '#2b6cb0'), ('full', '#c05621')):
        part = t[t.variant.eq(variant)]
        for s, ls in (('lower', '-'), ('upper', '--')):
            axes[0].errorbar(part.gamma, part[f'rmse_{s}'], yerr=[part[f'rmse_{s}'] - part[f'rmse_{s}_lo'], part[f'rmse_{s}_hi'] - part[f'rmse_{s}']],
                             color=color, ls=ls, marker='o', capsize=2, label=f'{variant} {s}')
            axes[1].plot(part.gamma, part[f'cov90_{s}'], color=color, ls=ls, marker='o', label=f'{variant} {s}')
    axes[1].axhline(0.9, color='grey', lw=0.8, ls=':')
    axes[0].set(xlabel='Gamma', ylabel='endpoint RMSE (obs.-std.)', title='MSM PI-FM vs exact reference')
    axes[1].set(xlabel='Gamma', ylabel='90% posterior coverage', ylim=(0, 1), title='Calibration')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXPORT / 'msm_rmse_coverage_by_gamma.png', dpi=130)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
    for row, variant in enumerate(('d10', 'full')):
        part = grid[(grid.variant.eq(variant)) & grid.rep.eq(0)]
        for col, q in enumerate(sorted(part.query_row.unique())[:3]):
            c = part[(part.query_row.eq(q)) & part.a_query.eq(1)].sort_values('gamma')
            ax = axes[row, col]
            ax.plot(c.gamma, c.theta_lower, 'k-', c.gamma, c.theta_upper, 'k-', lw=1.5)
            ax.plot(c.gamma, c.lower_mean, color='#2b6cb0'); ax.plot(c.gamma, c.upper_mean, color='#c05621')
            ax.fill_between(c.gamma, c.lower_q05, c.lower_q95, color='#2b6cb0', alpha=.2)
            ax.fill_between(c.gamma, c.upper_q05, c.upper_q95, color='#c05621', alpha=.2)
            ax.axhline(c.true_capo.iloc[0], color='green', ls=':', lw=1)
            ax.set_title(f'{variant} rep0 row {q}, a=1', fontsize=8)
    fig.supxlabel('Gamma'); fig.supylabel('CAPO bound (obs.-std.)')
    fig.tight_layout()
    fig.savefig(EXPORT / 'example_curves.png', dpi=130)
    print(tables['msm_endpoint_by_variant'].round(4).to_string())
    print(tables['msm_truth_by_kind'].round(3).to_string())
    print(tables['kl_truth_by_kind_gamma'].round(3).to_string())
    print(tables['msm_cate_truth'].round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'pfn', 'report'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'report': report()
    else: {'build': build, 'pfn': pfn}[args.stage](args.smoke)


if __name__ == '__main__':
    main()
