"""IHDP semi-synthetic mechanisms, numerical references and PI-FM/comparator evaluation."""


from __future__ import annotations

from experiments.shared.paths import REPORTS
import argparse
import experiments.applications.acic as dgp
import experiments.label_accuracy.kl_discrete as kl_reference
import experiments.label_accuracy.kl_discrete as kr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from dataclasses import asdict
from experiments.applications.acic import KL_GRID, MAX_QUERY_ORACLE_KL, MAX_QUERY_ORACLE_MSM, MSM_GRID, OVERLAP, boot, cate_endpoint_stats, cate_frame, endpoint_stats, standardize_context, truth_stats
from experiments.comparisons.benchmark import join
from experiments.comparisons.benchmark import references
from experiments.comparisons.benchmark import row_metrics
from experiments.comparisons.benchmark import summarize
from experiments.comparisons.benchmark import capo_to_cate
from experiments.comparisons.benchmark import load_tasks
from experiments.comparisons.benchmark import pfn_outputs
from experiments.shared.paths import DATA
from experiments.shared.runtime import Run, completed, dataset_root, read, sha, write
from experiments.shared.msm_reference import msm_reference_from_samples
from multiprocessing import get_context
from scipy.special import ndtr, ndtri


# Ihdp dgp


SOURCE = DATA / 'ihdp' / 'ihdp_npci_1.csv'
SOURCE_SHA256 = '2bdc8b94fdf2f669b1d3d49325fd9359d88418566545003867841312a1653411'
# Source: github.com/AMLab-Amsterdam/CEVAE, datasets/IHDP/csv/ihdp_npci_1.csv.
CEVAE_COMMIT = '9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd'
LEVELS = 16384

def load_covariates():
    assert sha(SOURCE) == SOURCE_SHA256, 'IHDP source file changed'
    raw = pd.read_csv(SOURCE, header=None)
    assert raw.shape == (747, 30)
    x = raw.iloc[:, 5:].astype(float)
    x.columns = [f'x{j}' for j in range(1, 26)]
    return x

def exact_quantiles(mech, rows, arm, levels=LEVELS):
    """Quantile midpoints (k+0.5)/levels of Y | x_row, A=arm for each row: (len(rows), levels)."""
    rows = np.asarray(rows)
    d = mech.z.shape[1]
    sign = 1.0 if arm == 1 else -1.0
    alpha, beta = dgp.latent_density_coefficients(mech.pi[rows], arm)
    u = (np.arange(400) + 0.5) / 400
    target = (np.arange(levels) + 0.5) / levels
    sd = dgp.NOISE_SD
    out = np.empty((len(rows), levels))
    for i, row in enumerate(rows):
        w = (alpha[i] + beta[i] * u) / 400
        assert abs(w.sum() - 1) < 1e-9 and (w >= -1e-15).all()
        m = sign * (mech.z[row].sum() + u - 0.5) / (d + 1)
        grid = np.linspace(m.min() - 9 * sd, m.max() + 9 * sd, 20001)
        cdf = ndtr((grid[:, None] - m[None]) / sd) @ w
        out[i] = np.interp(target, cdf, grid)
    return out

def atoms(quantiles, k=1024):
    """Equal-mass atoms (block means of sorted quantile midpoints)."""
    rows, n = quantiles.shape
    return quantiles.reshape(rows, k, n // k).mean(axis=2)


# Run

PACKAGE = 'e05b_ihdp'
REPLICATIONS = 5
CONTEXT_ROWS = 597
QUERY_ROWS = 64
ATOMS = 1024
WORKERS = 8

def median_oracles(mech):
    mask = (mech.pi > OVERLAP[0]) & (mech.pi < OVERLAP[1])
    return float(np.median(dgp.oracle_msm_gamma(mech.pi[mask]))), float(np.median(dgp.oracle_kl_gamma(mech.pi[mask], grid=20001))), int(mask.sum())

def _kl_chunk(args):
    os.environ['OMP_NUM_THREADS'] = '1'
    y_atoms, pi, gammas = args
    lo = np.empty((len(pi), gammas.shape[1]))
    up = np.empty_like(lo)
    gap = 0.0
    for i in range(len(pi)):
        mu = y_atoms[i].mean()
        su = sl = (0.0, 0.0)
        for j, g in enumerate(gammas[i]):
            u, iu = kl_reference.nu_mean_upper(y_atoms[i], g, su)
            l, il = kl_reference.nu_mean_upper(-y_atoms[i], g, sl)
            su, sl = iu['params'], il['params']
            gap = max(gap, abs(iu['dual_gap']), abs(il['dual_gap']))
            up[i, j] = pi[i] * mu + (1 - pi[i]) * u
            lo[i, j] = pi[i] * mu + (1 - pi[i]) * (-l)
    return lo, up, gap

def kl_references(y_atoms, pi, gammas, pool):
    """gammas: (rows, G) increasing per row. Returns lower, upper (rows, G) and the worst dual gap."""
    chunks = np.array_split(np.arange(len(pi)), WORKERS * 4)
    results = pool.map(_kl_chunk, [(y_atoms[c], pi[c], gammas[c]) for c in chunks if len(c)])
    lo = np.concatenate([r[0] for r in results])
    up = np.concatenate([r[1] for r in results])
    return lo, up, max(r[2] for r in results)

def build(smoke):
    run = Run(PACKAGE, 'build_smoke' if smoke else 'build', smoke=smoke, gpu=False, threads=WORKERS, dataset=True,
              source_sha256=SOURCE_SHA256, cevae_commit=CEVAE_COMMIT, replications=REPLICATIONS,
              context_rows=CONTEXT_ROWS, query_rows=QUERY_ROWS, levels=LEVELS, kl_atoms=ATOMS,
              msm_grid=MSM_GRID, kl_grid=KL_GRID, simulation_seed=1000, split_seed=2000)
    out_root = run.path
    x = load_covariates()
    mech = dgp.mechanism(x)
    med_msm, med_kl, n_overlap = median_oracles(mech)
    summary = dict(d=x.shape[1], median_oracle_msm=med_msm, median_oracle_kl=med_kl, overlap_rows=n_overlap,
                   pi_quantiles=np.quantile(mech.pi, [0, .05, .5, .95, 1]).tolist(), reps={})
    reps = 1 if smoke else REPLICATIONS
    nq = 8 if smoke else QUERY_ROWS
    pool = get_context('fork').Pool(WORKERS)
    try:
        for rep in range(reps):
            start = time.time()
            sim = dgp.simulate(mech, seed=1000 + rep)
            rng = np.random.default_rng(2000 + rep)
            context = np.sort(rng.choice(len(x), CONTEXT_ROWS, replace=False))
            pool_rows = np.setdiff1d(np.arange(len(x)), context)
            queries = np.sort(rng.choice(pool_rows, nq, replace=False))
            x_mean, x_std, y_mean, y_std = standardize_context(x.to_numpy()[context], sim['y'][context])
            xcols = [f'x_{j}' for j in range(x.shape[1])]
            ctx = pd.DataFrame((x.to_numpy()[context] - x_mean) / x_std, columns=xcols)
            ctx['treatment'], ctx['outcome'] = sim['a'][context], (sim['y'][context] - y_mean) / y_std
            ctx['propensity'], ctx['row'] = mech.pi[context], context
            o_msm, o_kl = dgp.oracle_msm_gamma(mech.pi[queries]), dgp.oracle_kl_gamma(mech.pi[queries], grid=20001)
            msm_rows, kl_rows, gaps = [], [], []
            for arm in (0, 1):
                pi_q = mech.pi[queries] if arm == 1 else 1 - mech.pi[queries]
                quant = exact_quantiles(mech, queries, arm)
                truth = (dgp.true_capo(mech, queries, arm) - y_mean) / y_std
                std = lambda v: (v - y_mean) / y_std
                common_cols = dict(query_row=queries, a_query=arm, true_capo=truth, pi_query=pi_q, oracle_msm=o_msm,
                                   oracle_kl=o_kl, obs_mean=std(quant.mean(axis=1)))
                for g, kind in [(g, 'grid') for g in MSM_GRID] + [(med_msm, 'median_oracle')]:
                    lo, up = msm_reference_from_samples(quant, pi_q, np.array([g]))
                    msm_rows.append(pd.DataFrame(dict(common_cols, gamma=g, gamma_kind=kind, theta_lower=std(lo[:, 0]), theta_upper=std(up[:, 0]))))
                for k in np.where(o_msm <= MAX_QUERY_ORACLE_MSM)[0]:
                    lo, up = msm_reference_from_samples(quant[k:k + 1], pi_q[k:k + 1], np.array([o_msm[k]]))
                    msm_rows.append(pd.DataFrame({**{c: [v[k]] if isinstance(v, np.ndarray) else v for c, v in common_cols.items()},
                                                  'gamma': o_msm[k], 'gamma_kind': 'query_oracle',
                                                  'theta_lower': std(lo[:, 0]), 'theta_upper': std(up[:, 0])}))
                # KL: grid + median oracle for every query; per-query oracle (<= 2) as an extra column set
                kl_g = np.array(sorted(KL_GRID) + [med_kl])
                order = np.argsort(kl_g)
                gam = np.tile(kl_g[order], (len(queries), 1))
                at = atoms(quant, ATOMS)
                lo, up, gap = kl_references(at, pi_q, gam, pool)
                gaps.append(gap)
                for j, g in enumerate(kl_g[order]):
                    kind = 'median_oracle' if g == med_kl else 'grid'
                    kl_rows.append(pd.DataFrame(dict(common_cols, gamma=g, gamma_kind=kind, theta_lower=std(lo[:, j]), theta_upper=std(up[:, j]))))
                ok = np.where(o_kl <= MAX_QUERY_ORACLE_KL)[0]
                if len(ok):
                    lo, up, gap = kl_references(at[ok], pi_q[ok], o_kl[ok][:, None], pool)
                    gaps.append(gap)
                    kl_rows.append(pd.DataFrame({**{c: v[ok] if isinstance(v, np.ndarray) else v for c, v in common_cols.items()},
                                                 'gamma': o_kl[ok], 'gamma_kind': 'query_oracle',
                                                 'theta_lower': std(lo[:, 0]), 'theta_upper': std(up[:, 0])}))
            msm, kl = pd.concat(msm_rows, ignore_index=True), pd.concat(kl_rows, ignore_index=True)
            for f in (msm, kl):
                f['p_a1'] = mech.pi[f.query_row]
                f['counterfactual_weight'] = 1 - f.pi_query
            tol = 4 * dgp.NOISE_SD / np.sqrt(LEVELS) / y_std + 1e-6
            for name, f in (('msm', msm), ('kl', kl)):
                o = f[f.gamma_kind.eq('query_oracle')]
                assert ((o.theta_lower <= o.true_capo + tol) & (o.true_capo <= o.theta_upper + tol)).all(), f'{name}: truth outside reference at oracle Gamma'
                assert (f.theta_lower <= f.theta_upper + 1e-9).all()
            qx = pd.DataFrame((x.to_numpy()[queries] - x_mean) / x_std, columns=xcols)
            qx['query_row'] = queries
            target = out_root / f'rep{rep}'
            target.mkdir(parents=True)
            ctx.to_parquet(target / 'context.parquet', index=False)
            qx.to_parquet(target / 'query_x.parquet', index=False)
            msm.to_parquet(target / 'msm_reference.parquet', index=False)
            kl.to_parquet(target / 'kl_reference.parquet', index=False)
            write(target / 'scaling.json', dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                                                treated_fraction=float(sim['a'][context].mean()), context=context, queries=queries))
            summary['reps'][rep] = dict(seconds=time.time() - start, worst_dual_gap=max(gaps), y_std=y_std,
                                        msm_rows=len(msm), kl_rows=len(kl))
            print(f'rep{rep}: {summary["reps"][rep]}', flush=True)
        pool.close()
        run.done(**summary)
    except BaseException:
        pool.terminate()
        run.done('failed')
        raise

def pfn(smoke):
    import torch
    from experiments.shared.evaluation import KL_CHECKPOINT, MSM_CHECKPOINT, add_gmm_diagnostics, load_pifm, pifm_predict_frame
    root = dataset_root(PACKAGE, smoke)
    run = Run(PACKAGE, 'pfn_smoke' if smoke else 'pfn', smoke=smoke, gpu=True, threads=4,
              msm_checkpoint_sha256=sha(MSM_CHECKPOINT), kl_checkpoint_sha256=sha(KL_CHECKPOINT), data_root=str(root))
    device = torch.device('cuda:0')
    models = {'msm': load_pifm(MSM_CHECKPOINT, device)[0], 'kl': load_pifm(KL_CHECKPOINT, device)[0]}
    out, timing = [], []
    try:
        for target in sorted(root.glob('rep*')):
            rep = int(target.name[3:])
            ctx = pd.read_parquet(target / 'context.parquet')
            qx = pd.read_parquet(target / 'query_x.parquet')
            xcols = [c for c in qx.columns if c.startswith('x_')]
            for model in ('msm', 'kl'):
                query = pd.read_parquet(target / f'{model}_reference.parquet').merge(qx, on='query_row', validate='many_to_one')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = pifm_predict_frame(models[model], ctx, xcols, query, device)
                torch.cuda.synchronize()
                timing.append(dict(rep=rep, model=model, rows=len(query), seconds=time.perf_counter() - t0))
                pred = add_gmm_diagnostics(pred, 'theta_upper').drop(columns=xcols)
                pred['rep'], pred['model'] = rep, model
                out.append(pred)
        frame = pd.concat(out, ignore_index=True)
        frame.to_parquet(run.path / 'predictions.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(rows=len(frame))
    except BaseException:
        run.done('failed')
        raise

def cohort(smoke):
    """Shared-benchmark task format (MSM): task ihdp_rep<r>, cohort 'ihdp'."""
    import torch
    from experiments.shared.evaluation import MSM_CHECKPOINT, load_pifm

    from experiments.comparisons.benchmark import GAMMAS
    from experiments.comparisons.benchmark import pfn_on_rows
    from experiments.comparisons.benchmark import write_task
    from experiments.comparisons.benchmark import prediction_root, finish_build
    from experiments.shared.runtime import fresh
    pfn_run = completed(PACKAGE, 'pfn_smoke' if smoke else 'pfn')
    root_in = dataset_root(PACKAGE, smoke)
    run = Run('benchmark_ihdp', 'cohort_ihdp_smoke' if smoke else 'cohort_ihdp', smoke=smoke, gpu=True, threads=4, dataset=True,
              gammas=GAMMAS, source=str(root_in), msm_checkpoint_sha256=sha(MSM_CHECKPOINT))
    root = run.path
    predictions_root = fresh(prediction_root('benchmark_ihdp', smoke))
    device = torch.device('cuda:0')
    model = load_pifm(MSM_CHECKPOINT, device)[0]
    pred = pd.read_parquet(pfn_run / 'predictions.parquet')
    x = load_covariates()
    mech = dgp.mechanism(x)
    try:
        for target in sorted(root_in.glob('rep*')):
            rep = int(target.name[3:])
            scaling = read(target / 'scaling.json')
            context = pd.read_parquet(target / 'context.parquet')
            xcols = [c for c in context.columns if c.startswith('x_')]
            qx = pd.read_parquet(target / 'query_x.parquet')
            ref = pd.read_parquet(target / 'msm_reference.parquet')
            ref = ref[ref.gamma_kind.isin(['grid', 'median_oracle'])]
            p = pred[(pred.rep == rep) & (pred.model == 'msm') & pred.gamma_kind.isin(['grid', 'median_oracle'])]
            cols = [c for c in p.columns if c.startswith(('upper_', 'lower_'))]
            queries = ref.merge(p[['query_row', 'a_query', 'gamma'] + cols], on=['query_row', 'a_query', 'gamma'], validate='one_to_one')
            queries = queries.merge(qx, on='query_row', validate='many_to_one').rename(columns={'query_row': 'row'})
            queries['query_id'] = queries.row.map({r: n for n, r in enumerate(sorted(queries.row.unique()))})
            rows = context.row.to_numpy()
            apo = []
            for arm in (0, 1):
                pi = mech.pi[rows] if arm == 1 else 1 - mech.pi[rows]
                quant = exact_quantiles(mech, rows, arm, levels=4096)
                lo, up = msm_reference_from_samples(quant, pi, np.array(GAMMAS))
                truth = dgp.true_capo(mech, rows, arm).mean()
                for k, g in enumerate(GAMMAS):
                    apo.append(dict(a_query=arm, gamma=g, apo_lower=(lo[:, k].mean() - scaling['y_mean']) / scaling['y_std'],
                                    apo_upper=(up[:, k].mean() - scaling['y_mean']) / scaling['y_std'],
                                    apo_true=(truth - scaling['y_mean']) / scaling['y_std']))
            pfn_ctx = pfn_on_rows(model, context, xcols, context[['row'] + xcols], device)
            task = f'ihdp_rep{rep}'
            meta = dict(task_id=task, cohort='ihdp', rep=rep, d=len(xcols), n_context=len(context), true_propensity_available=True,
                        y_units='observed-standardized', apo_levels=4096,
                        capo_reference='e05b_ihdp exact observational law, 16,384 quantile levels, closed-form MSM')
            write_task(root, predictions_root, task, meta, context, queries, pd.DataFrame(apo), pfn_ctx)
            run.manifest.setdefault('tasks', []).append(task)
            run.save()
            print(task, flush=True)
        finish_build(run, predictions_root)
    except BaseException:
        run.done('failed')
        raise


# Neuralcsa realistic

MSM_LEVELS = (1.4, 2.0, 3.0, 5.0)
KL_LEVELS = (0.1, 0.25, 0.5, 1.0)
TASKS = ('acic_d10_rep0', 'acic_d10_rep1', 'acic_full_rep0', 'acic_full_rep1', 'ihdp_rep0', 'ihdp_rep1')
NEURALCSA_QUERY_ROWS = 8

class ExactLawMap:
    """Oracle outcome map u -> F^{-1}(Phi(u)) per query instance, in observed-standardized units."""

    def __init__(self, x_query, tables, z_grid):
        import torch
        self.x = x_query
        self.tables = tables            # (m, Z) standardized outcomes at z_grid
        self.z0, self.dz = float(z_grid[0]), float(z_grid[1] - z_grid[0])
        self.Z = tables.shape[1]
        self.torch = torch

    def evaluate(self, x_query, a_query, latent):
        torch = self.torch
        assert x_query.shape == self.x.shape and torch.allclose(x_query, self.x), 'instance order changed'
        pos = ((latent - self.z0) / self.dz).clamp(0, self.Z - 1 - 1e-6)
        lo = pos.floor().long()
        frac = pos - lo
        left = torch.gather(self.tables, 1, lo)
        right = torch.gather(self.tables, 1, lo + 1)
        return left + frac * (right - left)

def mechanism_for(task):
    if task.meta['cohort'] == 'ihdp':
        root = dataset_root('e05b_ihdp')
        return dgp.mechanism(load_covariates()), read(root / f"rep{task.meta['rep']}" / 'scaling.json')
    raw = pd.read_parquet(DATA / 'acic2016' / 'input_2016.parquet')
    root = dataset_root('e05_acic')
    return (dgp.mechanism(dgp.encode(raw, task.meta['variant'])),
            read(root / task.meta['variant'] / f"rep{task.meta['rep']}" / 'scaling.json'))

def query_frame(task, levels):
    q = task.queries.drop_duplicates(['query_id', 'a_query'])
    q = q[q.query_id < NEURALCSA_QUERY_ROWS][['query_id', 'row', 'a_query', 'pi_query'] + task.xcols]
    return pd.concat([q.assign(gamma=g) for g in levels], ignore_index=True)

def neuralcsa(smoke):
    import torch
    from sklearn.linear_model import LogisticRegression
    from scipy.special import ndtr
    import experiments.comparisons.neuralcsa as neuralcsa
    stage1_cfg, stage2_cfg = neuralcsa.Stage1Config(seed=20261108), neuralcsa.Stage2Config(seed=20261118)
    if smoke:
        stage1_cfg = neuralcsa.Stage1Config(hidden_size=32, max_steps=100, batch_size=256, log_every=20, seed=20261108)
        stage2_cfg = neuralcsa.Stage2Config(train_mc_samples=128, q0_mc_samples=512, eval_mc_samples=1024, steps_per_penalty=30,
                                     penalty_weights=(10.0, 100.0, 1000.0), feasibility_tolerance=0.10,
                                     msm_dense_grid_points=1025, log_every=15, seed=20261118)
    run = Run('e05b_ihdp', 'neuralcsa_smoke' if smoke else 'neuralcsa', smoke=smoke, gpu=True, threads=4,
              stage1=asdict(stage1_cfg), stage2=asdict(stage2_cfg), tasks=TASKS[:1] if smoke else TASKS,
              query_rows=NEURALCSA_QUERY_ROWS, msm_levels=MSM_LEVELS, kl_levels=KL_LEVELS)
    device = torch.device('cuda:0')
    tasks = [t for t in load_tasks() if t.task_id in TASKS]
    if not smoke:
        tasks += [t for t in load_tasks(cohort_set='ihdp') if t.task_id in TASKS]
    tasks = tasks[:1] if smoke else tasks
    z_grid = np.linspace(-8, 8, 4001)
    pool = get_context('fork').Pool(8)
    rows, timing = [], []
    try:
        for task in tasks:
            mech, scaling = mechanism_for(task)
            x, a, y = task.observed()
            start = time.perf_counter()
            stage1 = neuralcsa.fit_stage1_outcome_flow(torch.tensor(x, dtype=torch.float32), torch.tensor(a, dtype=torch.float32),
                                                torch.tensor(y, dtype=torch.float32), stage1_cfg, device)
            e_model = LogisticRegression(max_iter=2000).fit(x, a)
            stage1_seconds = time.perf_counter() - start
            for model, levels in (('msm', MSM_LEVELS), ('kl', KL_LEVELS)):
                q = query_frame(task, levels)
                xq = torch.tensor(q[task.xcols].to_numpy(), dtype=torch.float32, device=device)
                aq = torch.tensor(q.a_query.to_numpy(), dtype=torch.float32, device=device)
                e_hat = e_model.predict_proba(q[task.xcols].to_numpy())[:, 1]
                pi_hat = np.clip(np.where(q.a_query == 1, e_hat, 1 - e_hat), 0.01, 0.99)
                gamma = torch.tensor(q.gamma.to_numpy(), dtype=torch.float32)
                # exact-law tables and references for these instances
                tables = np.empty((len(q), len(z_grid)))
                quant = {}
                for arm in (0, 1):
                    m = q.a_query.eq(arm).to_numpy()
                    qu = exact_quantiles(mech, q.row.to_numpy()[m], arm)
                    quant[arm] = (m, qu)
                    lev = (np.arange(qu.shape[1]) + 0.5) / qu.shape[1]
                    tables[m] = np.stack([np.interp(ndtr(z_grid), lev, row) for row in qu])
                tables = (tables - scaling['y_mean']) / scaling['y_std']
                oracle = ExactLawMap(xq, torch.tensor(tables, dtype=torch.float32, device=device), z_grid)
                ref_lo, ref_up = np.empty(len(q)), np.empty(len(q))
                for arm, (m, qu) in quant.items():
                    std_q = (qu - scaling['y_mean']) / scaling['y_std']
                    pi = q.pi_query.to_numpy()[m]
                    if model == 'msm':
                        from experiments.shared.msm_reference import msm_reference_from_samples
                        g = q.gamma.to_numpy()[m]
                        for k in range(len(pi)):
                            lo, up = msm_reference_from_samples(std_q[k:k + 1], pi[k:k + 1], np.array([g[k]]))
                            ref_lo[np.where(m)[0][k]], ref_up[np.where(m)[0][k]] = lo[0, 0], up[0, 0]
                    else:
                        lo, up, _ = kl_references(atoms(std_q, 1024), pi, q.gamma.to_numpy()[m][:, None], pool)
                        ref_lo[m], ref_up[m] = lo[:, 0], up[:, 0]
                for variant, outcome_map, pi in (('oracle_stage', oracle, q.pi_query.to_numpy()), ('full', stage1.model, pi_hat)):
                    for bound in ('upper', 'lower'):
                        t0 = time.perf_counter()
                        res = neuralcsa.solve_fixed_gamma_batched(outcome_map=outcome_map, x_query=xq, a_query=aq,
                                                           pi_query=torch.tensor(pi, dtype=torch.float32), gamma_target=gamma,
                                                           bound_type=bound, sensitivity_model=model, config=stage2_cfg, device=device)
                        torch.cuda.synchronize()
                        timing.append(dict(task_id=task.task_id, model=model, variant=variant, bound=bound,
                                           stage2_seconds=time.perf_counter() - t0, stage1_seconds=stage1_seconds))
                        rows.append(q[['query_id', 'row', 'a_query', 'gamma', 'pi_query']].assign(
                            task_id=task.task_id, model=model, variant=variant, head=bound, theta=res.theta,
                            gamma_realized=res.gamma, violation=res.violation,
                            reference=ref_up if bound == 'upper' else ref_lo))
                print(task.task_id, model, 'done', flush=True)
        pool.close()
        frame = pd.concat(rows, ignore_index=True)
        frame.to_parquet(run.path / 'neuralcsa.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(rows=len(frame))
    except BaseException:
        pool.terminate()
        run.done('failed')
        raise


# Validate kl

EXPORT = REPORTS / 'ihdp'
VALIDATION_ATOMS = (256, 1024, 4096)

def capo(y_atoms, pi, gamma):
    up, iu = kr.nu_mean_upper(y_atoms, gamma)
    lo, il = kr.nu_mean_upper(-y_atoms, gamma)
    mu = y_atoms.mean()
    return pi * mu + (1 - pi) * (-lo), pi * mu + (1 - pi) * up, max(abs(iu['dual_gap']), abs(il['dual_gap']))

def gaussian():
    mu, sigma, pi = 0.3, 0.7, 0.4
    fine = ndtri((np.arange(65536) + 0.5) / 65536)
    rows = []
    for k in VALIDATION_ATOMS:
        atoms = mu + sigma * fine.reshape(k, -1).mean(axis=1)
        for g in KL_GRID + (2.0,):
            lo, up, gap = capo(atoms, pi, g)
            exact = (pi * mu + (1 - pi) * (mu - sigma * np.sqrt(2 * g)), pi * mu + (1 - pi) * (mu + sigma * np.sqrt(2 * g)))
            rows.append(dict(atoms=k, gamma=g, lower=lo, upper=up, exact_lower=exact[0], exact_upper=exact[1],
                             err_lower=lo - exact[0], err_upper=up - exact[1], err_upper_rel_sigma=(up - exact[1]) / sigma, dual_gap=gap))
    return pd.DataFrame(rows)

def _job(args):
    os.environ['OMP_NUM_THREADS'] = '1'
    y_atoms, pi, gammas = args
    out = []
    for g in gammas:
        out.append(capo(y_atoms, pi, g))
    return out

def refinement(pool, build_root, mech, n_laws):
    scaling = read(build_root / 'rep0' / 'scaling.json')
    ref = pd.read_parquet(build_root / 'rep0' / 'kl_reference.parquet')
    grid = ref[ref.gamma_kind.isin(['grid', 'median_oracle'])]
    laws = grid.drop_duplicates(['query_row', 'a_query']).head(n_laws)
    gammas = sorted(grid.gamma.unique())
    jobs, keys = [], []
    for _, r in laws.iterrows():
        q = exact_quantiles(mech, [int(r.query_row)], int(r.a_query), levels=65536)
        q = (q - scaling['y_mean']) / scaling['y_std']
        for k in VALIDATION_ATOMS:
            jobs.append((atoms(q, k)[0], float(r.pi_query), gammas))
            keys.append((int(r.query_row), int(r.a_query), k))
    rows = []
    for (row, arm, k), res in zip(keys, pool.map(_job, jobs)):
        for g, (lo, up, gap) in zip(gammas, res):
            rows.append(dict(query_row=row, a_query=arm, atoms=k, gamma=g, lower=lo, upper=up, dual_gap=gap))
    f = pd.DataFrame(rows)
    fine = f[f.atoms == 4096].set_index(['query_row', 'a_query', 'gamma'])[['lower', 'upper']]
    f = f.join(fine, on=['query_row', 'a_query', 'gamma'], rsuffix='_4096')
    f['delta_lower'] = f.lower - f.lower_4096
    f['delta_upper'] = f.upper - f.upper_4096
    stored = grid.set_index(['query_row', 'a_query', 'gamma'])[['theta_lower', 'theta_upper']]
    f = f.join(stored, on=['query_row', 'a_query', 'gamma'])
    return f

def recertify(pool, build_root, mech, reps):
    out = []
    for rep in reps:
        scaling = read(build_root / f'rep{rep}' / 'scaling.json')
        ref = pd.read_parquet(build_root / f'rep{rep}' / 'kl_reference.parquet')
        laws = ref.drop_duplicates(['query_row', 'a_query'])
        jobs, meta = [], []
        for arm in (0, 1):
            rows = laws[laws.a_query.eq(arm)].query_row.to_numpy()
            quant = (exact_quantiles(mech, rows, arm) - scaling['y_mean']) / scaling['y_std']
            at = atoms(quant, 1024)
            for i, row in enumerate(rows):
                part = ref[(ref.query_row == row) & (ref.a_query == arm)].sort_values('gamma')
                jobs.append((at[i], float(part.pi_query.iloc[0]), part.gamma.tolist()))
                meta.append(part)
        for part, res in zip(meta, pool.map(_job, jobs)):
            out.append(part.assign(rep=rep, lower_new=[r[0] for r in res], upper_new=[r[1] for r in res], dual_gap_new=[r[2] for r in res]))
        print(f'recertified rep{rep}', flush=True)
    f = pd.concat(out, ignore_index=True)
    f['diff'] = np.maximum((f.lower_new - f.theta_lower).abs(), (f.upper_new - f.theta_upper).abs())
    return f

def recertify_neuralcsa(pool):
    """Re-solve the KL references used in the NeuralCSA realistic-data table (ACIC subset + IHDP)."""


    from experiments.comparisons.benchmark import load_tasks
    f = pd.read_parquet(completed('e05b_ihdp', 'neuralcsa') / 'neuralcsa.parquet')
    f = f[f.model.eq('kl') & f.variant.eq('full')]
    tasks = {t.task_id: t for t in load_tasks() + load_tasks(cohort_set='ihdp') if t.task_id in set(f.task_id)}
    out = []
    for tid, part in f.groupby('task_id'):
        mech, scaling = mechanism_for(tasks[tid])
        laws = part.drop_duplicates(['row', 'a_query'])
        jobs, meta = [], []
        for _, law in laws.iterrows():
            q = (exact_quantiles(mech, [int(law.row)], int(law.a_query)) - scaling['y_mean']) / scaling['y_std']
            sub = part[(part.row == law.row) & (part.a_query == law.a_query)].drop_duplicates('gamma').sort_values('gamma')
            jobs.append((atoms(q, 1024)[0], float(law.pi_query), sub.gamma.tolist()))
            meta.append(sub)
        for sub, res in zip(meta, pool.map(_job, jobs)):
            for (_, row), (lo, up, gap) in zip(sub.iterrows(), res):
                stored = part[(part.row == row.row) & (part.a_query == row.a_query) & (part.gamma == row.gamma)]
                ref = dict(zip(stored['head'], stored.reference))
                out.append(dict(task_id=tid, row=row.row, a_query=row.a_query, gamma=row.gamma,
                                diff=max(abs(lo - ref['lower']), abs(up - ref['upper'])), dual_gap=gap))
    return pd.DataFrame(out)

def validate_kl(smoke):
    run = Run('e05b_ihdp', 'validate_kl_smoke' if smoke else 'validate_kl', smoke=smoke, gpu=False, threads=8,
              atoms=VALIDATION_ATOMS, tolerances=dict(stationary=kr.TOL_STATIONARY, normal=kr.TOL_NORMAL, feasible=kr.TOL_FEASIBLE, gap=kr.TOL_GAP))
    build_root = dataset_root('e05b_ihdp')
    mech = dgp.mechanism(load_covariates())
    pool = get_context('fork').Pool(8)
    try:
        g = gaussian()
        r = refinement(pool, build_root, mech, 2 if smoke else 12)
        c = recertify(pool, build_root, mech, [0] if smoke else range(5))
        n = None if smoke else recertify_neuralcsa(pool)
        tag = '_smoke' if smoke else ''
        EXPORT.mkdir(parents=True, exist_ok=True)
        g.to_csv(EXPORT / f'kl_validation_gaussian{tag}.csv', index=False)
        r.to_csv(EXPORT / f'kl_validation_refinement{tag}.csv', index=False)
        c[['rep', 'query_row', 'a_query', 'gamma', 'gamma_kind', 'theta_lower', 'theta_upper', 'lower_new', 'upper_new', 'diff', 'dual_gap_new']].to_csv(
            EXPORT / f'kl_validation_recertify{tag}.csv', index=False)
        k1024 = r[r.atoms == 1024]
        summary = dict(
            gaussian_max_abs_err={int(k): float(g[g.atoms == k][['err_lower', 'err_upper']].abs().max().max()) for k in VALIDATION_ATOMS},
            gaussian_max_abs_err_rel_sigma={int(k): float((g[g.atoms == k][['err_lower', 'err_upper']].abs() / 0.7).max().max()) for k in VALIDATION_ATOMS},
            refinement_max_abs_change_vs_4096={int(k): float(r[r.atoms == k][['delta_lower', 'delta_upper']].abs().max().max()) for k in VALIDATION_ATOMS},
            refinement_median_abs_change_1024_vs_4096=float(k1024[['delta_lower', 'delta_upper']].abs().stack().median()),
            refinement_1024_vs_stored_max=float(np.maximum((k1024.lower - k1024.theta_lower).abs(), (k1024.upper - k1024.theta_upper).abs()).max()),
            recertified_rows=len(c), recertify_max_abs_diff=float(c['diff'].max()), recertify_max_dual_gap=float(c.dual_gap_new.max()))
        if n is not None:
            n.to_csv(EXPORT / 'kl_validation_recertify_neuralcsa.csv', index=False)
            summary.update(neuralcsa_recertified_rows=len(n), neuralcsa_recertify_max_abs_diff=float(n['diff'].max()))
        write(EXPORT / f'kl_validation_summary{tag}.json', summary)
        print(summary, flush=True)
        pool.close()
        run.done(**{k: v for k, v in summary.items() if not isinstance(v, dict)})
    except BaseException:
        pool.terminate()
        run.done('failed')
        raise


# Report


GAMMAS = (1.4, 2.0, 3.0, 5.0)

def pfn_tables(pred):
    out = {}
    for model in ('msm', 'kl'):
        p = pred[pred.model.eq(model)].assign(variant='ihdp')
        grid = p[p.gamma_kind.eq('grid')]
        out[f'{model}_endpoint'] = boot(grid, ['variant'], endpoint_stats)
        out[f'{model}_endpoint_by_gamma'] = boot(grid, ['variant', 'gamma'], endpoint_stats)
        out[f'{model}_endpoint_by_rep'] = grid.groupby('rep').apply(lambda f: pd.Series(endpoint_stats(f)), include_groups=False).reset_index()
        out[f'{model}_truth_by_kind'] = boot(p, ['variant', 'gamma_kind'], truth_stats)
        out[f'{model}_endpoint_oracle'] = boot(p[p.gamma_kind.ne('grid')], ['variant', 'gamma_kind'], endpoint_stats)
        out[f'{model}_cate_endpoint'] = boot(cate_frame(grid), ['variant'], cate_endpoint_stats)
    return out

def comparator_table():
    tasks = load_tasks(cohort_set='ihdp')
    ref_capo, ref_cate, ref_apo = references(tasks)
    pfn_capo, pfn_cate, pfn_apo = pfn_outputs(tasks)
    preds = {('CATE', 'PI-FM (direct)'): pfn_cate}
    e10 = completed('e10_tabpfn_msm', 'retained_ihdp')
    if e10:
        preds[('CATE', 'TabPFN + closed form')] = pd.read_parquet(e10 / 'cate.parquet')
    e8 = completed('e08_blearner', 'authors_ihdp')
    if e8:
        preds[('CATE', "B-Learner (authors' code)")] = pd.read_parquet(e8 / 'cate.parquet')
        preds[('CATE', "B-Learner plug-in (authors' code)")] = capo_to_cate(pd.read_parquet(e8 / 'capo_plugin.parquet'))
    ate = lambda apo: (lambda one, zero: pd.DataFrame(dict(lower=one.lower - zero.upper, upper=one.upper - zero.lower)).reset_index())(
        apo[apo.a_query.eq(1)].set_index(['task_id', 'gamma']), apo[apo.a_query.eq(0)].set_index(['task_id', 'gamma']))
    ref_ate = ate(ref_apo.rename(columns={'ref_lower': 'lower', 'ref_upper': 'upper'})).rename(columns={'lower': 'ref_lower', 'upper': 'ref_upper'})
    truth = ref_apo.pivot_table(index='task_id', columns='a_query', values='true', aggfunc='first')
    ref_ate['true'] = ref_ate.task_id.map(truth[1] - truth[0])
    preds[('ATE', 'PI-FM (mean of CAPO)')] = ate(pfn_apo)
    if e10:
        preds[('ATE', 'TabPFN + closed form')] = ate(pd.read_parquet(e10 / 'apo.parquet'))
    e9 = completed('e09_dorn_guo', 'qb_ihdp')
    if e9:
        qb = pd.read_parquet(e9 / 'qb_bounds.parquet')
        for method, part in qb.groupby('method'):
            preds[('ATE', f'Dorn-Guo {method}')] = part.rename(columns={'ate_lower': 'lower', 'ate_upper': 'upper'})[['task_id', 'gamma', 'lower', 'upper']]
    rows = []
    for (estimand, method), pred in preds.items():
        keys = ['task_id', 'query_id', 'gamma'] if estimand == 'CATE' else ['task_id', 'gamma']
        ref = ref_cate if estimand == 'CATE' else ref_ate
        pred = pred[np.isin(np.round(pred.gamma, 6), GAMMAS)][keys + ['lower', 'upper']].assign(gamma=lambda f: np.round(f.gamma, 6))
        r = ref.assign(gamma=lambda f: np.round(f.gamma, 6))
        m = row_metrics(join(pred, r[np.isin(r.gamma, GAMMAS)], keys))
        s = summarize(m, by=('group',), replicates=1000)
        rows.append(s.assign(estimand=estimand, method=method))
    out = pd.concat(rows, ignore_index=True)
    out['width_ratio'] = out.width / out.ref_width
    timing = []
    if e10:
        t = pd.read_csv(e10 / 'timing.csv'); timing.append(dict(method='TabPFN + closed form', median_seconds=float((t.fit_seconds + t.query_seconds).median())))
    if e9:
        t = pd.read_csv(e9 / 'timing.csv')
        for m_, part in t.groupby('method'):
            timing.append(dict(method=f'Dorn-Guo {m_}', median_seconds=float(part.fit_plus_query_seconds.median())))
    if e8:
        t = pd.read_csv(e8 / 'timing.csv')
        cols = [c for c in t.columns if c.endswith('seconds')]
        timing.append(dict(method="B-Learner (authors' code), all Gamma", median_seconds=float(t.groupby('task_id')[cols].sum().sum(axis=1).median())))
    return out, pd.DataFrame(timing)

def neuralcsa_table():
    run = completed('e05b_ihdp', 'neuralcsa')
    if run is None:
        return None, None
    f = pd.read_parquet(run / 'neuralcsa.parquet')
    group = lambda t: 'ihdp' if t.startswith('ihdp') else ('acic_full' if '_full_' in t else 'acic_d10')
    f['group'] = f.task_id.map(group)
    f['err'] = f.theta - f.reference
    rows = [f.rename(columns={'variant': 'method'}).assign(method=lambda d: 'NeuralCSA ' + d.method)[['task_id', 'group', 'model', 'method', 'head', 'query_id', 'a_query', 'gamma', 'err', 'reference', 'theta']]]
    # PI-FM and TabPFN on the same (task, query, arm, gamma)
    tasks = {t.task_id: t for t in load_tasks() + load_tasks(cohort_set='ihdp') if t.task_id in set(f.task_id)}
    e5 = pd.read_parquet(completed('e05_acic', 'pfn') / 'predictions.parquet')
    e5b = pd.read_parquet(completed('e05b_ihdp', 'pfn') / 'predictions.parquet')
    e10 = {c: pd.read_parquet(completed('e10_tabpfn_msm', r) / 'capo.parquet') for c, r in (('default', 'retained'), ('ihdp', 'retained_ihdp'))
           if completed('e10_tabpfn_msm', r)}
    for (tid, model), part in f.groupby(['task_id', 'model']):
        t = tasks[tid]
        key = part.drop_duplicates(['query_id', 'a_query', 'gamma', 'head'])
        rowmap = t.queries.drop_duplicates('query_id').set_index('query_id').row
        if tid.startswith('ihdp'):
            p = e5b[(e5b.rep == t.meta['rep']) & (e5b.model == model)]
        else:
            p = e5[(e5.variant == t.meta['variant']) & (e5.rep == t.meta['rep']) & (e5.model == model)]
        p = p[p.gamma_kind.eq('grid')]
        for head in ('upper', 'lower'):
            k = key[key['head'] == head].assign(query_row=lambda d: d.query_id.map(rowmap))
            j = k.merge(p[['query_row', 'a_query', 'gamma', f'{head}_mean']], on=['query_row', 'a_query', 'gamma'], how='left', validate='one_to_one')
            rows.append(j.assign(method='PI-FM', err=j[f'{head}_mean'] - j.reference, theta=j[f'{head}_mean'])[rows[0].columns])
            cs = 'ihdp' if tid.startswith('ihdp') else 'default'
            if model == 'msm' and cs in e10:
                c = e10[cs][e10[cs].task_id.eq(tid)]
                j2 = k.merge(c[['query_id', 'a_query', 'gamma', head]], on=['query_id', 'a_query', 'gamma'], how='left', validate='one_to_one')
                rows.append(j2.assign(method='TabPFN + closed form', err=j2[head] - j2.reference, theta=j2[head])[rows[0].columns])
    allr = pd.concat(rows, ignore_index=True)
    assert allr.err.notna().all(), 'missing comparator rows'
    table = allr.groupby(['model', 'group', 'method']).agg(rmse=('err', lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                                          bias=('err', 'mean'), rows=('err', 'size')).reset_index()
    t = pd.read_csv(run / 'timing.csv')
    timing = t.groupby(['variant', 'model']).apply(lambda d: d.stage2_seconds.sum() / d.task_id.nunique(), include_groups=False).rename('stage2_seconds_per_task').reset_index()
    timing['stage1_seconds_per_task'] = float(t.drop_duplicates('task_id').stage1_seconds.median())
    return table, timing

def report():
    EXPORT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(completed('e05b_ihdp', 'pfn') / 'predictions.parquet')
    tables = pfn_tables(pred)
    comp, comp_timing = comparator_table()
    tables['comparison_summary'], tables['comparison_timing'] = comp, comp_timing
    ncsa, ncsa_timing = neuralcsa_table()
    if ncsa is not None:
        tables['neuralcsa_realistic'], tables['neuralcsa_timing'] = ncsa, ncsa_timing
    for name, t in tables.items():
        t.to_csv(EXPORT / f'{name}.csv', index=False)
    write(EXPORT / 'design_summary.json', dict(build=read(dataset_root(PACKAGE) / 'manifest.json')['summary'],
                                                pfn_timing=pd.read_csv(completed('e05b_ihdp', 'pfn') / 'timing.csv').seconds.median()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for model, color in (('msm', '#2b6cb0'), ('kl', '#c05621')):
        t = tables[f'{model}_endpoint_by_gamma']
        for s, ls in (('lower', '-'), ('upper', '--')):
            axes[0].plot(t.gamma, t[f'rmse_{s}'], color=color, ls=ls, marker='o', label=f'{model} {s}')
            axes[1].plot(t.gamma, t[f'cov90_{s}'], color=color, ls=ls, marker='o', label=f'{model} {s}')
    axes[0].set(xscale='log', xlabel='Gamma', ylabel='endpoint RMSE (obs.-std.)', title='IHDP: PI-FM vs MSM exact / KL numerical ref.')
    axes[1].set(xscale='log', xlabel='Gamma', ylabel='90% posterior coverage', ylim=(0, 1), title='Calibration')
    axes[1].axhline(0.9, color='grey', lw=0.8, ls=':')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXPORT / 'ihdp_rmse_coverage_by_gamma.png', dpi=130)
    for name in ('msm_endpoint', 'kl_endpoint', 'msm_truth_by_kind', 'kl_truth_by_kind', 'comparison_summary', 'neuralcsa_realistic'):
        if name in tables:
            print(name); print(tables[name].round(3).to_string()[:3000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'pfn', 'cohort', 'neuralcsa', 'validate-kl', 'report'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'report': report()
    else: {'build': build, 'pfn': pfn, 'cohort': cohort, 'neuralcsa': neuralcsa, 'validate-kl': validate_kl}[args.stage](args.smoke)


if __name__ == '__main__':
    main()
