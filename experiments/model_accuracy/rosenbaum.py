"""Rosenbaum references, PI-FM and NeuralCSA evaluation and figures."""

from __future__ import annotations

from experiments.shared.paths import REPORTS
from experiments.shared.evaluation import ROSENBAUM_CHECKPOINT
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from experiments.shared.paths import RUNS as RUN_ROOT, OBSERVED_DATA
import argparse
import numpy as np
import pandas as pd
from experiments.shared.runtime import DATA, Run, completed, dataset_root, read, sha
from experiments.shared.msm_reference import msm_reference_from_samples
from experiments.comparisons.benchmark import capo_to_cate
from experiments.comparisons.benchmark import load_tasks as _load_tasks
import experiments.comparisons.benchmark as ev
import experiments.label_accuracy.rosenbaum as ros
from dataclasses import asdict, replace
import time
from experiments.comparisons.benchmark import load_tasks as benchmark_tasks
import shutil
import matplotlib.pyplot as plt

# Shared observed-only benchmark contexts, sharp references, and separate endpoint heads.
# References use 16,384 draws per query and retain Ros(G) subset MSM(G) subset Ros(G^2).



def load_tasks(smoke=False, cohorts=('synthetic', 'acic')):
    """Shared cohort ('synthetic', 'acic') or the IHDP cohort set ('ihdp'); never mixed."""
    cohorts = tuple(cohorts)
    if 'ihdp' in cohorts:
        assert cohorts == ('ihdp',), 'IHDP tasks live in a separate cohort set'
        return _load_tasks(smoke=smoke, cohort_set='ihdp')
    return _load_tasks(smoke=smoke, cohorts=cohorts)

PACKAGE = 'e15_rosenbaum_eval'
ROS_GAMMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 9.0, 10.0, 25.0)
REF_RUN = 'references'
BANDS = [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0)]
DRAWS = 16384

def draws_for(task, rows, arm, pos, device, artifacts, legacy, mech_cache):
    """Observed-standardized outcome draws of Y | x_row, A = arm: (len(rows), DRAWS)."""
    import torch
    if task.meta['cohort'] == 'synthetic':
        i = task.meta['dataset_id']
        outcome = legacy.load_outcome_map(artifacts[i], device)
        xcols = task.xcols
        x = task.context.set_index('row').loc[rows, xcols].to_numpy()
        latent = legacy.seeded_normal((len(rows), DRAWS), 20261108 + 1000 * pos + 10 * arm + i, device)
        with torch.no_grad():
            return outcome.evaluate(torch.tensor(x, dtype=torch.float32, device=device),
                                    torch.full((len(rows),), float(arm), device=device), latent).cpu().numpy()
    if task.meta['cohort'] == 'ihdp':                  # e05b_ihdp exact law (quantile midpoints)

        import experiments.applications.ihdp as ihdp_dgp
        if 'ihdp' not in mech_cache:
            mech_cache['ihdp'] = ihdp_dgp.dgp.mechanism(ihdp_dgp.load_covariates())
        build = dataset_root('e05b_ihdp')
        scaling = read(build / f"rep{task.meta['rep']}" / 'scaling.json')
        levels = DRAWS if len(rows) <= 256 else 4096
        quant = ihdp_dgp.exact_quantiles(mech_cache['ihdp'], np.asarray(rows), arm, levels=levels)
        return (quant - scaling['y_mean']) / scaling['y_std']
    import experiments.applications.acic as dgp
    variant, rep = task.meta['variant'], task.meta['rep']
    if variant not in mech_cache:
        mech_cache[variant] = dgp.mechanism(dgp.encode(pd.read_parquet(DATA / 'acic2016' / 'input_2016.parquet'), variant))
    e5 = dataset_root('e05_acic')
    scaling = read(e5 / variant / f'rep{rep}' / 'scaling.json')
    raw = dgp.observational_samples(mech_cache[variant], np.asarray(rows), arm, DRAWS, seed=7000 + 10 * rep + arm)
    return (raw - scaling['y_mean']) / scaling['y_std']

def build_references(smoke, cohorts=('synthetic', 'acic')):
    import torch
    from setting_standard.causaldataset import discover_causal_artifacts
    run = Run(PACKAGE, ref_run_name(smoke, cohorts), smoke=smoke, gpu=True, threads=4, gammas=ROS_GAMMAS, draws=DRAWS, cohorts=list(cohorts))
    import experiments.shared.msm_reference as legacy
    artifacts = {a['dataset_id']: a for a in discover_causal_artifacts(str(OBSERVED_DATA / 'fresh_observed'))}
    device = torch.device('cuda:0')
    tasks = load_tasks(smoke=smoke, cohorts=tuple(cohorts))
    capo, apo, mech_cache = [], [], {}
    pairs = [(ROS_GAMMAS.index(g), ROS_GAMMAS.index(g * g)) for g in ROS_GAMMAS if g * g in ROS_GAMMAS and g > 1]
    try:
        for pos, t in enumerate(tasks):
            q = t.queries.drop_duplicates(['query_id', 'a_query'])[['query_id', 'row', 'a_query', 'pi_query', 'true_capo']]
            for arm in (0, 1):
                qa = q[q.a_query.eq(arm)].sort_values('query_id')
                ctx_pi = t.context.propensity.to_numpy() if arm == 1 else 1 - t.context.propensity.to_numpy()
                for kind, rows, pi in (('query', qa.row.to_numpy(), qa.pi_query.to_numpy()),
                                       ('context', t.context.row.to_numpy(), ctx_pi)):
                    d = draws_for(t, rows, arm, pos, device, artifacts, legacy, mech_cache)
                    lo, up = ros.rosenbaum_capo_from_draws(d, pi, ROS_GAMMAS)
                    mlo, mup = msm_reference_from_samples(d, pi, np.array(ROS_GAMMAS))
                    assert np.all(up <= mup + 1e-9) and np.all(lo >= mlo - 1e-9), 'Ros(G) subset MSM(G) violated'
                    for i_m, i_r in pairs:                      # MSM(G) subset Ros(G^2), same draws
                        assert np.all(mup[:, i_m] <= up[:, i_r] + 1e-9) and np.all(mlo[:, i_m] >= lo[:, i_r] - 1e-9), 'MSM(G) subset Ros(G^2) violated'
                    if kind == 'query':
                        for j, g in enumerate(ROS_GAMMAS):
                            capo.append(pd.DataFrame(dict(task_id=t.task_id, query_id=qa.query_id.to_numpy(), a_query=arm,
                                                          gamma=g, ref_lower=lo[:, j], ref_upper=up[:, j], msm_lower=mlo[:, j],
                                                          msm_upper=mup[:, j], true=qa.true_capo.to_numpy(), pi_query=pi)))
                    else:
                        for j, g in enumerate(ROS_GAMMAS):
                            apo.append(dict(task_id=t.task_id, a_query=arm, gamma=g, ref_lower=lo[:, j].mean(),
                                            ref_upper=up[:, j].mean()))
            print(t.task_id, flush=True)
        capo = pd.concat(capo, ignore_index=True)
        capo.to_parquet(run.path / 'capo_reference.parquet', index=False)
        pd.DataFrame(apo).to_parquet(run.path / 'apo_reference.parquet', index=False)
        run.done(tasks=len(tasks), rows=len(capo), nesting_pairs=len(pairs))
    except BaseException:
        run.done('failed')
        raise

def predict(smoke, checkpoint, name, snapshot=False, cohorts=('synthetic', 'acic')):
    import shutil
    import time
    import torch
    from experiments.shared.evaluation import gmm_arrays, gmm_quantiles, load_pifm, pifm_predict_frame
    checkpoint = Path(checkpoint).resolve()
    run = Run(PACKAGE, f'predict_{name}', smoke=smoke, gpu=True, threads=4, source_checkpoint=str(checkpoint),
              gammas=ROS_GAMMAS, snapshot=snapshot, cohorts=list(cohorts))
    if snapshot:                                   # freeze best_model.pth while training may overwrite it
        frozen = run.path / 'checkpoint.pth'
        shutil.copy2(checkpoint, frozen)
        checkpoint = frozen
    device = torch.device('cuda:0')
    model, raw = load_pifm(checkpoint, device)
    gamma_transform = raw.get('gamma_transform', 'none')     # must match training (training_standard --gamma_transform)
    assert gamma_transform in ('none', 'log'), gamma_transform
    run.manifest['config'].update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), gamma_transform=gamma_transform,
                                  checkpoint_epoch=raw.get('epoch'), checkpoint_val_loss=raw.get('best_val_loss', raw.get('val_loss')))
    run.save()
    capo, apo, timing = [], [], []
    try:
        for t in load_tasks(smoke=smoke, cohorts=tuple(cohorts)):
            base = t.queries.drop_duplicates(['query_id', 'a_query'])[['query_id', 'a_query'] + t.xcols]
            ctx = t.context[['row'] + t.xcols]
            frames = [base.assign(gamma=g) for g in ROS_GAMMAS] + \
                     [ctx.assign(a_query=a, gamma=g, query_id=-1) for a in (0, 1) for g in ROS_GAMMAS]
            query = pd.concat(frames, ignore_index=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            model_query = query.assign(gamma=np.log(query.gamma)) if gamma_transform == 'log' else query
            pred = pifm_predict_frame(model, t.context, t.xcols, model_query, device, chunk_size=4096)
            pred['gamma'] = query.gamma.to_numpy()                 # report on the raw Gamma scale
            torch.cuda.synchronize()
            timing.append(dict(task_id=t.task_id, predictions=len(query), seconds=time.perf_counter() - start))
            for side in ('lower', 'upper'):
                pi, mu, sigma = gmm_arrays(pred, side)
                qs = gmm_quantiles(pi, mu, sigma, [0.05, 0.95])
                pred[f'{side}_q05'], pred[f'{side}_q95'] = qs[:, 0], qs[:, 1]
            on_q = pred.query_id.ge(0)
            p = pred[on_q]
            capo.append(pd.DataFrame(dict(task_id=t.task_id, query_id=p.query_id, a_query=p.a_query, gamma=p.gamma,
                                          lower=p.lower_mean, upper=p.upper_mean, lower_q05=p.lower_q05, lower_q95=p.lower_q95,
                                          upper_q05=p.upper_q05, upper_q95=p.upper_q95)))
            apo.append(pred[~on_q].groupby(['a_query', 'gamma'], as_index=False)[['lower_mean', 'upper_mean']].mean()
                       .rename(columns={'lower_mean': 'lower', 'upper_mean': 'upper'}).assign(task_id=t.task_id))
        capo = pd.concat(capo, ignore_index=True)
        capo.to_parquet(run.path / 'capo.parquet', index=False)
        capo_to_cate(capo[['task_id', 'query_id', 'a_query', 'gamma', 'lower', 'upper', 'lower_q05', 'upper_q95']]).to_parquet(run.path / 'cate.parquet', index=False)
        pd.concat(apo, ignore_index=True).to_parquet(run.path / 'apo.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(rows=len(capo), median_seconds_per_task=float(pd.DataFrame(timing).seconds.median()))
    except BaseException:
        run.done('failed')
        raise

# Synthetic tasks 1–8 inform the Rosenbaum lambda grid; tasks 9–24 are held out.
DEV_TASKS = {f'syn_{i}' for i in range(1, 9)}

def split_group(task_id, group):
    if group != 'synthetic':
        return group
    return 'synthetic_dev8' if task_id in DEV_TASKS else 'synthetic_test16'

def with_split(rows):
    """Primary grouping (dev/test split of synthetic) plus a pooled copy labelled synthetic_all24 (secondary)."""
    primary = rows.assign(group=[split_group(t, g) for t, g in zip(rows.task_id, rows.group)])
    pooled = rows[rows.group.eq('synthetic')].assign(group='synthetic_all24_secondary')
    return pd.concat([primary, pooled], ignore_index=True)

def band_of(gamma):
    for lo, hi in BANDS:
        if (gamma > lo or lo == 1.0) and gamma <= hi:
            return f'[{lo:g},{hi:g}]' if lo == 1.0 else f'({lo:g},{hi:g}]'
    return 'other'

def head_calibration(rows):
    f = rows.copy()
    f['cov90_lower'] = (f.lower_q05 <= f.ref_lower) & (f.ref_lower <= f.lower_q95)
    f['cov90_upper'] = (f.upper_q05 <= f.ref_upper) & (f.ref_upper <= f.upper_q95)
    f['outward_joint'] = (f.lower_q05 <= f.ref_lower) & (f.upper_q95 >= f.ref_upper)
    f['width_ratio'] = (f.upper - f.lower) / (f.ref_upper - f.ref_lower).where(lambda w: w > 1e-9)
    return f

def ref_run_name(smoke, cohorts):
    return REF_RUN + ('_smoke' if smoke else '') + ('' if tuple(cohorts) == ('synthetic', 'acic') else '_' + '_'.join(cohorts))

def report(smoke, name, neuralcsa=None, cohorts=('synthetic', 'acic'), tag=''):
    refs = completed(PACKAGE, ref_run_name(smoke, cohorts))
    preds = completed(PACKAGE, f'predict_{name}')
    assert refs is not None and preds is not None
    run = Run(PACKAGE, f'report_{name}{tag}', smoke=smoke, gpu=False, threads=4, references=str(refs), predictions=str(preds),
              neuralcsa=neuralcsa)
    reps = 50 if smoke else 2000
    capo_ref = pd.read_parquet(refs / 'capo_reference.parquet')
    keys = ['task_id', 'query_id', 'gamma']
    one, zero = capo_ref[capo_ref.a_query.eq(1)].set_index(keys), capo_ref[capo_ref.a_query.eq(0)].set_index(keys)
    cate_ref = pd.DataFrame(dict(ref_lower=one.ref_lower - zero.ref_upper, ref_upper=one.ref_upper - zero.ref_lower,
                                 true=one['true'] - zero['true'])).reset_index()
    tables = {}
    capo_pred = pd.read_parquet(preds / 'capo.parquet')
    capo_rows = ev.row_metrics(ev.join(capo_pred, capo_ref.drop(columns=['msm_lower', 'msm_upper', 'pi_query']), keys + ['a_query']))
    capo_rows['band'] = capo_rows.gamma.map(band_of)
    capo_rows = with_split(capo_rows)
    cal = head_calibration(capo_rows)
    tables['capo_by_group'] = ev.summarize(capo_rows, ('group',), replicates=reps)
    tables['capo_by_group_band'] = ev.summarize(capo_rows, ('group', 'band'), replicates=reps)
    tables['capo_by_group_gamma'] = ev.summarize(capo_rows, ('group', 'gamma'), replicates=max(200, reps // 10))
    tables['capo_calibration_by_group_band'] = cal.groupby(['group', 'band'], as_index=False).agg(
        rows=('task_id', 'size'), cov90_lower=('cov90_lower', 'mean'), cov90_upper=('cov90_upper', 'mean'),
        outward_joint=('outward_joint', 'mean'), width_ratio_median=('width_ratio', 'median'))
    cate_rows = ev.row_metrics(ev.join(pd.read_parquet(preds / 'cate.parquet'), cate_ref, keys))
    cate_rows['band'] = cate_rows.gamma.map(band_of)
    cate_rows = with_split(cate_rows)
    tables['cate_by_group_band'] = ev.summarize(cate_rows, ('group', 'band'), replicates=reps)
    apo = pd.read_parquet(preds / 'apo.parquet').merge(pd.read_parquet(refs / 'apo_reference.parquet'),
                                                       on=['task_id', 'a_query', 'gamma'], validate='one_to_one')
    apo['abs_err_lower'] = (apo.lower - apo.ref_lower).abs()
    apo['abs_err_upper'] = (apo.upper - apo.ref_upper).abs()
    tables['apo'] = apo
    tables['timing'] = pd.read_csv(preds / 'timing.csv')
    if neuralcsa:
        nc = pd.read_parquet(completed(PACKAGE, neuralcsa) / 'capo.parquet')
        known = nc.task_id.isin(capo_ref.task_id)
        assert smoke or known.all(), 'NeuralCSA tasks without references'
        nc = nc[known]                                                  # smoke references cover the smoke tasks only
        k = ['task_id', 'query_id', 'a_query', 'gamma']
        sub = capo_rows.merge(nc[k].drop_duplicates(), on=k)            # PI-FM on the NeuralCSA subset
        comp = [ev.summarize(sub, ('group',), replicates=reps).assign(method='pifm')]
        for variant, part in nc.groupby('variant'):
            rows = with_split(ev.row_metrics(ev.join(part[k + ['lower', 'upper']], capo_ref.drop(columns=['msm_lower', 'msm_upper', 'pi_query']), k)))
            comp.append(ev.summarize(rows, ('group',), replicates=reps).assign(method=f'neuralcsa_{variant}'))
            feas = part.assign(group=[split_group(t, ev.group_of(t)) for t in part.task_id]).groupby('group').apply(
                lambda f: pd.Series(dict(infeasible=float(np.mean(np.r_[f.logviol_lower, f.logviol_upper] > np.log(1.02))))),
                include_groups=False).reset_index().assign(method=f'neuralcsa_{variant}')
            tables[f'neuralcsa_{variant}_feasibility'] = feas
        tables['neuralcsa_comparison'] = pd.concat(comp, ignore_index=True)
        tables['neuralcsa_timing'] = pd.read_csv(completed(PACKAGE, neuralcsa) / 'timing.csv')
    for key, table in tables.items():
        table.to_csv(run.path / f'{key}.csv', index=False)
    cols = ['group', 'band', 'rows', 'rmse_lower', 'rmse_upper', 'covers_ref', 'width', 'ref_width']
    print(tables['capo_by_group_band'][[c for c in cols if c in tables['capo_by_group_band']]].round(3).to_string())
    print(tables['capo_calibration_by_group_band'].round(3).to_string())
    if neuralcsa:
        print(tables['neuralcsa_comparison'][['method', 'group', 'rows', 'rmse_lower', 'rmse_upper', 'width', 'ref_width']].round(3).to_string())
    run.done(tables=list(tables))

# NeuralCSA uses separate cold starts for each query, Gamma, and head. Its outcome
# shift enters the compatible mixture pi*phi + (1-pi)*nu. Rosenbaum constraints
# use log(Delta)-log(Gamma); final feasibility uses dense support and 2% tolerance.
# Oracle controls use saved synthetic maps or exact conditional quantiles. IHDP
# additionally compares the same fitted outcome flow with true and fitted propensity.



NCSA_GAMMAS = (2.0, 4.0, 9.0, 25.0)       # subset of the evaluation grid; pairs with MSM {1.41, 2, 3, 5}
ACIC_QUERY_ROWS = 16                      # first 16 held-out query rows of each selected ACIC task
ACIC_TASKS = ('acic_d10_rep0', 'acic_d10_rep1', 'acic_full_rep0', 'acic_full_rep1')
IHDP_TASKS = ('ihdp_rep0', 'ihdp_rep1')   # the two IHDP replications of the MSM/KL realistic comparison
IHDP_QUERY_ROWS = 8                       # query_id < 8, as in e05b_ihdp/neuralcsa_realistic.py
LOG_TOL = float(np.log(1.02))
ORACLE_DRAWS = 16384

def baseline():
    import experiments.comparisons.neuralcsa as solver
    return solver

def dense_rosenbaum_gamma(spline_params, config, device):
    """Delta_dense = sup r / inf r of the spline over a dense base grid covering its support."""
    import torch
    from setting_standard.frontier_optimization import batched_spline_forward, standard_normal_logprob
    flow = config.flow_config()
    edge = max(8.0, flow.tail_bound + 1.0)
    base = torch.linspace(-edge, edge, config.msm_dense_grid_points, device=device)[None].expand(spline_params.shape[0], -1)
    nu, log_nu = batched_spline_forward(base.unsqueeze(0), spline_params.unsqueeze(0), flow)
    log_r = log_nu.squeeze(0) - standard_normal_logprob(nu.squeeze(0))
    hi = log_r.amax(dim=1).clamp(min=0.0)
    lo = log_r.amin(dim=1).clamp(max=0.0)
    return torch.exp(hi - lo), torch.exp(hi), torch.exp(-lo)

def solve_rosenbaum(outcome_map, x_query, a_query, pi_query, gamma_target, bound_type, config, device):
    """Cold-start, independent prescribed-Gamma Rosenbaum solves (batched for throughput only)."""
    import torch
    from torch import nn
    B = baseline()
    assert bound_type in ('upper', 'lower')
    n = x_query.shape[0]
    a_query = a_query.reshape(n).to(device=device, dtype=torch.float32)
    pi_query = pi_query.reshape(n).to(device=device, dtype=torch.float32)
    gamma_target = gamma_target.reshape(n).to(device=device, dtype=torch.float32)
    x_query = x_query.to(device=device, dtype=torch.float32)
    assert torch.all((pi_query > 0) & (pi_query < 1)) and torch.all(gamma_target >= 1.0)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    flow = config.flow_config()
    log_target = torch.log(gamma_target)
    with torch.no_grad():
        q0 = outcome_map.evaluate(x_query, a_query, B._seeded_normal((n, config.q0_mc_samples), config.seed + 1, device)).mean(dim=1)
    params = nn.Parameter(B.identity_spline_params(n, config, device))
    optimizer = torch.optim.Adam([params], lr=config.learning_rate)
    multiplier = torch.zeros(n, device=device)
    sign = 1.0 if bound_type == 'upper' else -1.0
    started = time.perf_counter()

    def evaluate(seed):
        base = B._seeded_normal((n, config.eval_mc_samples), seed, device)
        theta, _ = B._theta_and_gamma(params, outcome_map, x_query, a_query, pi_query, q0, 'rosenbaum', base, flow)
        dense, _, _ = dense_rosenbaum_gamma(params, config, device)
        return theta, dense

    with torch.no_grad():
        theta0, dense0 = evaluate(config.seed + 500_002)
    best_params = params.detach().clone()
    best_signed = sign * theta0
    best_violation = torch.relu(torch.log(dense0) - log_target)
    for k, weight in enumerate(config.penalty_weights):
        for _ in range(config.steps_per_penalty):
            base = torch.randn(n, config.train_mc_samples, device=device)
            theta, gamma = B._theta_and_gamma(params, outcome_map, x_query, a_query, pi_query, q0, 'rosenbaum', base, flow)
            constraint = torch.log(gamma) - log_target
            shifted = torch.relu(multiplier + weight * constraint)
            loss = (-sign * theta + (shifted.square() - multiplier.square()) / (2.0 * weight)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            B._clip_spline_grad_per_instance_(params, max_norm=5.0)
            optimizer.step()
        with torch.no_grad():
            theta_k, dense_k = evaluate(config.seed + 1000 + k)
            multiplier = torch.relu(multiplier + weight * (torch.log(dense_k) - log_target))
            violation = torch.relu(torch.log(dense_k) - log_target)
            feasible, best_feasible = violation <= LOG_TOL, best_violation <= LOG_TOL
            signed = sign * theta_k
            improves = (feasible & (~best_feasible | (signed > best_signed))) | (~feasible & ~best_feasible & (violation < best_violation))
            best_params[improves] = params.detach()[improves]
            best_signed[improves] = signed[improves]
            best_violation[improves] = violation[improves]
    with torch.no_grad():
        params.copy_(best_params)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        theta, dense = evaluate(config.seed + 999_999)
    return dict(theta=theta.cpu().numpy(), gamma_dense=dense.cpu().numpy(),
                log_violation=torch.relu(torch.log(dense) - log_target).cpu().numpy(),
                q0=q0.cpu().numpy(), seconds=elapsed)

class QuantileTableMap:
    """Exact conditional quantile map u -> F^{-1}_{Y|x,a}(Phi(u)) per instance (rows in instance order)."""

    def __init__(self, sorted_draws, device):
        import torch
        self.table = torch.tensor(sorted_draws, dtype=torch.float32, device=device)

    def evaluate(self, x_query, a_query, latent):
        import torch
        m, k = self.table.shape
        assert latent.shape[0] == m
        pos = (torch.special.ndtr(latent.double()) * k - 0.5).clamp(0.0, k - 1.0)
        i0 = pos.floor().long().clamp(max=k - 2)
        w = (pos - i0).float()
        lo = torch.gather(self.table, 1, i0)
        hi = torch.gather(self.table, 1, i0 + 1)
        return lo + w * (hi - lo)

def ihdp_tables(task, inst, mech_cache):
    """Sorted observed-standardized exact quantile midpoints of Y | x_row, A=arm per instance (instance order).

    Identical construction to run.draws_for (IHDP branch), hence to the frozen references_ihdp.
    """

    import experiments.applications.ihdp as ihdp_dgp
    if 'ihdp' not in mech_cache:
        mech_cache['ihdp'] = ihdp_dgp.dgp.mechanism(ihdp_dgp.load_covariates())
    build = dataset_root('e05b_ihdp')
    scaling = read(build / f"rep{task.meta['rep']}" / 'scaling.json')
    tables = np.empty((len(inst), ORACLE_DRAWS))
    for arm in (0, 1):
        m = inst.a_query.to_numpy() == arm
        uniq, inv = np.unique(inst.row.to_numpy()[m], return_inverse=True)
        q = ihdp_dgp.exact_quantiles(mech_cache['ihdp'], uniq, arm, levels=ORACLE_DRAWS)
        tables[m] = ((q - scaling['y_mean']) / scaling['y_std'])[inv]
    assert np.all(np.diff(tables, axis=1) >= -1e-12)
    return tables

def check_ihdp_references(task, inst, lo, up):
    """The exact-quantile sharp bounds must equal the frozen references_ihdp values on these rows."""
    ref = pd.read_parquet(completed(PACKAGE, 'references_ihdp') / 'capo_reference.parquet')
    k = ['query_id', 'a_query', 'gamma']
    m = inst[k].assign(lo=lo, up=up).merge(ref[ref.task_id.eq(task.task_id)][k + ['ref_lower', 'ref_upper']],
                                            on=k, validate='one_to_one')
    assert len(m) == len(inst)
    diff = float(np.max(np.abs(np.r_[m.lo - m.ref_lower, m.up - m.ref_upper])))
    assert diff < 1e-6, ('IHDP references differ from references_ihdp', diff)
    return diff

def instances(task, gammas, rows_limit):
    q = task.queries.drop_duplicates(['query_id', 'a_query'])
    if rows_limit is not None:
        q = q[q.query_id.isin(sorted(q.query_id.unique())[:rows_limit])]
    q = q.sort_values(['query_id', 'a_query'])
    out = pd.concat([q.assign(gamma=g) for g in gammas], ignore_index=True)
    return out[['query_id', 'row', 'a_query', 'pi_query'] + task.xcols + ['gamma']]

def oracle_map(task, inst, device, pos, artifacts, legacy, mech_cache):
    if task.meta['cohort'] == 'synthetic':
        return legacy.load_outcome_map(artifacts[task.meta['dataset_id']], device)
    import experiments.applications.acic as dgp
    variant, rep = task.meta['variant'], task.meta['rep']
    if variant not in mech_cache:
        mech_cache[variant] = dgp.mechanism(dgp.encode(pd.read_parquet(DATA / 'acic2016' / 'input_2016.parquet'), variant))
    e5 = dataset_root('e05_acic')
    scaling = read(e5 / variant / f'rep{rep}' / 'scaling.json')
    tables = []
    for row, arm in zip(inst.row.to_numpy(), inst.a_query.to_numpy()):
        raw = dgp.observational_samples(mech_cache[variant], np.array([row]), int(arm), ORACLE_DRAWS, seed=9000 + 10 * rep + int(arm))
        tables.append(np.sort((raw[0] - scaling['y_mean']) / scaling['y_std']))
    return QuantileTableMap(np.stack(tables), device)

def sharp(task, inst, device, pos, artifacts, legacy, mech_cache, omap):
    """Sharp Rosenbaum CAPO for each instance from 16,384 draws of the oracle map."""
    import torch
    lat = legacy.seeded_normal((len(inst), ORACLE_DRAWS), 4242 + pos, device)
    x = torch.tensor(inst[task.xcols].to_numpy(), dtype=torch.float32, device=device)
    a = torch.tensor(inst.a_query.to_numpy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        d = omap.evaluate(x, a, lat).cpu().numpy()
    lo = np.empty(len(inst)); up = np.empty(len(inst))
    for g in inst.gamma.unique():
        m = inst.gamma.to_numpy() == g
        l, u = ros.rosenbaum_capo_from_draws(d[m], inst.pi_query.to_numpy()[m], (g,))
        lo[m], up[m] = l[:, 0], u[:, 0]
    return lo, up

def configs(smoke):
    B = baseline()
    s1 = B.Stage1Config(seed=20261108)
    s2 = B.Stage2Config(seed=20261118)
    if smoke:
        s1 = replace(s1, max_steps=150)
        s2 = replace(s2, steps_per_penalty=30, train_mc_samples=128, eval_mc_samples=1024, msm_dense_grid_points=1025)
    return s1, s2

def run_neuralcsa(stage, smoke):
    import torch
    from setting_standard.causaldataset import discover_causal_artifacts
    B = baseline()
    s1, s2 = configs(smoke)
    name = f'neuralcsa_{stage}' + ('_smoke' if smoke else '')
    run = Run(PACKAGE, name, smoke=smoke, gpu=True, threads=4, gammas=NCSA_GAMMAS, acic_tasks=ACIC_TASKS,
              acic_query_rows=ACIC_QUERY_ROWS, stage1=asdict(s1), stage2=asdict(s2), log_tolerance=LOG_TOL,
              ihdp_tasks=IHDP_TASKS, ihdp_query_rows=IHDP_QUERY_ROWS,
              propensity={'oracle_stage': 'true', 'full': 'true',
                          'full_estimated_pi': 'logistic regression on context, clipped [0.01, 0.99]'})
    import experiments.shared.msm_reference as legacy
    artifacts = {a['dataset_id']: a for a in discover_causal_artifacts(str(OBSERVED_DATA / 'fresh_observed'))}
    device = torch.device('cuda:0')
    tasks = benchmark_tasks()
    if stage == 'ihdp':
        tasks = [t for t in benchmark_tasks(cohort_set='ihdp') if t.task_id in IHDP_TASKS][: (1 if smoke else 2)]
        assert smoke or [t.task_id for t in tasks] == list(IHDP_TASKS)
        variants = ('oracle_stage', 'full', 'full_estimated_pi')
    elif stage == 'validate':
        tasks = [t for t in tasks if t.task_id in ('syn_1', 'syn_2', 'acic_d10_rep0')][: (1 if smoke else 3)]
        variants = ('oracle_stage',)
    else:
        keep = [t for t in tasks if t.meta['cohort'] == 'synthetic'] + [t for t in tasks if t.task_id in ACIC_TASKS]
        tasks = keep[:2] if smoke else keep
        variants = ('oracle_stage', 'full')
    rows, timing, mech_cache = [], [], {}
    try:
        for pos, t in enumerate(tasks):
            ihdp = t.meta['cohort'] == 'ihdp'
            limit = IHDP_QUERY_ROWS if ihdp else (None if t.meta['cohort'] == 'synthetic' else ACIC_QUERY_ROWS)
            inst = instances(t, NCSA_GAMMAS, limit)
            if ihdp:
                tables = ihdp_tables(t, inst, mech_cache)
                omap = QuantileTableMap(tables, device)
                ref_lo, ref_up = np.empty(len(inst)), np.empty(len(inst))
                for gv in inst.gamma.unique():
                    mg = inst.gamma.to_numpy() == gv
                    l, u = ros.rosenbaum_capo_from_draws(tables[mg], inst.pi_query.to_numpy()[mg], (gv,))
                    ref_lo[mg], ref_up[mg] = l[:, 0], u[:, 0]
                run.manifest.setdefault('reference_check_max_abs_diff', {})[t.task_id] = check_ihdp_references(t, inst, ref_lo, ref_up)
            else:
                omap = oracle_map(t, inst, device, pos, artifacts, legacy, mech_cache)
                ref_lo, ref_up = sharp(t, inst, device, pos, artifacts, legacy, mech_cache, omap)
            x = torch.tensor(inst[t.xcols].to_numpy(), dtype=torch.float32)
            a = torch.tensor(inst.a_query.to_numpy(), dtype=torch.float32)
            pi = torch.tensor(inst.pi_query.to_numpy(), dtype=torch.float32)
            g = torch.tensor(inst.gamma.to_numpy(), dtype=torch.float32)
            fitted = None                                        # one Stage-1 fit shared by both fitted variants
            for variant in variants:
                stage1_seconds = 0.0
                pi_v = pi
                if variant == 'full_estimated_pi':
                    from sklearn.linear_model import LogisticRegression
                    X, A, _ = t.observed()
                    e_hat = LogisticRegression(max_iter=2000).fit(X, A).predict_proba(inst[t.xcols].to_numpy())[:, 1]
                    pi_v = torch.tensor(np.clip(np.where(inst.a_query.to_numpy() == 1, e_hat, 1 - e_hat), 0.01, 0.99),
                                        dtype=torch.float32)
                if variant == 'oracle_stage':
                    model = omap
                elif fitted is not None:
                    model, stage1_seconds = fitted
                else:
                    X, A, Y = t.observed()
                    torch.cuda.synchronize()
                    s = time.perf_counter()
                    fit = B.fit_stage1_outcome_flow(torch.tensor(X, dtype=torch.float32), torch.tensor(A, dtype=torch.float32),
                                                   torch.tensor(Y, dtype=torch.float32), replace(s1, seed=s1.seed + pos), device)
                    torch.cuda.synchronize()
                    stage1_seconds = time.perf_counter() - s
                    model = fit.model
                    model.eval()
                    model.requires_grad_(False)
                    fitted = (model, stage1_seconds)
                out = {}
                for head in ('upper', 'lower'):
                    cfg = replace(s2, seed=s2.seed + 1000 * pos + (0 if head == 'upper' else 10))
                    out[head] = solve_rosenbaum(model, x, a, pi_v, g, head, cfg, device)
                rows.append(inst[['query_id', 'a_query', 'gamma', 'pi_query']].assign(pi_used=pi_v.numpy(),
                    task_id=t.task_id, variant=variant, lower=out['lower']['theta'], upper=out['upper']['theta'],
                    gamma_dense_lower=out['lower']['gamma_dense'], gamma_dense_upper=out['upper']['gamma_dense'],
                    logviol_lower=out['lower']['log_violation'], logviol_upper=out['upper']['log_violation'],
                    sharp_lower=ref_lo, sharp_upper=ref_up))
                timing.append(dict(task_id=t.task_id, variant=variant, instances=len(inst), stage1_seconds=stage1_seconds,
                                   stage2_seconds=out['upper']['seconds'] + out['lower']['seconds']))
                r = rows[-1]
                err = np.r_[r.upper - r.sharp_upper, r.lower - r.sharp_lower]
                print(f"{t.task_id} {variant}: rmse_vs_sharp={np.sqrt(np.mean(err ** 2)):.4f} "
                      f"infeasible={np.mean(np.r_[r.logviol_upper, r.logviol_lower] > LOG_TOL):.3f} "
                      f"s1={stage1_seconds:.1f}s s2={timing[-1]['stage2_seconds']:.1f}s", flush=True)
        frame = pd.concat(rows, ignore_index=True)
        frame.to_parquet(run.path / 'capo.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        summary = {}
        for variant, part in frame.groupby('variant'):
            e = np.r_[part.upper - part.sharp_upper, part.lower - part.sharp_lower]
            # signed outward error: > 0 means the returned bound is wider than sharp
            outward = np.r_[part.upper - part.sharp_upper, part.sharp_lower - part.lower]
            summary[variant] = dict(rmse_vs_sharp=float(np.sqrt(np.mean(e ** 2))), median_outward=float(np.median(outward)),
                                    infeasible_rate=float(np.mean(np.r_[part.logviol_upper, part.logviol_lower] > LOG_TOL)))
        print(summary, flush=True)
        run.done(**{f'{k}_{m}': v for k, d in summary.items() for m, v in d.items()})
    except BaseException:
        run.done('failed')
        raise


REPORT_RUNS = RUN_ROOT / 'e15_rosenbaum_eval'
EXPORT = REPORTS / 'rosenbaum_model'
IHDP_EXPORT = EXPORT / 'ihdp_neuralcsa'



def group(task_id):
    if task_id.startswith('syn_'):
        return 'synthetic_dev8' if task_id in DEV_TASKS else 'synthetic_test16'
    if task_id.startswith('ihdp'):
        return 'ihdp'
    return 'acic_full' if '_full_' in task_id else 'acic_d10'

def reference_summary():
    out = []
    for name in ('references', 'references_ihdp'):
        run = completed('e15_rosenbaum_eval', name)
        if run is None:
            continue
        r = pd.read_parquet(run / 'capo_reference.parquet')
        r['group'] = r.task_id.map(group)
        r = r[r.gamma > 1]
        w = r.ref_upper - r.ref_lower
        r['ratio'] = w / (r.msm_upper - r.msm_lower)
        r['true_in'] = np.where(r['true'].notna(), (r.ref_lower <= r['true']) & (r['true'] <= r.ref_upper), np.nan)
        t = r.groupby(['group', 'gamma']).agg(rows=('ratio', 'size'), ros_width=('ref_upper', lambda s: float((s - r.loc[s.index, 'ref_lower']).mean())),
                                              width_ratio_ros_over_msm_median=('ratio', 'median'),
                                              true_capo_in_sharp_ros=('true_in', 'mean')).reset_index()
        t.insert(0, 'run', name)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(EXPORT / 'references_summary.csv', index=False)
    return t

def ncsa(name):
    run = completed('e15_rosenbaum_eval', name)
    f = pd.read_parquet(run / 'capo.parquet')
    f['group'] = f.task_id.map(group)
    rows = []
    for keys, part in f.groupby(['variant', 'group', 'gamma']):
        err = np.r_[part.upper - part.sharp_upper, part.lower - part.sharp_lower]
        outward = np.r_[part.upper - part.sharp_upper, part.sharp_lower - part.lower]
        rows.append(dict(zip(('variant', 'group', 'gamma'), keys), rows=len(part), tasks=part.task_id.nunique(),
                         rmse_vs_sharp=float(np.sqrt(np.mean(err ** 2))), median_signed_outward=float(np.median(outward)),
                         p05_signed_outward=float(np.quantile(outward, 0.05)), p95_signed_outward=float(np.quantile(outward, 0.95)),
                         infeasible_rate=float(np.mean(np.r_[part.logviol_lower, part.logviol_upper] > LOG_TOL)),
                         attained_over_target_median=float(np.median(np.r_[part.gamma_dense_lower, part.gamma_dense_upper] / part.gamma.iloc[0]))))
    by_gamma = pd.DataFrame(rows)
    pooled = []
    for keys, part in f.groupby(['variant', 'group']):
        err = np.r_[part.upper - part.sharp_upper, part.lower - part.sharp_lower]
        pooled.append(dict(zip(('variant', 'group'), keys), rows=len(part), tasks=part.task_id.nunique(),
                           rmse_vs_sharp=float(np.sqrt(np.mean(err ** 2))),
                           median_signed_outward=float(np.median(np.r_[part.upper - part.sharp_upper, part.sharp_lower - part.lower])),
                           infeasible_rate=float(np.mean(np.r_[part.logviol_lower, part.logviol_upper] > LOG_TOL))))
    pooled = pd.DataFrame(pooled)
    timing = pd.read_csv(run / 'timing.csv')
    timing['group'] = timing.task_id.map(group)
    tsum = timing.groupby(['variant', 'group'], as_index=False)[['instances', 'stage1_seconds', 'stage2_seconds']].median()
    by_gamma.to_csv(EXPORT / f'{name}_by_gamma.csv', index=False)
    pooled.to_csv(EXPORT / f'{name}_by_group.csv', index=False)
    tsum.to_csv(EXPORT / f'{name}_timing_median.csv', index=False)
    return pooled, by_gamma, tsum


def report_ihdp():
    IHDP_EXPORT.mkdir(parents=True, exist_ok=True)
    for f in (REPORT_RUNS / 'report_ros_final_ihdp_ncsa').glob('neuralcsa_*.csv'):
        shutil.copy2(f, IHDP_EXPORT / f.name)
    k = ['task_id', 'query_id', 'a_query', 'gamma']
    ref = pd.read_parquet(REPORT_RUNS / 'references_ihdp' / 'capo_reference.parquet')[k + ['ref_lower', 'ref_upper']]
    nc = pd.read_parquet(REPORT_RUNS / 'neuralcsa_ihdp' / 'capo.parquet')
    pf = pd.read_parquet(REPORT_RUNS / 'predict_ros_final_ihdp' / 'capo.parquet')
    frames = [pf.merge(nc[k].drop_duplicates(), on=k)[k + ['lower', 'upper']].assign(method='pifm_frozen')]
    frames += [part[k + ['lower', 'upper']].assign(method=f'neuralcsa_{v}') for v, part in nc.groupby('variant')]
    rows = pd.concat(frames, ignore_index=True).merge(ref, on=k, validate='many_to_one')
    assert rows.groupby('method').size().nunique() == 1, 'methods not on identical rows'
    rows['e_lo'], rows['e_up'] = rows.lower - rows.ref_lower, rows.upper - rows.ref_upper
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    by_gamma = rows.groupby(['method', 'gamma']).apply(lambda f: pd.Series(dict(
        rows=len(f), rmse_lower=rmse(f.e_lo), rmse_upper=rmse(f.e_up), width=float((f.upper - f.lower).mean()),
        ref_width=float((f.ref_upper - f.ref_lower).mean()))), include_groups=False).reset_index()
    by_gamma.to_csv(IHDP_EXPORT / 'by_gamma.csv', index=False)
    t = pd.read_csv(REPORT_RUNS / 'neuralcsa_ihdp' / 'timing.csv')
    t.groupby('variant')[['stage1_seconds', 'stage2_seconds']].median().reset_index().to_csv(IHDP_EXPORT / 'timing_summary.csv', index=False)
    pt = pd.read_csv(REPORT_RUNS / 'predict_ros_final_ihdp' / 'timing.csv')
    pd.DataFrame([dict(method='pifm_frozen', median_seconds_per_task=float(pt.seconds.median()),
                       predictions_per_task=int(pt.predictions.median()))]).to_csv(IHDP_EXPORT / 'pifm_timing.csv', index=False)
    print(by_gamma.round(3).to_string())


PLOT_RUNS = RUN_ROOT / 'e15_rosenbaum_eval'
PLOT_GROUPS = ['synthetic_test16', 'acic_d10', 'acic_full', 'ihdp']
COLORS = {'synthetic_test16': '#1f4e79', 'acic_d10': '#2f855a', 'acic_full': '#c05621', 'ihdp': '#6b46c1'}

def plot(name):
    src = PLOT_RUNS / f'report_{name}'
    out = REPORTS / 'rosenbaum_model' / name
    out.mkdir(parents=True, exist_ok=True)
    shared_comparator = {'neuralcsa_oracle_stage_feasibility.csv',
                         'neuralcsa_full_feasibility.csv', 'neuralcsa_timing.csv'}
    for f in src.glob('*.csv'):
        shutil.copy2(f, out / f.name)
    ihdp = PLOT_RUNS / f'report_{name}_ihdp'
    frames = {k: [pd.read_csv(src / f'{k}.csv')] for k in ('capo_by_group_gamma', 'capo_calibration_by_group_band')}
    if ihdp.exists():
        for f in ihdp.glob('*.csv'):
            if f.name in shared_comparator | {'neuralcsa_full_estimated_pi_feasibility.csv'}:
                continue  # The matched IHDP comparator export owns these model-independent diagnostics.
            shutil.copy2(f, out / f'ihdp_{f.name}')
        for k in frames:
            if (ihdp / f'{k}.csv').exists():
                frames[k].append(pd.read_csv(ihdp / f'{k}.csv'))
    g = pd.concat(frames['capo_by_group_gamma'], ignore_index=True)
    cal = pd.concat(frames['capo_calibration_by_group_band'], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for grp in PLOT_GROUPS:
        p = g[g.group.eq(grp)].sort_values('gamma')
        if p.empty:
            continue
        axes[0].plot(p.gamma, 0.5 * (p.rmse_lower + p.rmse_upper), marker='o', color=COLORS[grp], label=grp)
        q = p[p.gamma > 1]                       # Gamma = 1: sharp width ~0 (point identification)
        axes[1].plot(q.gamma, q.width / q.ref_width, marker='o', color=COLORS[grp], label=grp)
    axes[0].set(xscale='log', xlabel='Rosenbaum Γ', ylabel='endpoint RMSE vs sharp reference', title='Accuracy')
    axes[1].axhline(1, color='grey', lw=0.8, ls=':')
    axes[1].set(xscale='log', xlabel='Rosenbaum Γ', ylabel='mean width / sharp width', title='Width (Γ > 1)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'rmse_width_by_gamma.png', dpi=130)

    bands = ['[1,2]', '(2,5]', '(5,10]', '(10,25]']
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.2
    for k, grp in enumerate(PLOT_GROUPS):
        p = cal[cal.group.eq(grp)].set_index('band').reindex(bands)
        if p.cov90_lower.isna().all():
            continue
        x = np.arange(len(bands)) + (k - 1.5) * width
        ax.bar(x, 0.5 * (p.cov90_lower + p.cov90_upper), width, color=COLORS[grp], label=f'{grp} (mean head cov90)')
        ax.scatter(x, p.outward_joint, color='black', marker='_', s=120)
    ax.axhline(0.9, color='grey', ls=':', lw=0.8)
    ax.set_xticks(np.arange(len(bands)), bands)
    ax.set(ylim=(0, 1), xlabel='Γ band', ylabel='coverage', title='90% endpoint coverage (bars) and outward joint containment (—)')
    ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'calibration_by_band.png', dpi=130)

    if (src / 'neuralcsa_comparison.csv').exists():
        c = pd.read_csv(src / 'neuralcsa_comparison.csv')
        c = c[c.group.isin(PLOT_GROUPS)]
        methods = ['pifm', 'neuralcsa_full', 'neuralcsa_oracle_stage']
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for k, m in enumerate(methods):
            p = c[c.method.eq(m)].set_index('group').reindex([x for x in PLOT_GROUPS if x in set(c.group)])
            x = np.arange(len(p)) + (k - 1) * 0.27
            ax.bar(x, 0.5 * (p.rmse_lower + p.rmse_upper), 0.27, label=m)
        ax.set_xticks(np.arange(len(p)), p.index)
        ax.set(ylabel='endpoint RMSE (same rows)', title='Rosenbaum: PI-FM vs NeuralCSA (Γ ∈ {2, 4, 9, 25})')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'neuralcsa_comparison.png', dpi=130)
    print('exported', out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['references', 'predict', 'report', 'neuralcsa-validate', 'neuralcsa-run', 'neuralcsa-ihdp', 'summary-references', 'summary-ihdp', 'plot'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checkpoint', default=str(ROSENBAUM_CHECKPOINT))
    parser.add_argument('--name')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--neuralcsa')
    parser.add_argument('--cohorts', nargs='+', default=['synthetic', 'acic'])
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.stage == 'references':
        build_references(args.smoke, args.cohorts)
    elif args.stage == 'predict':
        predict(args.smoke, args.checkpoint, args.name, args.snapshot, args.cohorts)
    elif args.stage == 'report':
        report(args.smoke, args.name, args.neuralcsa, args.cohorts, args.tag)
    elif args.stage.startswith('neuralcsa-'):
        run_neuralcsa(args.stage.removeprefix('neuralcsa-'), args.smoke)
    elif args.stage == 'summary-references':
        EXPORT.mkdir(parents=True, exist_ok=True)
        print(reference_summary().round(3).to_string())
        for name in ('neuralcsa_validate', 'neuralcsa_run'):
            pooled, by_gamma, timing = ncsa(name)
            print(name, pooled.round(4).to_string(), by_gamma.round(4).to_string(), timing.round(1).to_string(), sep='\n')
    elif args.stage == 'summary-ihdp':
        report_ihdp()
    else:
        assert args.name, 'Select a completed report with --name'
        plot(args.name)

if __name__ == '__main__':
    main()
