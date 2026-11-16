"""Shared observational cohorts, matched MSM references, metrics and comparison reports.

Comparators receive context X/A/Y and query covariates only; references and frozen PI-FM
predictions are reserved for evaluation. Bounds use observed-context outcome units.
Cluster bootstrap resamples whole tasks."""

from __future__ import annotations

from experiments.shared.paths import REPORTS

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from experiments.shared.runtime import completed, dataset_root, read
import numpy as np
from experiments.shared.runtime import cluster_bootstrap
import argparse
from experiments.shared.runtime import DATA, RUNS, OBSERVED_RUNS, Run, fresh, sha, write
from experiments.shared.msm_reference import msm_reference_from_samples
from experiments.shared.inputs import cohort_ids, cohort_artifacts
import time

GAMMAS = (1.0, 1.4, 2.0, 3.0, 5.0)

@dataclass
class Task:
    task_id: str
    meta: dict
    context: pd.DataFrame
    queries: pd.DataFrame
    apo: pd.DataFrame
    pfn_dir: Path

    @property
    def xcols(self):
        return [c for c in self.context.columns if c.startswith('x_')]

    def observed(self):
        """What a comparator may see: X, A, Y of the context."""
        return (self.context[self.xcols].to_numpy(), self.context.treatment.to_numpy().astype(int),
                self.context.outcome.to_numpy())

    def query_x(self):
        """Unique query covariate rows ordered by query_id."""
        unique = self.queries.drop_duplicates('query_id').sort_values('query_id')
        return unique.query_id.to_numpy(), unique[self.xcols].to_numpy()

    def grid_gammas(self):
        return [g for g in GAMMAS]

COHORT_SETS = {'default': ('benchmark', ('synthetic', 'acic')), 'ihdp': ('benchmark_ihdp', ('ihdp',))}

def cohort_root(smoke=False, cohort_set='default'):
    return dataset_root(COHORT_SETS[cohort_set][0], smoke)

def prediction_root(package, smoke=False):
    return RUNS / package / ('pifm_smoke' if smoke else 'pifm')

def load_tasks(smoke=False, cohorts=None, task_ids=None, cohort_set='default'):
    """cohort_set 'default': shared synthetic + ACIC tasks; 'ihdp': IHDP tasks."""
    root = cohort_root(smoke, cohort_set)
    cohorts = COHORT_SETS[cohort_set][1] if cohorts is None else cohorts
    tasks = []
    targets = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(targets) == read(root / 'manifest.json')['summary']['tasks'], 'Incomplete benchmark task set'
    for target in targets:
        meta = read(target / 'task.json')
        if meta['cohort'] not in cohorts or (task_ids is not None and meta['task_id'] not in task_ids):
            continue
        queries = pd.read_parquet(target / 'queries.parquet')
        assert not any(c.startswith(('upper_', 'lower_')) for c in queries), 'Model predictions mixed into evaluation inputs'
        tasks.append(Task(meta['task_id'], meta, pd.read_parquet(target / 'context.parquet'),
                          queries, pd.read_parquet(target / 'apo_reference.parquet'),
                          prediction_root(COHORT_SETS[cohort_set][0], smoke) / target.name))
    assert tasks
    return tasks

def pfn_query_predictions(task):
    """Load PI-FM outputs explicitly; comparator inputs never contain predictions."""
    metadata = read(task.pfn_dir.parent / 'manifest.json')
    assert metadata['status'] == 'completed', 'PI-FM benchmark predictions incomplete'
    inputs = dataset_root(metadata['dataset'], metadata['smoke'])
    assert metadata['dataset_manifest_sha256'] == sha(inputs / 'manifest.json'), 'PI-FM outputs belong to a different dataset build'
    predictions = pd.read_parquet(task.pfn_dir / 'query_predictions.parquet')
    keys = ['query_id', 'a_query', 'gamma']
    matched = task.queries[keys].merge(predictions[keys], on=keys, how='outer', validate='one_to_one', indicator=True)
    assert matched._merge.eq('both').all(), 'PI-FM predictions and query set differ'
    return predictions

def pfn_outputs(tasks):
    """Frozen MSM PI-FM in the comparator contract (GMM means; CATE/APO by differencing/averaging)."""
    capo, cate, apo = [], [], []
    for t in tasks:
        predictions = pfn_query_predictions(t)
        q = predictions[predictions.gamma.isin(GAMMAS)]
        capo.append(pd.DataFrame(dict(task_id=t.task_id, query_id=q.query_id, a_query=q.a_query, gamma=q.gamma,
                                      lower=q.lower_mean, upper=q.upper_mean, lower_q05=q.lower_q05, upper_q95=q.upper_q95)))
        context = pd.read_parquet(t.pfn_dir / 'context_predictions.parquet')
        apo.append(context.groupby(['a_query', 'gamma'], as_index=False)[['lower_mean', 'upper_mean']].mean()
                   .rename(columns={'lower_mean': 'lower', 'upper_mean': 'upper'}).assign(task_id=t.task_id))
    capo = pd.concat(capo, ignore_index=True)
    cate = capo_to_cate(capo)
    return capo, cate, pd.concat(apo, ignore_index=True)

def capo_to_cate(capo):
    """Sharp-under-MSM combination: [lower(1) - upper(0), upper(1) - lower(0)] (arms confound separately)."""
    keys = ['task_id', 'query_id', 'gamma']
    one = capo[capo.a_query.eq(1)].set_index(keys)
    zero = capo[capo.a_query.eq(0)].set_index(keys)
    out = pd.DataFrame(dict(lower=one.lower - zero.upper, upper=one.upper - zero.lower))
    if 'lower_q05' in capo:
        out['lower_q05'] = one.lower_q05 - zero.upper_q95
        out['upper_q95'] = one.upper_q95 - zero.lower_q05
    return out.reset_index()

TOL = 1e-9

def group_of(task_id):
    if task_id.startswith('ihdp'):
        return 'ihdp'
    return 'synthetic' if task_id.startswith('syn_') else ('acic_full' if '_full_' in task_id else 'acic_d10')

def references(tasks):
    capo, cate, apo = [], [], []
    for t in tasks:
        q = t.queries[t.queries.gamma.isin(GAMMAS)]
        ref = q[['query_id', 'a_query', 'gamma', 'theta_lower', 'theta_upper', 'true_capo', 'pi_query']].assign(task_id=t.task_id)
        capo.append(ref)
        keys = ['query_id', 'gamma']
        one, zero = ref[ref.a_query.eq(1)].set_index(keys), ref[ref.a_query.eq(0)].set_index(keys)
        cate.append(pd.DataFrame(dict(ref_lower=one.theta_lower - zero.theta_upper, ref_upper=one.theta_upper - zero.theta_lower,
                                      true=one.true_capo - zero.true_capo, p_a1=one.pi_query)).reset_index().assign(task_id=t.task_id))
        apo.append(t.apo.rename(columns={'apo_lower': 'ref_lower', 'apo_upper': 'ref_upper', 'apo_true': 'true'}).assign(task_id=t.task_id))
    capo = pd.concat(capo, ignore_index=True).rename(columns={'theta_lower': 'ref_lower', 'theta_upper': 'ref_upper', 'true_capo': 'true'})
    apo = pd.concat(apo, ignore_index=True)
    if 'true' not in apo:
        apo['true'] = np.nan
    return capo, pd.concat(cate, ignore_index=True), apo

def join(pred, ref, keys):
    out = pred.merge(ref, on=keys, how='inner', validate='one_to_one')
    assert len(out) == len(pred), 'prediction rows without a reference'
    out['group'] = out.task_id.map(group_of)
    return out

def row_metrics(frame):
    f = frame.copy()
    f['err_lower'] = f.lower - f.ref_lower
    f['err_upper'] = f.upper - f.ref_upper
    f['covers_ref'] = (f.lower <= f.ref_lower + TOL) & (f.upper >= f.ref_upper - TOL)
    f['ref_width'] = f.ref_upper - f.ref_lower
    f['width'] = f.upper - f.lower
    f['true_in'] = np.where(f['true'].notna(), (f.lower <= f['true']) & (f['true'] <= f.upper), np.nan)
    if 'lower_q05' in f:
        f['outward_covers_ref'] = (f.lower_q05 <= f.ref_lower + TOL) & (f.upper_q95 >= f.ref_upper - TOL)
    return f

def statistic(frame):
    out = dict(rmse_lower=float(np.sqrt(np.mean(frame.err_lower ** 2))),
               rmse_upper=float(np.sqrt(np.mean(frame.err_upper ** 2))),
               mae=float(np.mean(np.abs(np.r_[frame.err_lower, frame.err_upper]))),
               covers_ref=float(frame.covers_ref.mean()),
               width=float(frame.width.mean()), ref_width=float(frame.ref_width.mean()))
    if frame['true'].notna().any():
        out['true_in'] = float(frame.true_in.astype(float).mean())
    return out

def summarize(frame, by=('group',), replicates=2000):
    rows = []
    for key, part in frame.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        stats = cluster_bootstrap(part, 'task_id', statistic, replicates=replicates)
        rows.append({**dict(zip(by, key)), 'rows': len(part), 'tasks': part.task_id.nunique(), **stats})
    return pd.DataFrame(rows)

PACKAGE = 'benchmark'

SYN_COUNT = 24

DRAWS = 16384

def gamma_mask(values):
    return np.any(np.isclose(np.asarray(values)[:, None], np.array(GAMMAS)[None, :], atol=1e-9), axis=1)

def snap(values):
    values = np.asarray(values, float)
    out = values.copy()
    for g in GAMMAS:
        out[np.isclose(values, g, atol=1e-9)] = g
    return out

def write_task(root, predictions_root, task_id, meta, context, queries, apo, pfn_context):
    target = root / task_id
    target.mkdir(parents=True)
    predictions = predictions_root / task_id
    predictions.mkdir()
    model_columns = [c for c in queries if c.startswith(('upper_', 'lower_'))]
    assert model_columns, 'Missing PI-FM query predictions'
    context.to_parquet(target / 'context.parquet', index=False)
    queries.drop(columns=model_columns).to_parquet(target / 'queries.parquet', index=False)
    apo.to_parquet(target / 'apo_reference.parquet', index=False)
    queries[['query_id', 'a_query', 'gamma'] + model_columns].to_parquet(predictions / 'query_predictions.parquet', index=False)
    pfn_context.to_parquet(predictions / 'context_predictions.parquet', index=False)
    write(target / 'task.json', meta)

def finish_build(run, predictions_root):
    run.done(tasks=len(run.manifest['tasks']))
    write(predictions_root / 'manifest.json', dict(
        status='completed', smoke=run.manifest['smoke'], dataset=run.path.parent.name,
        dataset_manifest_sha256=sha(run.path / 'manifest.json'),
        checkpoint_sha256=run.manifest['config']['msm_checkpoint_sha256'],
        git={k: run.manifest['git'][k] for k in ('commit', 'dirty')},
    ))

def pfn_on_rows(model, context, xcols, rows_x, device):
    from experiments.shared.evaluation import pifm_predict_frame
    grid = []
    for arm in (0, 1):
        for g in GAMMAS:
            frame = rows_x.copy()
            frame['a_query'], frame['gamma'] = arm, g
            grid.append(frame)
    query = pd.concat(grid, ignore_index=True)
    pred = pifm_predict_frame(model, context, xcols, query, device, chunk_size=4096)
    return pred[['row', 'a_query', 'gamma', 'lower_mean', 'upper_mean']]

def build_synthetic(run, root, predictions_root, model, device, count):
    import torch
    from experiments.shared.evaluation import normalized_dataset
    import experiments.shared.msm_reference as legacy
    artifacts = cohort_artifacts('msm')
    ids = cohort_ids('msm')[:count]
    for pos, i in enumerate(ids):
        frame, xcols = normalized_dataset(artifacts[i])
        assert xcols == [f'x_{j}' for j in range(10)]
        fragment = pd.read_parquet(OBSERVED_RUNS / 'e2' / 'fragments' / f'dgp_{i}.parquet')
        fragment = fragment[gamma_mask(fragment.gamma)].copy()
        fragment['gamma'] = snap(fragment.gamma)
        assert len(fragment) == 8 * 2 * len(GAMMAS)
        keys = fragment.drop_duplicates(['source_row_index']).source_row_index.tolist()
        fragment['query_id'] = fragment.source_row_index.map({k: n for n, k in enumerate(keys)})
        queries = fragment.rename(columns={'source_row_index': 'row'})
        queries['true_capo'] = np.nan  # synthetic truth is the Gamma=1 reference; no hidden-U truth stored
        context = frame[xcols + ['treatment', 'outcome', 'propensity']].copy()
        context['row'] = np.arange(len(context))
        # Population APO reference over all context rows with the shared oracle protocol.
        outcome = legacy.load_outcome_map(artifacts[i], device)
        rows = []
        for arm in (0, 1):
            pi = frame.propensity.to_numpy() if arm == 1 else 1 - frame.propensity.to_numpy()
            latent = legacy.seeded_normal((len(frame), DRAWS), 20261107 + 1000 * pos + 10 * arm + i, device)
            with torch.no_grad():
                samples = outcome.evaluate(torch.tensor(frame[xcols].to_numpy(), dtype=torch.float32, device=device),
                                           torch.full((len(frame),), float(arm), device=device), latent).cpu().numpy()
            lo, up = msm_reference_from_samples(samples, pi, np.array(GAMMAS))
            for k, g in enumerate(GAMMAS):
                rows.append(dict(a_query=arm, gamma=g, apo_lower=lo[:, k].mean(), apo_upper=up[:, k].mean()))
        rows_x = context[['row'] + xcols]
        pfn_ctx = pfn_on_rows(model, frame, xcols, rows_x, device)
        meta = dict(task_id=f'syn_{i}', cohort='synthetic', dataset_id=i, d=10, n_context=len(frame),
                    true_propensity_available=True, y_units='observed-standardized',
                    capo_reference='16,384-draw closed-form MSM', apo_draws=DRAWS)
        write_task(root, predictions_root, f'syn_{i}', meta, context, queries, pd.DataFrame(rows), pfn_ctx)
        run.manifest.setdefault('tasks', []).append(f'syn_{i}')
        run.save()
        print(f'synthetic {pos + 1}/{len(ids)}', flush=True)

def build_acic(run, root, predictions_root, model, device, reps):

    import experiments.applications.acic as dgp
    e5_root = dataset_root('e05_acic')
    pfn = pd.read_parquet(completed('e05_acic', 'pfn') / 'predictions.parquet')
    raw = pd.read_parquet(DATA / 'acic2016' / 'input_2016.parquet')
    for variant in ('d10', 'full'):
        mech = dgp.mechanism(dgp.encode(raw, variant))
        for rep in range(reps):
            src = e5_root / variant / f'rep{rep}'
            scaling = read(src / 'scaling.json')
            context = pd.read_parquet(src / 'context.parquet')
            xcols = [c for c in context.columns if c.startswith('x_')]
            qx = pd.read_parquet(src / 'query_x.parquet')
            ref = pd.read_parquet(src / 'msm_reference.parquet')
            ref = ref[ref.gamma_kind.isin(['grid', 'median_oracle'])]
            p = pfn[(pfn.variant == variant) & (pfn.rep == rep) & (pfn.model == 'msm')]
            p = p[p.gamma_kind.isin(['grid', 'median_oracle'])]
            cols = [c for c in p.columns if c.startswith(('upper_', 'lower_'))]
            queries = ref.merge(p[['query_row', 'a_query', 'gamma'] + cols], on=['query_row', 'a_query', 'gamma'],
                                validate='one_to_one').merge(qx, on='query_row', validate='many_to_one')
            queries = queries.rename(columns={'query_row': 'row'})
            queries['query_id'] = queries.row.map({r: n for n, r in enumerate(sorted(queries.row.unique()))})
            rows = []
            context_rows = context.row.to_numpy()
            for arm in (0, 1):
                pi = mech.pi[context_rows] if arm == 1 else 1 - mech.pi[context_rows]
                draws = dgp.observational_samples(mech, context_rows, arm, DRAWS, seed=5000 + 10 * rep + arm)
                lo, up = msm_reference_from_samples(draws, pi, np.array(GAMMAS))
                truth = dgp.true_capo(mech, context_rows, arm).mean()
                for k, g in enumerate(GAMMAS):
                    rows.append(dict(a_query=arm, gamma=g, apo_lower=(lo[:, k].mean() - scaling['y_mean']) / scaling['y_std'],
                                     apo_upper=(up[:, k].mean() - scaling['y_mean']) / scaling['y_std'],
                                     apo_true=(truth - scaling['y_mean']) / scaling['y_std']))
            frame = context.copy()
            pfn_ctx = pfn_on_rows(model, frame, xcols, context[['row'] + xcols], device)
            task = f'acic_{variant}_rep{rep}'
            meta = dict(task_id=task, cohort='acic', variant=variant, rep=rep, d=len(xcols), n_context=len(context),
                        true_propensity_available=True, y_units='observed-standardized', apo_draws=DRAWS,
                        capo_reference='E5 exact observational law, 16,384 draws, closed-form MSM')
            write_task(root, predictions_root, task, meta, context, queries, pd.DataFrame(rows), pfn_ctx)
            run.manifest.setdefault('tasks', []).append(task)
            run.save()
            print(task, flush=True)

def build(smoke):
    import torch
    from experiments.shared.evaluation import MSM_CHECKPOINT, load_pifm
    run = Run(PACKAGE, 'cohort_smoke' if smoke else 'cohort', smoke=smoke, gpu=True, threads=4, dataset=True,
              gammas=GAMMAS, synthetic_count=SYN_COUNT, msm_checkpoint_sha256=sha(MSM_CHECKPOINT))
    root = run.path
    predictions_root = fresh(prediction_root(PACKAGE, smoke))
    device = torch.device('cuda:0')
    model = load_pifm(MSM_CHECKPOINT, device)[0]
    try:
        build_synthetic(run, root, predictions_root, model, device, 2 if smoke else SYN_COUNT)
        build_acic(run, root, predictions_root, model, device, 1 if smoke else 5)
        finish_build(run, predictions_root)
    except BaseException:
        run.done('failed')
        raise

def time_pfn():
    import torch
    from experiments.shared.evaluation import MSM_CHECKPOINT, load_pifm, pifm_predict_frame
    run = Run('benchmark', 'time_pfn', gpu=True, threads=4, repeats=3, checkpoint_sha256=sha(MSM_CHECKPOINT))
    device = torch.device('cuda:0')
    start = time.perf_counter()
    model = load_pifm(MSM_CHECKPOINT, device)[0]
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    rows = []
    try:
        for t in load_tasks():
            xcols = t.xcols
            q = t.queries[['a_query', 'gamma'] + xcols]
            grid = pd.concat([t.context[xcols].assign(a_query=a, gamma=g) for a in (0, 1) for g in GAMMAS], ignore_index=True)
            query = pd.concat([q, grid], ignore_index=True)
            times = []
            for repeat in range(4):
                torch.cuda.synchronize()
                s = time.perf_counter()
                pifm_predict_frame(model, t.context, xcols, query, device, chunk_size=4096)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - s)
            rows.append(dict(task_id=t.task_id, predictions=len(query), seconds=float(np.median(times[1:]))))
            print(rows[-1], flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(run.path / 'timing.csv', index=False)
        run.done(load_seconds=load_seconds, median_seconds=float(frame.seconds.median()))
    except BaseException:
        run.done('failed')
        raise

EXPORT = REPORTS / 'baseline_comparison'

COMPARE_GAMMAS = (1.4, 2.0, 3.0, 5.0)

def ate_from_apo(apo):
    keys = ['task_id', 'gamma']
    one, zero = apo[apo.a_query.eq(1)].set_index(keys), apo[apo.a_query.eq(0)].set_index(keys)
    out = pd.DataFrame(dict(lower=one.lower - zero.upper, upper=one.upper - zero.lower))
    return out.reset_index()

def compare(e8_run):
    tasks = load_tasks()
    ref_capo, ref_cate, ref_apo = references(tasks)
    ref_ate = ate_from_apo(ref_apo.rename(columns={'ref_lower': 'lower', 'ref_upper': 'upper'})).rename(
        columns={'lower': 'ref_lower', 'upper': 'ref_upper'})
    truth = ref_apo.pivot_table(index='task_id', columns='a_query', values='true', aggfunc='first')
    ref_ate['true'] = ref_ate.task_id.map((truth[1] - truth[0]) if 1 in truth else {})
    pfn_capo, pfn_cate, pfn_apo = pfn_outputs(tasks)
    e10 = completed('e10_tabpfn_msm', 'retained')
    e8 = completed('e08_blearner', e8_run)
    e9 = completed('e09_dorn_guo', 'qb')
    cate = {'PI-FM (direct)': pfn_cate, 'TabPFN + closed form': pd.read_parquet(e10 / 'cate.parquet')}
    impl = "authors' code" if e8_run.startswith('authors') else 'reimpl.'
    impl += '' if 'clip' not in e8_run else ', propensity clipped (control)'
    if e8 is not None:
        assert read(e8 / 'manifest.json')['status'] == 'completed'
        cate[f'B-Learner ({impl})'] = pd.read_parquet(e8 / 'cate.parquet')
        cate[f'B-Learner plug-in ({impl})'] = capo_to_cate(pd.read_parquet(e8 / 'capo_plugin.parquet'))
    ate = {'PI-FM (direct, mean of CAPO)': ate_from_apo(pfn_apo), 'TabPFN + closed form': ate_from_apo(pd.read_parquet(e10 / 'apo.parquet'))}
    qb = pd.read_parquet(e9 / 'qb_bounds.parquet')
    for method, part in qb.groupby('method'):
        ate[f'Dorn-Guo {method}'] = part.rename(columns={'ate_lower': 'lower', 'ate_upper': 'upper'})[['task_id', 'gamma', 'lower', 'upper']]
    EXPORT.mkdir(parents=True, exist_ok=True)
    tables = []
    for estimand, preds, ref, keys in (('CATE', cate, ref_cate, ['task_id', 'query_id', 'gamma']),
                                       ('ATE', ate, ref_ate, ['task_id', 'gamma'])):
        for method, pred in preds.items():
            pred = pred[np.isin(np.round(pred.gamma, 6), COMPARE_GAMMAS)][keys + ['lower', 'upper']].copy()
            pred['gamma'] = np.round(pred.gamma, 6)
            r = ref.copy()
            r['gamma'] = np.round(r.gamma, 6)
            m = row_metrics(join(pred, r[np.isin(r.gamma, COMPARE_GAMMAS)], keys))
            s = summarize(m, by=('group',), replicates=1000)
            s.insert(0, 'method', method)
            s.insert(0, 'estimand', estimand)
            tables.append(s)
    out = pd.concat(tables, ignore_index=True)
    out['width_ratio'] = out.width / out.ref_width
    suffix = '' if e8_run == 'cohort' else f'_e8{e8_run}'   # never overwrite the committed reimplementation table
    out['gamma_grid'] = 'Gamma in {1.4, 2, 3, 5} (Gamma=1 excluded; package summaries pool {1,1.4,2,3,5})'
    out['e8_run'] = str(e8.relative_to(e8.parents[2])) if e8 is not None else ''
    out.to_csv(EXPORT / f'comparison_summary{suffix}.csv', index=False)
    tp = completed('benchmark', 'time_pfn')
    timing = [pd.read_csv(tp / 'timing.csv').assign(method='PI-FM (direct), all queries+context rows, all Gamma')] if tp else []
    timing += [pd.read_csv(e10 / 'timing.csv').assign(method='TabPFN + closed form', seconds=lambda f: f.fit_seconds + f.query_seconds)]
    if e8 is not None:
        t8 = pd.read_csv(e8 / 'timing.csv')
        assert {'base_fit_seconds', 'gamma_fit_seconds', 'query_seconds'} <= set(t8.columns)
        per_task = t8.groupby('task_id').apply(lambda f: f.base_fit_seconds.iloc[0] + (f.gamma_fit_seconds + f.query_seconds).sum(),
                                               include_groups=False)
        timing.append(per_task.rename('seconds').reset_index().assign(method=f'B-Learner ({impl}), all Gamma'))
    t9 = pd.read_csv(e9 / 'timing.csv')
    secs = [c for c in t9.columns if 'second' in c]
    t9['seconds'] = t9[secs[0]]
    timing.append(t9.assign(method='Dorn-Guo ' + t9['method']) if 'method' in t9 else t9.assign(method='Dorn-Guo'))
    timing = pd.concat([t[['task_id', 'method', 'seconds']] for t in timing], ignore_index=True)
    timing['group'] = timing.task_id.map(group_of)
    timing.groupby(['method', 'group']).seconds.median().reset_index().to_csv(EXPORT / f'comparison_timing_median{suffix}.csv', index=False)
    cols = ['estimand', 'method', 'group', 'rmse_lower', 'rmse_upper', 'covers_ref', 'width_ratio'] + (['true_in'] if 'true_in' in out else [])
    print(out[cols].round(3).to_string())
    print(timing.groupby(['method', 'group']).seconds.median().round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'time', 'compare'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--e8-run', default='authors')
    args = parser.parse_args()
    if args.stage == 'build':
        build(args.smoke)
    else:
        assert not args.smoke, 'time/compare use the retained cohort'
        time_pfn() if args.stage == 'time' else compare(args.e8_run)


if __name__ == '__main__':
    main()
