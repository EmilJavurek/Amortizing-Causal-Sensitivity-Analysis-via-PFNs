"""Local-fork TabPFN nuisances followed by closed-form MSM bounds and matched reports.

Per-arm predictive midpoint quantiles represent the observational outcome distribution.
The classifier supplies propensity; Gamma-independent fitting precedes bound evaluation."""

from __future__ import annotations

from experiments.shared.paths import REPORTS

from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
from experiments.shared.runtime import Run
from experiments.shared.msm_reference import msm_reference_from_samples
from experiments.comparisons.benchmark import GAMMAS, capo_to_cate, load_tasks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.shared.runtime import cluster_bootstrap, completed, write
from experiments.comparisons.benchmark import join, references, row_metrics, summarize
from experiments.comparisons.benchmark import pfn_outputs

PACKAGE = 'e10_tabpfn_msm'

LEVELS = 512

SEED = 0

def fit_predict(task, regressor_cls, classifier_cls):
    x, a, y = task.observed()
    query_ids, qx = task.query_x()
    every = np.vstack([qx, x])                         # queries first, then all context rows
    levels = (np.arange(LEVELS) + 0.5) / LEVELS
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    clf = classifier_cls(device='cuda', random_state=SEED).fit(x, a)
    e = clf.predict_proba(every)[:, list(clf.classes_).index(1)]
    quantiles, means = {}, {}
    for arm in (0, 1):
        reg = regressor_cls(device='cuda', random_state=SEED).fit(x[a == arm], y[a == arm])
        q = np.stack(reg.predict(every, output_type='quantiles', quantiles=list(levels)), axis=1)
        quantiles[arm] = np.sort(q, axis=1)            # guard against tiny quantile crossings
        means[arm] = reg.predict(every, output_type='mean')
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    bounds = {}
    for arm in (0, 1):
        pi = e if arm == 1 else 1 - e
        bounds[arm] = msm_reference_from_samples(quantiles[arm], pi, np.array(GAMMAS))
    bound_seconds = time.perf_counter() - start
    nq = len(qx)
    capo, apo, nuis = [], [], []
    for arm in (0, 1):
        lo, up = bounds[arm]
        for k, g in enumerate(GAMMAS):
            capo.append(pd.DataFrame(dict(task_id=task.task_id, query_id=query_ids, a_query=arm, gamma=g,
                                          lower=lo[:nq, k], upper=up[:nq, k])))
            apo.append(dict(task_id=task.task_id, a_query=arm, gamma=g, lower=lo[nq:, k].mean(), upper=up[nq:, k].mean()))
        nuis.append(pd.DataFrame(dict(task_id=task.task_id, kind=['query'] * nq + ['context'] * len(x),
                                      index=np.r_[query_ids, task.context.row.to_numpy()], a_query=arm,
                                      e_hat=e, mean_hat=means[arm], q_mean_hat=quantiles[arm].mean(axis=1),
                                      q05_hat=quantiles[arm][:, int(0.05 * LEVELS)], q95_hat=quantiles[arm][:, int(0.95 * LEVELS)])))
    timing = dict(task_id=task.task_id, fit_seconds=fit_seconds, query_seconds=bound_seconds,
                  rows_predicted=len(every), quantile_levels=LEVELS)
    return pd.concat(capo, ignore_index=True), pd.DataFrame(apo), pd.concat(nuis, ignore_index=True), timing

def run(smoke, cohort_set='default'):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    import tabpfn
    suffix = '' if cohort_set == 'default' else f'_{cohort_set}'
    run = Run(PACKAGE, ('smoke' if smoke else 'retained') + suffix, smoke=smoke, cohort_set=cohort_set, gpu=True, threads=4,
              tabpfn_implementation="local fork", tabpfn_source=str(Path(tabpfn.__file__).resolve()), quantile_levels=LEVELS, seed=SEED, gammas=GAMMAS)
    tasks = load_tasks(smoke=smoke, cohort_set=cohort_set)
    out = dict(capo=[], apo=[], nuisances=[], timing=[])
    try:
        for t in tasks:
            capo, apo, nuis, timing = fit_predict(t, TabPFNRegressor, TabPFNClassifier)
            out['capo'].append(capo); out['apo'].append(apo); out['nuisances'].append(nuis); out['timing'].append(timing)
            print(t.task_id, f"fit={timing['fit_seconds']:.1f}s", flush=True)
        capo = pd.concat(out['capo'], ignore_index=True)
        capo.to_parquet(run.path / 'capo.parquet', index=False)
        capo_to_cate(capo).to_parquet(run.path / 'cate.parquet', index=False)
        pd.concat(out['apo'], ignore_index=True).to_parquet(run.path / 'apo.parquet', index=False)
        pd.concat(out['nuisances'], ignore_index=True).to_parquet(run.path / 'nuisances.parquet', index=False)
        pd.DataFrame(out['timing']).to_csv(run.path / 'timing.csv', index=False)
        run.done(tasks=len(tasks))
    except BaseException:
        run.done('failed')
        raise


EXPORT = REPORTS / 'tabpfn'

def paired(a, b, keys):
    """Per-row absolute endpoint error difference (TabPFN - PI-FM); negative favours TabPFN."""
    m = a.merge(b, on=keys + ['group'], suffixes=('_tab', '_pfn'), validate='one_to_one')
    m['d_abs'] = 0.5 * (m.err_lower_tab.abs() + m.err_upper_tab.abs() - m.err_lower_pfn.abs() - m.err_upper_pfn.abs())
    rows = []
    for g, part in m.groupby('group'):
        stats = cluster_bootstrap(part, 'task_id', lambda f: dict(mean_abs_diff=float(f.d_abs.mean()),
                                                                tab_better=float((f.d_abs < 0).mean())))
        rows.append(dict(group=g, rows=len(part), **stats))
    return pd.DataFrame(rows)

def report():
    run = completed('e10_tabpfn_msm', 'retained')
    tasks = load_tasks()
    ref_capo, ref_cate, ref_apo = references(tasks)
    pfn_capo, pfn_cate, pfn_apo = pfn_outputs(tasks)
    tab = {k: pd.read_parquet(run / f'{k}.parquet') for k in ('capo', 'cate', 'apo')}
    EXPORT.mkdir(parents=True, exist_ok=True)
    out = {}
    capo_keys, cate_keys, apo_keys = ['task_id', 'query_id', 'a_query', 'gamma'], ['task_id', 'query_id', 'gamma'], ['task_id', 'a_query', 'gamma']
    for name, pred_tab, pred_pfn, ref, keys in (('capo', tab['capo'], pfn_capo, ref_capo, capo_keys),
                                                ('cate', tab['cate'], pfn_cate, ref_cate, cate_keys),
                                                ('apo', tab['apo'], pfn_apo, ref_apo, apo_keys)):
        m_tab = row_metrics(join(pred_tab, ref, keys))
        m_pfn = row_metrics(join(pred_pfn.drop(columns=[c for c in ('lower_q05', 'upper_q95') if c in pred_pfn and name == 'apo']), ref, keys))
        both = pd.concat([m_tab.assign(method='tabpfn_msm'), m_pfn.assign(method='pifm_msm')], ignore_index=True)
        summary = summarize(both, by=('group', 'method'))
        summary.to_csv(EXPORT / f'{name}_summary.csv', index=False)
        summarize(both[both.gamma.gt(1)], by=('group', 'method', 'gamma'), replicates=500).to_csv(EXPORT / f'{name}_by_gamma.csv', index=False)
        if name != 'apo':
            paired(m_tab, m_pfn, keys).to_csv(EXPORT / f'{name}_paired.csv', index=False)
        out[name] = summary
    timing = pd.read_csv(run / 'timing.csv')
    timing['group'] = timing.task_id.map(lambda t: 'synthetic' if t.startswith('syn_') else ('acic_full' if '_full_' in t else 'acic_d10'))
    timing.groupby('group')[['fit_seconds', 'query_seconds']].median().to_csv(EXPORT / 'timing_median.csv')
    write(EXPORT / 'run_info.json', dict(run='runs/e10_tabpfn_msm/retained', tasks=len(tasks),
                                         fit_seconds_median=float(timing.fit_seconds.median()),
                                         closed_form_seconds_median=float(timing.query_seconds.median())))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, group in zip(axes, ('synthetic', 'acic_d10', 'acic_full')):
        t = pd.read_csv(EXPORT / 'capo_by_gamma.csv')
        for method, color in (('pifm_msm', '#2b6cb0'), ('tabpfn_msm', '#c05621')):
            p = t[(t.group == group) & (t.method == method)]
            ax.plot(p.gamma, 0.5 * (p.rmse_lower + p.rmse_upper), marker='o', color=color, label=method)
        ax.set(title=group, xlabel='Gamma', ylabel='CAPO endpoint RMSE')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EXPORT / 'capo_rmse_by_gamma.png', dpi=130)
    for name, s in out.items():
        cols = ['group', 'method', 'rmse_lower', 'rmse_upper', 'covers_ref', 'width', 'ref_width'] + (['true_in'] if 'true_in' in s else [])
        print(name); print(s[cols].round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'report'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', default='default', choices=['default', 'ihdp'])
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.smoke, args.cohort)
    else:
        assert not args.smoke and args.cohort == 'default', 'report covers the retained default cohort'
        report()


if __name__ == '__main__':
    main()
