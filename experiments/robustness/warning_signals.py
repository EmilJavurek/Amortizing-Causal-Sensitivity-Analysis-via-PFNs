"""Fit warning-signal nuisances, freeze thresholds, and assess held-out harmful errors."""

from __future__ import annotations

from experiments.shared.paths import REPORTS
import matplotlib
matplotlib.use('Agg')
from experiments.shared.paths import OBSERVED_RUNS, OBSERVED_DATA
import argparse
import time
import numpy as np
import pandas as pd
from experiments.shared.runtime import Run
import matplotlib.pyplot as plt
from experiments.shared.runtime import Run, cluster_bootstrap, completed


PACKAGE = 'e12_warning_signals'
PER_STRATUM = 32
FAMILIES = ('oscillatory_outcome', 'heavy_tailed_outcome', 'extreme_overlap', 'gamma_extrapolation')

def rename_bare(frame):
    return frame.rename(columns={c: f'x_{c[1:]}' for c in frame.columns if c[:1] == 'x' and c[1:].isdigit()})

def contexts(smoke):
    """Yield (cohort, stratum, row_dataset_id, context_frame, fragment_queries)."""
    limit = 2 if smoke else PER_STRATUM
    from experiments.shared.inputs import cohort_ids, input_path
    e2_ids = cohort_ids('msm')[:2 if smoke else 2 * PER_STRATUM]
    for i in e2_ids:
        ctx = rename_bare(pd.read_csv(input_path('normalized_root') / f'datasets/synthetic_continous_dataset_{i}.csv'))
        yield 'E2_fresh_in_prior', 'all', str(i), ctx, pd.read_parquet(OBSERVED_RUNS / f'e2/fragments/dgp_{i}.parquet')
    for path in sorted(OBSERVED_DATA.glob('e7_context/dgp_*'), key=lambda p: int(p.name[4:]))[:limit]:
        i = path.name[4:]
        frag = pd.read_parquet(OBSERVED_RUNS / f'e7_context/fragments/dgp_{i}.parquet')
        full = pd.read_csv(path / 'context_4096.csv')   # nested contexts; queries index the longest one
        for n in (128, 256, 512, 1024, 2048, 4096):
            ctx = pd.read_csv(path / f'context_{n}.csv')
            ctx.attrs['query_source'] = full
            yield 'E7_context', f'n={n}', i, ctx, frag[frag.context_length.eq(n)]
    for d in (2, 5, 10, 20, 40):
        frags = sorted(OBSERVED_RUNS.glob(f'e7_dimension/fragments/dim_{d}_dgp_*.parquet'), key=lambda p: int(p.stem.split('_')[-1]))[:limit]
        for frag_path in frags:
            k = frag_path.stem.split('_')[-1]
            frag = pd.read_parquet(frag_path)
            ctx = rename_bare(pd.read_csv(OBSERVED_DATA / f'e7_dimension/dimension_{d}/datasets/synthetic_continous_dataset_{k}.csv'))
            yield 'E7_dimension', f'd={d}', f'd={d}:{frag.dataset_id.iloc[0]}', ctx, frag
    for family in FAMILIES:
        for path in sorted(OBSERVED_DATA.glob(f'e9/{family}/dgp_*'))[:limit]:
            k = path.name[4:]
            yield f'E9_{family}', None, f'{family}_{k}', pd.read_csv(path / 'context_observed.csv'), \
                pd.read_parquet(OBSERVED_RUNS / f'e9/fragments/{family}_{k}.parquet')

def fit_nuisances(smoke):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    run = Run(PACKAGE, 'nuisances_smoke' if smoke else 'nuisances', smoke=smoke, gpu=True, threads=4,
              per_stratum=PER_STRATUM, split='alternating sorted DGP order: even position -> threshold, odd -> assessment')
    out, position = [], {}
    try:
        for cohort, stratum, dataset_id, ctx, frag in contexts(smoke):
            xcols = [c for c in ctx.columns if c.startswith('x_')]
            q = frag.drop_duplicates(['source_row_index', 'a_query'])[['source_row_index', 'a_query'] + xcols]
            src = ctx.attrs.get('query_source', ctx).iloc[q.source_row_index.to_numpy()][xcols].to_numpy()
            assert np.allclose(src, q[xcols].to_numpy(), atol=1e-5), (cohort, dataset_id, 'query/context mismatch')
            x, a, y = ctx[xcols].to_numpy(), ctx.treatment.to_numpy().astype(int), ctx.outcome.to_numpy()
            start = time.perf_counter()
            clf = TabPFNClassifier(device='cuda', random_state=0).fit(x, a)
            e = clf.predict_proba(q[xcols].to_numpy())[:, list(clf.classes_).index(1)]
            mean = np.empty(len(q))
            for arm in (0, 1):
                mask = q.a_query.to_numpy() == arm
                reg = TabPFNRegressor(device='cuda', random_state=0).fit(x[a == arm], y[a == arm])
                if mask.any():
                    mean[mask] = reg.predict(q[xcols].to_numpy()[mask], output_type='mean')
            g1 = frag[np.isclose(frag.gamma, 1.0)].set_index(['source_row_index', 'a_query'])
            key = (cohort, stratum)
            position[key] = position.get(key, 0) + 1
            out.append(pd.DataFrame(dict(cohort=cohort, stratum=stratum, dataset_id=dataset_id,
                                         split='threshold' if position[key] % 2 == 1 else 'assessment',
                                         source_row_index=q.source_row_index.to_numpy(), a_query=q.a_query.to_numpy(),
                                         n_context=len(ctx), d=len(xcols), treated_fraction=a.mean(),
                                         e_hat=e, mean_hat=mean,
                                         pfn_mid_g1=0.5 * (g1.lower_mean + g1.upper_mean).reindex(list(zip(q.source_row_index, q.a_query))).to_numpy(),
                                         fit_seconds=time.perf_counter() - start)))
            if len(out) % 20 == 0:
                print(len(out), cohort, stratum, dataset_id, flush=True)
        frame = pd.concat(out, ignore_index=True)
        frame['stratum'] = frame.stratum.fillna('trained_range')
        assert frame.pfn_mid_g1.notna().all()
        frame.to_parquet(run.path / 'nuisances.parquet', index=False)
        run.done(contexts=len(out), rows=len(frame))
    except BaseException:
        run.done('failed')
        raise


HARM = 0.1
EXPORT = REPORTS / 'warning_signals'

def observed_rows(nuis):
    rows = pd.read_parquet(completed('e02_outward_intervals', 'retained') / 'rows.parquet')
    rows = rows[rows.sensitivity_model.eq('msm')]
    keys = ['cohort', 'stratum', 'dataset_id', 'source_row_index', 'a_query']
    rows['stratum'] = np.where(rows.cohort.str.startswith('E9'), 'trained_range', rows.stratum)
    if 'E9_gamma_extrapolation' in set(nuis.cohort):
        pass
    m = rows.merge(nuis.drop_duplicates(keys), on=keys, how='inner', validate='many_to_one')
    covered = nuis.drop_duplicates(keys).merge(rows[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    assert (covered._merge == 'both').all(), 'nuisance rows without E2 labels'
    return m[['cohort', 'stratum', 'dataset_id', 'split', 'source_row_index', 'a_query', 'gamma', 'severity',
              'n_context', 'd', 'e_hat', 'mean_hat', 'pfn_mid_g1']]

def acic_rows():
    from experiments.comparisons.benchmark import load_tasks, pfn_query_predictions
    tasks = load_tasks(cohorts=('acic',))
    e10 = pd.read_parquet(completed('e10_tabpfn_msm', 'retained') / 'nuisances.parquet')
    e10 = e10[e10.kind.eq('query')].rename(columns={'index': 'query_id'})
    out = []
    for t in tasks:
        q = t.queries.merge(pfn_query_predictions(t), on=['query_id', 'a_query', 'gamma'], validate='one_to_one')
        g1 = q[np.isclose(q.gamma, 1.0)].set_index(['query_id', 'a_query'])
        f = q[['query_id', 'a_query', 'gamma', 'lower_q05', 'upper_q95', 'theta_lower', 'theta_upper']].copy()
        f['severity'] = np.maximum(0, f.lower_q05 - f.theta_lower) + np.maximum(0, f.theta_upper - f.upper_q95)
        f['pfn_mid_g1'] = 0.5 * (g1.lower_mean + g1.upper_mean).reindex(list(zip(f.query_id, f.a_query))).to_numpy()
        n = e10[e10.task_id.eq(t.task_id)][['query_id', 'a_query', 'e_hat', 'mean_hat']]
        f = f.merge(n, on=['query_id', 'a_query'], validate='many_to_one')
        f = f.assign(cohort='ACIC', stratum=f"{t.meta['variant']}", dataset_id=t.task_id, split='assessment',
                     source_row_index=f.query_id, n_context=t.meta['n_context'], d=t.meta['d'])
        out.append(f[['cohort', 'stratum', 'dataset_id', 'split', 'source_row_index', 'a_query', 'gamma', 'severity',
                      'n_context', 'd', 'e_hat', 'mean_hat', 'pfn_mid_g1']])
    return pd.concat(out, ignore_index=True)

def add_scores(f):
    f = f.copy()
    f['harmful'] = f.severity > HARM
    f['any_miss'] = f.severity > 0
    f['pi_hat_query'] = np.where(f.a_query.eq(1), f.e_hat, 1 - f.e_hat)
    f['g1_gap'] = (f.pfn_mid_g1 - f.mean_hat).abs()
    f['flag_regime'] = (f.n_context < 512) | (f.d > 10) | (f.gamma > 5 + 1e-9)
    return f

def apply_thresholds(f, t_support, t_g1):
    f = f.copy()
    f['flag_support'] = f.pi_hat_query < t_support
    f['flag_g1'] = f.g1_gap > t_g1
    f['flag_any'] = f.flag_regime | f.flag_support | f.flag_g1
    return f

def signal_stats(f, flag):
    flagged, harm = f[flag], f.harmful
    n_h = harm.sum()
    return dict(flag_rate=float(flagged.mean()),
                harmful_rate=float(harm.mean()),
                recall=float((flagged & harm).sum() / n_h) if n_h else np.nan,
                precision=float((flagged & harm).sum() / flagged.sum()) if flagged.sum() else np.nan,
                false_alarm=float((flagged & ~harm).sum() / (~harm).sum()) if (~harm).sum() else np.nan,
                harmful_rate_flagged=float(harm[flagged].mean()) if flagged.sum() else np.nan,
                harmful_rate_retained=float(harm[~flagged].mean()) if (~flagged).sum() else np.nan,
                containment_retained=float((~f.any_miss[~flagged]).mean()) if (~flagged).sum() else np.nan,
                containment_all=float((~f.any_miss).mean()),
                retained_fraction=float((~flagged).mean()))

def table(frame, by, signals, boot=False):
    rows = []
    for key, part in frame.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for s in signals:
            if boot:
                stats = cluster_bootstrap(part, 'dataset_id', lambda g, s=s: {k: (0.0 if v != v else v) for k, v in signal_stats(g, f'flag_{s}').items()}, replicates=1000)
            else:
                stats = signal_stats(part, f'flag_{s}')
            rows.append({**dict(zip(by, key)), 'signal': s, 'rows': len(part), 'dgps': part.dataset_id.nunique(), **stats})
    return pd.DataFrame(rows)

def evaluate(smoke):
    nuis_run = completed(PACKAGE, 'nuisances_smoke' if smoke else 'nuisances')
    run = Run(PACKAGE, 'analyze_smoke' if smoke else 'analyze', smoke=smoke, gpu=False, threads=4, harm_threshold=HARM)
    try:
        nuis = pd.read_parquet(nuis_run / 'nuisances.parquet')
        frame = add_scores(pd.concat([observed_rows(nuis)] + ([] if smoke else [acic_rows()]), ignore_index=True))
        calib = frame[frame.cohort.eq('E2_fresh_in_prior') & frame.split.eq('threshold')].drop_duplicates(['dataset_id', 'source_row_index', 'a_query'])
        t_support = float(np.quantile(calib.pi_hat_query, 0.10))
        t_g1 = float(np.quantile(calib.g1_gap, 0.90))
        frame = apply_thresholds(frame, t_support, t_g1)
        frame['group'] = np.where(frame.cohort.eq('ACIC'), 'ACIC_' + frame.stratum, frame.cohort + ' ' + frame.stratum)
        signals = ['regime', 'support', 'g1', 'any']
        assess = frame[frame.split.eq('assessment')]
        out = dict(
            thresholds=pd.DataFrame([dict(t_support=t_support, t_g1=t_g1, calibration_query_arms=len(calib),
                                          harm_threshold=HARM)]),
            assessment_by_group=table(assess, ['group'], signals),
            threshold_split_by_group=table(frame[frame.split.eq('threshold')], ['group'], signals),
            assessment_pooled=table(assess.assign(pool=np.where(assess.cohort.eq('E2_fresh_in_prior'), 'in_prior',
                                                    np.where(assess.cohort.eq('ACIC'), 'realistic_ACIC', 'OOD'))), ['pool'], signals, boot=not smoke),
            score_auc=auc_table(assess),
        )
        EXPORT.mkdir(parents=True, exist_ok=True)
        tag = '_smoke' if smoke else ''
        for name, t in out.items():
            t.to_csv((run.path if smoke else EXPORT) / f'{name}{tag}.csv', index=False)
        frame.to_parquet(run.path / 'rows.parquet', index=False)
        if not smoke:
            plot(assess)
        print(out['thresholds'].to_string())
        print(out['assessment_pooled'][['pool', 'signal', 'rows', 'flag_rate', 'harmful_rate', 'recall', 'false_alarm', 'harmful_rate_retained', 'containment_retained', 'containment_all']].round(3).to_string())
        run.done(rows=len(frame), t_support=t_support, t_g1=t_g1)
    except BaseException:
        run.done('failed')
        raise

def auc(score, label):
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))

def auc_table(frame):
    rows = []
    for g, part in frame.groupby('group'):
        rows.append(dict(group=g, rows=len(part), harmful=int(part.harmful.sum()),
                         auc_g1_gap=auc(part.g1_gap, part.harmful), auc_low_support=auc(-part.pi_hat_query, part.harmful)))
    return pd.DataFrame(rows)

def plot(assess):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = sorted(assess.group.unique())
    for s, color, off in (('g1', '#2b6cb0', -0.2), ('support', '#2f855a', 0.0), ('any', '#c05621', 0.2)):
        st = [signal_stats(assess[assess.group.eq(g)], f'flag_{s}') for g in groups]
        axes[0].barh(np.arange(len(groups)) + off, [x['recall'] for x in st], height=0.2, color=color, label=s)
        axes[1].barh(np.arange(len(groups)) + off, [x['flag_rate'] for x in st], height=0.2, color=color, label=s)
    harm = [assess[assess.group.eq(g)].harmful.mean() for g in groups]
    axes[1].scatter(harm, np.arange(len(groups)), color='black', marker='|', s=200, label='harmful rate')
    for ax, title in zip(axes, ('recall of harmful failures', 'flag rate (| = harmful rate)')):
        ax.set_yticks(np.arange(len(groups)), groups, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXPORT / 'signals_by_group.png', dpi=130)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['nuisances', 'evaluate'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    {'nuisances': fit_nuisances, 'evaluate': evaluate}[args.stage](args.smoke)

if __name__ == '__main__':
    main()
