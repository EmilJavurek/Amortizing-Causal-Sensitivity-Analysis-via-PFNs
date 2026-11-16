"""Outward interval containment, width/severity summaries, and figures."""

from __future__ import annotations

from experiments.shared.paths import REPORTS
import matplotlib
matplotlib.use('Agg')
from experiments.shared.paths import OBSERVED_RUNS
import argparse
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
from experiments.shared.runtime import RUNS, Run, cluster_bootstrap, write
import matplotlib.pyplot as plt
from experiments.shared.runtime import RUNS, read

"""Outward q05(lower) / q95(upper) intervals from freshly saved PI-FM predictions.

Data flow: saved prediction fragments -> per-row reference pairs (MSM exact grid; KL paired on common
attained-Gamma support) -> per-row containment/width/severity -> per-DGP sums -> whole-DGP
bootstrap summaries, stratified tables, figures. No new inference.

The interval [lower_q05, upper_q95] combines two per-head marginal predictive quantiles. Each
one-sided statement is nominally 95%; joint containment of the reference set [theta-, theta+] is
not guaranteed at 90% or any level.
"""


pa.set_cpu_count(2)
pa.set_io_thread_count(2)
EXPORT = REPORTS / 'outward_intervals'
PACKAGE = 'e02_outward_intervals'
WIDTH_EPS = 1e-12
KEEP = ['dataset_id', 'source_row_index', 'a_query', 'gamma', 'p_a1', 'pi_query', 'sigma_obs',
        'lower_q05', 'upper_q95', 'lower_mean', 'upper_mean', 'lower_q50', 'upper_q50']
OVERLAP_BINS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5000001]
KL_GAMMA_BINS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]

def load(pattern, limit=None):
    paths = sorted(glob.glob(str(OBSERVED_RUNS / pattern)))
    assert paths, pattern
    return pd.concat([pd.read_parquet(p) for p in paths[:limit]], ignore_index=True)

def msm_cohorts(limit):
    """MSM cohorts: both reference heads live on the same exact target-Gamma grid."""
    out = []
    e2 = load('e2/fragments/dgp_*.parquet', limit)
    out.append(e2.assign(cohort='E2_fresh_in_prior', stratum='all'))
    e9 = load('e9/fragments/*.parquet', None if limit is None else 4 * limit)
    out.append(e9.assign(cohort='E9_' + e9.family, stratum=e9.gamma_region))
    e7c = load('e7_context/fragments/dgp_*.parquet', limit)
    out.append(e7c.assign(cohort='E7_context', stratum='n=' + e7c.context_length.astype(str)))
    e7d = load('e7_dimension/fragments/dim_*.parquet', None if limit is None else 5 * limit)
    out.append(e7d.assign(cohort='E7_dimension', stratum='d=' + e7d.covariate_dimension.astype(str)))
    frames = []
    for frame in out:
        frame = frame.rename(columns={'theta_lower': 'ref_lower', 'theta_upper': 'ref_upper'})
        frame = frame[KEEP + ['cohort', 'stratum', 'ref_lower', 'ref_upper']].copy()
        frame['sensitivity_model'] = 'msm'
        # Cluster key: Dimension cohorts reuse integer IDs for different DGPs per dimension.
        prefix = frame.stratum + ':' if frame.cohort.iloc[0] == 'E7_dimension' else ''
        frame['dataset_id'] = prefix + frame.dataset_id.astype(str)
        frame['anchor_head'] = 'both'
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def kl_pairs(limit):
    """Pair KL reference heads on common numerical Gamma support (no extrapolation).

    Each KL reference row is a selected reference point of one head at its attained Gamma; the PI-FM was
    queried at that Gamma for both heads. The opposite head's reference curve is linearly
    interpolated at that Gamma only if it lies inside the opposite head's attained-Gamma range.
    """
    e1 = load('e1/fragments/dgp_*.parquet', limit)
    keys = ['dataset_id', 'source_row_index', 'a_query']
    paired, audit = [], []
    for key, group in e1.groupby(keys, sort=True):
        curves = {}
        for head in ('lower', 'upper'):
            # Same convention as curve interpolation: tied attained Gammas keep the extreme value,
            # then the monotone envelope in Gamma order is interpolated inside its own support.
            curve = group[group.bound_type.eq(head)]
            raw = curve.groupby('gamma_reference').theta_reference.agg('max' if head == 'upper' else 'min').sort_index()
            g = raw.index.to_numpy()
            t = np.maximum.accumulate(raw.to_numpy()) if head == 'upper' else np.minimum.accumulate(raw.to_numpy())
            assert len(g) >= 2, (key, head)
            curves[head] = (g, t)
            sign = 1 if head == 'upper' else -1
            ordered = curve.sort_values('gamma_reference').theta_reference.to_numpy()
            audit.append(dict(zip(keys, key), head=head, points=len(curve), distinct_gammas=len(g), gamma_min=g[0],
                              gamma_max=g[-1], raw_monotone=bool(np.all(sign * np.diff(ordered) >= -1e-9))))
        for head, other in (('lower', 'upper'), ('upper', 'lower')):
            rows = group[group.bound_type.eq(head)].copy()
            g_other, t_other = curves[other]
            inside = (rows.gamma_reference >= g_other[0]) & (rows.gamma_reference <= g_other[-1])
            rows['supported'] = inside
            rows[f'ref_{head}'] = rows.theta_reference
            rows[f'ref_{other}'] = np.where(inside, np.interp(rows.gamma_reference, g_other, t_other), np.nan)
            rows['anchor_head'] = head
            paired.append(rows)
    frame = pd.concat(paired, ignore_index=True)
    assert np.allclose(frame.gamma, frame.gamma_reference)
    support = frame.groupby('anchor_head').supported.agg(['size', 'sum']).rename(columns={'size': 'rows', 'sum': 'supported'})
    frame = frame[frame.supported].copy()
    frame['dataset_id'] = frame.dataset_id.astype(str)
    frame['cohort'] = 'E1_KL_fresh'
    frame['stratum'] = 'anchor=' + frame.anchor_head
    frame['sensitivity_model'] = 'kl'
    return frame[KEEP + ['cohort', 'stratum', 'ref_lower', 'ref_upper', 'sensitivity_model', 'anchor_head']], support, pd.DataFrame(audit)

def row_metrics(frame):
    f = frame.copy()
    f['ref_width'] = f.ref_upper - f.ref_lower
    f['out_width'] = f.upper_q95 - f.lower_q05
    f['lower_ok'] = f.lower_q05 <= f.ref_lower
    f['upper_ok'] = f.upper_q95 >= f.ref_upper
    f['joint_ok'] = f.lower_ok & f.upper_ok
    f['miss_below'] = np.maximum(0.0, f.lower_q05 - f.ref_lower)
    f['miss_above'] = np.maximum(0.0, f.ref_upper - f.upper_q95)
    f['severity'] = f.miss_below + f.miss_above
    f['severity_original'] = f.severity * f.sigma_obs
    f['width_defined'] = f.ref_width > WIDTH_EPS
    defined = f.width_defined
    f['width_ratio'] = np.where(defined, f.out_width / f.ref_width.where(defined, 1.0), np.nan)
    f['severity_rel_width'] = np.where(defined, f.severity / f.ref_width.where(defined, 1.0), np.nan)
    f['excess_width'] = f.out_width - f.ref_width
    f['ref_crossed'] = f.ref_width < -1e-9
    f['overlap'] = np.minimum(f.p_a1, 1 - f.p_a1)
    f['overlap_bin'] = pd.cut(f.overlap, OVERLAP_BINS, right=False).astype(str)
    kl_bin = pd.cut(f.gamma, KL_GAMMA_BINS, right=False).astype(str)
    f['gamma_level'] = np.where(f.sensitivity_model.eq('kl'), 'KL' + kl_bin, f.gamma.round(3).astype(str))
    return f

SUMS = ['joint_ok', 'lower_ok', 'upper_ok', 'severity', 'severity_original', 'out_width', 'ref_width',
        'excess_width']

def per_dgp(rows, by):
    g = rows.groupby(by + ['dataset_id'], sort=True)
    out = g[SUMS].sum()
    out['rows'] = g.size()
    out['failures'] = out.rows - out.joint_ok
    out['joint_rate'] = out.joint_ok / out.rows
    out['max_severity'] = g.severity.max()
    out['median_width_ratio'] = g.width_ratio.median()
    return out.reset_index()

def rate_statistic(frame):
    n = frame.rows.sum()
    return dict(joint=frame.joint_ok.sum() / n, lower_one_sided=frame.lower_ok.sum() / n,
                upper_one_sided=frame.upper_ok.sum() / n, mean_severity=frame.severity.sum() / n,
                mean_severity_original=frame.severity_original.sum() / n,
                mean_out_width_over_mean_ref_width=frame.out_width.sum() / frame.ref_width.sum(),
                mean_excess_width=frame.excess_width.sum() / n)

def summaries(rows, by, replicates):
    dgp = per_dgp(rows, by)
    records = []
    for key, group in dgp.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        stats = cluster_bootstrap(group, 'dataset_id', rate_statistic, replicates=replicates)
        sub = rows[np.logical_and.reduce([rows[c].eq(v).to_numpy() for c, v in zip(by, key)])]
        fail = sub[~sub.joint_ok]
        record = dict(zip(by, key), rows=len(sub), dgps=group.dataset_id.nunique(), **stats,
                      median_width_ratio=float(sub.width_ratio.median()),
                      width_undefined_rows=int((~sub.width_defined).sum()),
                      failure_rows=len(fail),
                      failure_severity_median=float(fail.severity.median()) if len(fail) else 0.0,
                      failure_severity_p95=float(fail.severity.quantile(0.95)) if len(fail) else 0.0,
                      failure_severity_max=float(fail.severity.max()) if len(fail) else 0.0,
                      failure_severity_rel_width_median=float(fail.severity_rel_width.median()) if len(fail) else 0.0,
                      dgp_joint_rate_min=float(group.joint_rate.min()),
                      dgp_joint_rate_q10=float(group.joint_rate.quantile(0.10)),
                      dgp_joint_rate_median=float(group.joint_rate.median()),
                      dgps_below_0p5=int((group.joint_rate < 0.5).sum()),
                      dgps_all_contained=int((group.joint_rate == 1).sum()))
        records.append(record)
    return pd.DataFrame(records), dgp

def stratified(rows, extra):
    by = ['cohort', 'stratum', extra]
    g = rows.groupby(by, sort=True, observed=True)
    out = g[['joint_ok', 'lower_ok', 'upper_ok', 'severity']].mean()
    out['rows'] = g.size()
    out['median_width_ratio'] = g.width_ratio.median()
    out['mean_out_width'] = g.out_width.mean()
    out['mean_ref_width'] = g.ref_width.mean()
    return out.reset_index().rename(columns={'joint_ok': 'joint', 'lower_ok': 'lower_one_sided',
                                             'upper_ok': 'upper_one_sided', 'severity': 'mean_severity'})

def validate(rows):
    checks = {}
    checks['finite_quantities'] = bool(np.isfinite(rows[['lower_q05', 'upper_q95', 'ref_lower', 'ref_upper']].to_numpy()).all())
    checks['reference_crossings'] = int(rows.ref_crossed.sum())
    checks['predictive_quantile_crossing_rows'] = int((rows.out_width < 0).sum())
    msm = rows[rows.sensitivity_model.eq('msm')]
    gamma_one = msm[np.isclose(msm.gamma, 1.0)]
    checks['msm_gamma1_max_ref_width'] = float(gamma_one.ref_width.abs().max())
    checks['joint_implies_one_sided'] = bool((~rows.joint_ok | (rows.lower_ok & rows.upper_ok)).all())
    checks['severity_zero_iff_joint'] = bool(((rows.severity == 0) == rows.joint_ok).all())
    assert checks['finite_quantities'] and checks['joint_implies_one_sided'] and checks['severity_zero_iff_joint']
    checks['kl_reference_crossings'] = int(rows.ref_crossed[rows.sensitivity_model.eq('kl')].sum())
    assert int(rows.ref_crossed[rows.sensitivity_model.eq('msm')].sum()) == 0, checks
    assert checks['msm_gamma1_max_ref_width'] < 1e-6, checks
    return checks

def analyze(smoke):
    run = Run(PACKAGE, 'smoke' if smoke else 'retained', smoke=smoke, gpu=False, threads=4)
    limit = 2 if smoke else None
    replicates = 50 if smoke else 2000
    try:
        msm = msm_cohorts(limit)
        kl, kl_support, kl_audit = kl_pairs(limit)
        rows = row_metrics(pd.concat([msm, kl], ignore_index=True))
        checks = validate(rows)
        rows.to_parquet(run.path / 'rows.parquet', index=False)
        summary, dgp = summaries(rows, ['cohort', 'stratum'], replicates)
        cohort_summary, _ = summaries(rows, ['cohort'], replicates)
        dgp.to_parquet(run.path / 'per_dgp.parquet', index=False)
        by_gamma = stratified(rows, 'gamma_level')
        by_overlap = stratified(rows, 'overlap_bin')
        worst = (dgp.sort_values(['cohort', 'stratum', 'joint_rate', 'max_severity'], ascending=[True, True, True, False])
                 .groupby(['cohort', 'stratum']).head(5))
        target = run.path / 'export' if smoke else EXPORT
        target.mkdir(exist_ok=True)
        summary.to_csv(target / 'summary_by_stratum.csv', index=False)
        cohort_summary.to_csv(target / 'summary_by_cohort.csv', index=False)
        by_gamma.to_csv(target / 'by_gamma.csv', index=False)
        by_overlap.to_csv(target / 'by_overlap.csv', index=False)
        worst.to_csv(target / 'worst_dgps.csv', index=False)
        dgp.to_csv(target / 'per_dgp.csv', index=False)
        kl_support.reset_index().to_csv(target / 'kl_pairing_support.csv', index=False)
        kl_audit.groupby('head')[['points', 'distinct_gammas', 'gamma_min', 'gamma_max']].describe().to_csv(target / 'kl_curve_support_audit.csv')
        checks['kl_reference_curve_raw_nonmonotone_curves'] = int((~kl_audit.raw_monotone).sum())
        checks['kl_reference_curves'] = len(kl_audit)
        checks['kl_pairing_support'] = kl_support.reset_index().to_dict('records')
        write(target / 'validation.json', checks)
        run.manifest.update(rows=len(rows), cohorts=sorted(rows.cohort.unique()), bootstrap_replicates=replicates,
                            row_parquet=str(run.path / 'rows.parquet'), checks=checks)
        run.done(rows=len(rows))
    except BaseException:
        run.done('failed')
        raise


FIG = REPORTS / 'outward_intervals' / 'figures'
RUN = RUNS / 'e02_outward_intervals' / 'retained'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e6e6e6'
BLUE, ORANGE, GREEN, RED = '#2f6db3', '#d9822b', '#3a923a', '#c23b3b'

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)

def per_dgp_histograms(dgp):
    panels = [('E2_fresh_in_prior', 'all', 'MSM in-prior (512 DGPs)'), ('E1_KL_fresh', None, 'KL in-prior (64 DGPs)'),
              ('E9_heavy_tailed_outcome', None, 'E9 heavy-tailed'), ('E9_oscillatory_outcome', None, 'E9 oscillatory'),
              ('E9_extreme_overlap', None, 'E9 extreme overlap'), ('E9_gamma_extrapolation', 'extrapolated', 'E9 Γ>5 extrapolated')]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for ax, (cohort, stratum, title) in zip(axes.ravel(), panels):
        sub = dgp[dgp.cohort.eq(cohort)]
        if stratum is not None:
            sub = sub[sub.stratum.eq(stratum)]
        rate = sub.groupby('dataset_id').joint_ok.sum() / sub.groupby('dataset_id').rows.sum()
        ax.hist(rate, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor='white')
        ax.axvline(rate.mean(), color=INK, linewidth=1, linestyle='--')
        ax.set_title(f'{title}\nmean {rate.mean():.3f}, min {rate.min():.2f}', fontsize=9, color=INK)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('Per-DGP joint containment rate', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('DGPs', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'per_dgp_joint_containment.png', dpi=130)
    plt.close(fig)

def by_gamma():
    table = pd.read_csv(REPORTS / 'outward_intervals' / 'by_gamma.csv')
    msm = table[~table.gamma_level.astype(str).str.startswith('KL')].copy()
    msm['gamma'] = msm.gamma_level.astype(float)
    series = [('E2_fresh_in_prior', 'all', 'In-prior MSM', BLUE), ('E9_heavy_tailed_outcome', 'all', 'E9 heavy-tailed', ORANGE),
              ('E9_oscillatory_outcome', 'all', 'E9 oscillatory', GREEN), ('E9_extreme_overlap', 'all', 'E9 extreme overlap', RED),
              ('E9_gamma_extrapolation', None, 'E9 Γ-extrapolation', MUTED)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for cohort, stratum, label, color in series:
        sub = msm[msm.cohort.eq(cohort)]
        if stratum is not None and stratum in set(sub.stratum):
            sub = sub[sub.stratum.eq(stratum)]
        sub = sub.groupby('gamma').apply(lambda g: pd.Series(dict(joint=np.average(g.joint, weights=g.rows),
                                                                    width=np.average(g.median_width_ratio, weights=g.rows))),
                                         include_groups=False).reset_index()
        axes[0].plot(sub.gamma, sub.joint, color=color, label=label, linewidth=1.6)
        positive = sub[sub.gamma > 1]
        axes[1].plot(positive.gamma, positive.width, color=color, label=label, linewidth=1.6)
    axes[0].axhline(0.9, color=INK, linewidth=0.8, linestyle=':')
    axes[0].axvline(5, color=MUTED, linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Joint containment of [θ⁻, θ⁺]', fontsize=8)
    axes[1].set_ylabel('Median outward width / reference width', fontsize=8)
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Γ (MSM)', fontsize=8)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'by_gamma_msm.png', dpi=130)
    plt.close(fig)

def context_dimension():
    table = pd.read_csv(REPORTS / 'outward_intervals' / 'summary_by_stratum.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, cohort, prefix, xlabel in ((axes[0], 'E7_context', 'n=', 'Context rows (trained at 1024)'),
                                       (axes[1], 'E7_dimension', 'd=', 'Covariate dimension (trained at 10)')):
        sub = table[table.cohort.eq(cohort)].copy()
        sub['x'] = sub.stratum.str.replace(prefix, '').astype(int)
        sub = sub.sort_values('x')
        ax.errorbar(sub.x, sub.joint, yerr=[sub.joint - sub.joint_lo, sub.joint_hi - sub.joint], color=BLUE,
                    marker='o', label='joint', linewidth=1.5, capsize=2)
        ax.plot(sub.x, sub.lower_one_sided, color=ORANGE, marker='s', linewidth=1, label='lower one-sided')
        ax.plot(sub.x, sub.upper_one_sided, color=GREEN, marker='^', linewidth=1, label='upper one-sided')
        ax.axhline(0.9, color=INK, linewidth=0.8, linestyle=':')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sub.x)
        ax.set_xticklabels(sub.x)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylim(0.3, 1.02)
        style(ax)
    axes[0].set_ylabel('Containment rate', fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'context_dimension.png', dpi=130)
    plt.close(fig)

def severity(rows):
    fail = rows[~rows.joint_ok]
    groups = [('E2_fresh_in_prior', 'In-prior MSM'), ('E1_KL_fresh', 'In-prior KL'), ('E9_heavy_tailed_outcome', 'E9 heavy'),
              ('E9_oscillatory_outcome', 'E9 oscill.'), ('E9_extreme_overlap', 'E9 overlap'), ('E9_gamma_extrapolation', 'E9 Γ-extrap.'),
              ('E7_context', 'E7 context'), ('E7_dimension', 'E7 dimension')]
    data = [fail[fail.cohort.eq(c)].severity.to_numpy() for c, _ in groups]
    rel = [fail[fail.cohort.eq(c) & fail.width_defined].severity_rel_width.to_numpy() for c, _ in groups]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, values, label in ((axes[0], data, 'Miss distance (observed-std. units)'),
                              (axes[1], rel, 'Miss distance / reference width')):
        ax.boxplot([np.maximum(v, 1e-6) for v in values], whis=(5, 95), showfliers=False, widths=0.6,
                   medianprops=dict(color=RED), boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([g[1] for g in groups], rotation=35, ha='right', fontsize=7)
        ax.set_yscale('log')
        ax.set_ylabel(label, fontsize=8)
        style(ax)
    fig.suptitle('Failure severity among non-contained rows (box 25–75%, whiskers 5–95%)', fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'failure_severity.png', dpi=130)
    plt.close(fig)

def plot():
    assert read(RUN / 'manifest.json')['status'] == 'completed'
    FIG.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(RUN / 'rows.parquet')
    dgp = pd.read_parquet(RUN / 'per_dgp.parquet')
    per_dgp_histograms(dgp)
    by_gamma()
    context_dimension()
    severity(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analyze', 'plot'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'plot':
        assert not args.smoke, 'Plots use the completed full-cohort analysis'
        plot()
    else:
        analyze(args.smoke)

if __name__ == '__main__':
    main()
