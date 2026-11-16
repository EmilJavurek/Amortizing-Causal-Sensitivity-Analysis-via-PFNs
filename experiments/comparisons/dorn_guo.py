"""Dorn–Guo quantile balancing for MSM APO/ATE bounds.

The LP and Lemma 1 implementations are independent numerical cross-checks.
Matched reports average arm-specific CAPO references over empirical context covariates."""

from __future__ import annotations

from experiments.shared.paths import REPORTS

from experiments.shared.paths import CODE
import numpy as np
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.model_selection import KFold
import argparse
import os
import time
import pandas as pd
from experiments.shared.runtime import Run, cluster_bootstrap, completed, read, write
from experiments.comparisons.benchmark import GAMMAS, load_tasks, pfn_outputs
from scipy.special import expit
from scipy.stats import norm
from experiments.shared.msm_reference import msm_reference_from_samples

def tau_of(lam):
    return lam / (1.0 + lam)

def weight_box(e, lam):
    odds_inv = (1.0 - e) / e
    return 1.0 + odds_inv / lam, 1.0 + lam * odds_inv

def psi_T_lp(y, z, e, q, lam, upper):
    """QB bound for E[Y(1)] (treated arm = z == 1) via the linear program in w = 1/ebar."""
    t = z == 1
    yt, et, qt = y[t], e[t], q[t]
    if lam == 1.0:
        w = 1.0 / et
        return float(np.sum(yt * w) / np.sum(w))
    lo, hi = weight_box(et, lam)
    g = np.vstack([np.ones_like(qt), qt])
    rhs = g @ (1.0 / et)
    sign = -1.0 if upper else 1.0
    res = linprog(sign * yt, A_eq=g, b_eq=rhs, bounds=np.c_[lo, hi], method='highs')
    assert res.status == 0, res.message
    return float(np.sum(yt * res.x) / np.sum(1.0 / et))

def psi_T_lemma(y, z, e, q, lam, upper):
    """Same optimum via Lemma 1: weighted tau-quantile regression of Y on g(X) among treated."""
    if not upper:
        return -psi_T_lemma(-y, z, e, -q, lam, True)
    t = z == 1
    yt, et, qt = y[t], e[t], q[t]
    tau = tau_of(lam)
    g = np.c_[np.ones_like(qt), qt]
    model = QuantileRegressor(quantile=tau, alpha=0.0, fit_intercept=False, solver='highs')
    model.fit(g, yt, sample_weight=(1 - et) / et)
    fit = g @ model.coef_
    v = np.sign(yt - fit)
    num = np.sum((yt - fit) * (1 + lam ** v * (1 - et) / et)) + np.sum(fit / et)
    return float(num / np.sum(1.0 / et))

def nominal_propensity(x, z, clip=1e-3):
    """Logistic regression (sklearn default L2, C=1, on the standardized context covariates)."""
    model = LogisticRegression(max_iter=5000).fit(x, z)
    return np.clip(model.predict_proba(x)[:, 1], clip, 1 - clip)

def quantile_forest(x_train, y_train, x_pred, levels, seed, n_estimators=300, min_samples_leaf=10):
    """Meinshausen (2006) quantile regression forest: leaf co-membership weights on training Y."""
    forest = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                   max_features=1 / 3, random_state=seed, n_jobs=1).fit(x_train, y_train)
    train_leaves = forest.apply(x_train)
    pred_leaves = forest.apply(x_pred)
    order = np.argsort(y_train)
    ys = y_train[order]
    weights = np.zeros((len(x_pred), len(x_train)))
    for tree in range(train_leaves.shape[1]):
        same = pred_leaves[:, [tree]] == train_leaves[None, :, tree]
        weights += same / same.sum(axis=1, keepdims=True)
    weights = weights[:, order] / train_leaves.shape[1]
    cdf = np.cumsum(weights, axis=1)
    out = np.empty((len(x_pred), len(levels)))
    for k, level in enumerate(levels):
        out[:, k] = ys[np.minimum((cdf < level - 1e-12).sum(axis=1), len(ys) - 1)]
    return out

def crossfit_quantiles(x, y, arm_mask, levels, learner, folds=5, seed=0):
    """Cross-fitted conditional quantiles Q_t(x, arm) for every row (fit on the arm's rows only)."""
    out = np.empty((len(x), len(levels)))
    for k, (train, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(x)):
        tr = train[arm_mask[train]]
        if learner == 'linear':
            for j, level in enumerate(levels):
                model = QuantileRegressor(quantile=level, alpha=0.0, solver='highs').fit(x[tr], y[tr])
                out[test, j] = model.predict(x[test])
        else:
            out[test] = quantile_forest(x[tr], y[tr], x[test], levels, seed=seed + k)
    return out

def qb_bounds(x, z, y, lambdas, learner, seed=0):
    """APO and ATE bounds for each Lambda. Propensity and quantile fits are shared across Lambda."""
    e = nominal_propensity(x, z)
    levels = sorted({0.5} | {tau_of(l) for l in lambdas} | {1 - tau_of(l) for l in lambdas})
    index = {round(level, 12): j for j, level in enumerate(levels)}
    q = {arm: crossfit_quantiles(x, y, z == arm, levels, learner, seed=seed + 10 * arm) for arm in (0, 1)}
    rows = []
    for lam in lambdas:
        tau = tau_of(lam)
        hi, lo = index[round(tau, 12)], index[round(1 - tau, 12)]
        t_up = psi_T_lp(y, z, e, q[1][:, hi], lam, True)
        t_lo = psi_T_lp(y, z, e, q[1][:, lo], lam, False)
        c_up = psi_T_lp(y, 1 - z, 1 - e, q[0][:, hi], lam, True)
        c_lo = psi_T_lp(y, 1 - z, 1 - e, q[0][:, lo], lam, False)
        rows.append(dict(gamma=lam, apo1_lower=t_lo, apo1_upper=t_up, apo0_lower=c_lo, apo0_upper=c_up,
                         ate_lower=t_lo - c_up, ate_upper=t_up - c_lo))
    return rows, dict(e_min=float(e.min()), e_max=float(e.max()))

PACKAGE = 'e09_dorn_guo'

LEARNERS = ('linear', 'forest')

WORKERS = 6

def one_task(args):
    task_id, x, z, y, learner = args
    os.environ['OMP_NUM_THREADS'] = '1'
    start = time.perf_counter()
    rows, diag = qb_bounds(x, z, y, GAMMAS, learner, seed=0)
    seconds = time.perf_counter() - start
    return [dict(task_id=task_id, method=f'qb_{learner}', **r) for r in rows], dict(
        task_id=task_id, method=f'qb_{learner}', fit_plus_query_seconds=seconds, **diag)

def run(smoke, cohort_set='default'):
    from concurrent.futures import ProcessPoolExecutor
    tasks = load_tasks(smoke, cohort_set=cohort_set)
    if smoke:
        tasks = [t for t in tasks if t.task_id in ('syn_1', 'acic_d10_rep0', 'acic_full_rep0', 'ihdp_rep0')]
    suffix = '' if cohort_set == 'default' else f'_{cohort_set}'
    r = Run(PACKAGE, ('qb_smoke' if smoke else 'qb') + suffix, smoke=smoke, cohort_set=cohort_set, gpu=False, threads=1,
            workers=WORKERS, learners=LEARNERS, gammas=GAMMAS, tasks=[t.task_id for t in tasks])
    jobs = [(t.task_id, *t.observed(), learner) for t in tasks for learner in LEARNERS]
    bounds, timing = [], []
    try:
        with ProcessPoolExecutor(WORKERS) as pool:
            for rows, tim in pool.map(one_task, jobs):
                bounds += rows
                timing.append(tim)
                print(tim, flush=True)
        pd.DataFrame(bounds).to_parquet(r.path / 'qb_bounds.parquet', index=False)
        pd.DataFrame(timing).to_csv(r.path / 'timing.csv', index=False)
        r.done(rows=len(bounds))
    except BaseException:
        r.done('failed')
        raise

def group_of(task_id):
    return 'synthetic' if task_id.startswith('syn_') else ('acic_full' if '_full_' in task_id else 'acic_d10')

def long_table(tasks, qb_bounds):
    """Rows: task, method, estimand (apo0, apo1, ate), gamma, lower/upper, references, truth."""
    refs = []
    for t in tasks:
        a = t.apo.set_index(['a_query', 'gamma'])
        for g in GAMMAS:
            r1, r0 = a.loc[(1, g)], a.loc[(0, g)]
            true1, true0 = r1.get('apo_true', np.nan), r0.get('apo_true', np.nan)
            refs += [dict(task_id=t.task_id, estimand='apo1', gamma=g, ref_lower=r1.apo_lower, ref_upper=r1.apo_upper, true=true1),
                     dict(task_id=t.task_id, estimand='apo0', gamma=g, ref_lower=r0.apo_lower, ref_upper=r0.apo_upper, true=true0),
                     dict(task_id=t.task_id, estimand='ate', gamma=g, ref_lower=r1.apo_lower - r0.apo_upper,
                          ref_upper=r1.apo_upper - r0.apo_lower, true=true1 - true0)]
    refs = pd.DataFrame(refs)
    _, _, pfn_apo = pfn_outputs(tasks)
    p = pfn_apo.set_index(['task_id', 'a_query', 'gamma'])
    preds = []
    for t in tasks:
        for g in GAMMAS:
            l1, u1 = p.loc[(t.task_id, 1, g)][['lower', 'upper']]
            l0, u0 = p.loc[(t.task_id, 0, g)][['lower', 'upper']]
            preds += [dict(task_id=t.task_id, method='pifm', estimand='apo1', gamma=g, lower=l1, upper=u1),
                      dict(task_id=t.task_id, method='pifm', estimand='apo0', gamma=g, lower=l0, upper=u0),
                      dict(task_id=t.task_id, method='pifm', estimand='ate', gamma=g, lower=l1 - u0, upper=u1 - l0)]
    for _, row in qb_bounds.iterrows():
        for est, lo, up in (('apo1', 'apo1_lower', 'apo1_upper'), ('apo0', 'apo0_lower', 'apo0_upper'), ('ate', 'ate_lower', 'ate_upper')):
            preds.append(dict(task_id=row.task_id, method=row.method, estimand=est, gamma=row.gamma, lower=row[lo], upper=row[up]))
    out = pd.DataFrame(preds).merge(refs, on=['task_id', 'estimand', 'gamma'], validate='many_to_one')
    out['group'] = out.task_id.map(group_of)
    out['err_lower'] = out.lower - out.ref_lower
    out['err_upper'] = out.upper - out.ref_upper
    out['covers_ref'] = (out.lower <= out.ref_lower + 1e-9) & (out.upper >= out.ref_upper - 1e-9)
    out['width'] = out.upper - out.lower
    out['ref_width'] = out.ref_upper - out.ref_lower
    out['true_in'] = np.where(out['true'].notna(), (out.lower <= out['true']) & (out['true'] <= out.upper), np.nan)
    return out

def statistic(frame):
    out = dict(rmse=float(np.sqrt(np.mean(np.r_[frame.err_lower, frame.err_upper] ** 2))),
               mean_signed_lower=float(frame.err_lower.mean()), mean_signed_upper=float(frame.err_upper.mean()),
               covers_ref=float(frame.covers_ref.mean()), width=float(frame.width.mean()), ref_width=float(frame.ref_width.mean()))
    if frame['true'].notna().all():
        out['true_in'] = float(frame.true_in.astype(float).mean())
    return out

def report(smoke):
    source = completed(PACKAGE, 'qb_smoke' if smoke else 'qb')
    assert source is not None
    qbb = pd.read_parquet(source / 'qb_bounds.parquet')
    tasks = [t for t in load_tasks(smoke) if t.task_id in set(qbb.task_id)]
    r = Run(PACKAGE, 'report_smoke' if smoke else 'report', smoke=smoke, gpu=False, threads=4, source=str(source))
    export = REPORTS / ('smoke/dorn_guo' if smoke else 'dorn_guo')
    export.mkdir(parents=True, exist_ok=True)
    try:
        table = long_table(tasks, qbb)
        qbrows = table[table.method.ne('pifm')]
        assert np.all(qbrows.lower <= qbrows.upper + 1e-9), 'QB bounds must be ordered'
        pf = table[table.method.eq('pifm')]
        crossing = pf[pf.lower > pf.upper + 1e-9].groupby(['group', 'estimand', 'gamma']).size().rename('pifm_crossed_rows').reset_index()
        crossing.to_csv(export / 'pifm_crossing.csv', index=False)  # raw separate-head means may cross; retained
        assert table.groupby(['task_id', 'method', 'estimand', 'gamma']).size().max() == 1
        table.to_parquet(r.path / 'rows.parquet', index=False)
        summ = []
        for (group, method, est), part in table.groupby(['group', 'method', 'estimand']):
            stats = cluster_bootstrap(part, 'task_id', statistic, replicates=200 if smoke else 2000)
            summ.append(dict(group=group, method=method, estimand=est, tasks=part.task_id.nunique(), **stats))
        summ = pd.DataFrame(summ)
        by_gamma = pd.DataFrame([dict(group=k[0], method=k[1], estimand=k[2], gamma=k[3], **statistic(f))
                                 for k, f in table.groupby(['group', 'method', 'estimand', 'gamma'])])
        per_task = table[table.estimand.eq('ate')][['group', 'task_id', 'method', 'gamma', 'lower', 'upper', 'ref_lower',
                                                    'ref_upper', 'true', 'covers_ref', 'true_in']]
        timing = pd.read_csv(source / 'timing.csv')
        timing['group'] = timing.task_id.map(group_of)
        tim = timing.groupby(['group', 'method']).fit_plus_query_seconds.describe()[['count', 'mean', '50%', 'max']].reset_index()
        # Paired PI-FM vs QB absolute endpoint error difference (ATE, Gamma > 1), whole-task bootstrap.
        paired = []
        ate = table[table.estimand.eq('ate')]
        for group, part in ate.groupby('group'):
            wide = part.assign(abs_err=(part.err_lower.abs() + part.err_upper.abs()) / 2).pivot_table(
                index=['task_id', 'gamma'], columns='method', values='abs_err').reset_index()
            for other in ('qb_linear', 'qb_forest'):
                diff = wide.assign(d=wide['pifm'] - wide[other])
                paired.append(dict(group=group, contrast=f'pifm_minus_{other}', **cluster_bootstrap(
                    diff, 'task_id', lambda f: dict(mean_abs_err_diff=float(f.d.mean())), replicates=200 if smoke else 2000)))
        paired = pd.DataFrame(paired)
        for name, frame in (('summary', summ), ('by_gamma', by_gamma), ('per_task_ate', per_task), ('timing', tim), ('paired_ate', paired)):
            frame.to_csv(export / f'{name}.csv', index=False)
        plot(by_gamma, export)
        write(export / 'provenance.json', dict(qb_run=str(source.relative_to(CODE)), report_run=str(r.path.relative_to(CODE)),
                                               validate_checks=read(completed(PACKAGE, 'validate') / 'checks.json') if completed(PACKAGE, 'validate') else None))
        print(summ.round(4).to_string())
        print(tim.round(2).to_string())
        print(paired.round(4).to_string())
        r.done(rows=len(table))
    except BaseException:
        r.done('failed')
        raise

def plot(by_gamma, export):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['synthetic', 'acic_d10', 'acic_full']
    colors = {'pifm': '#1f77b4', 'qb_linear': '#d62728', 'qb_forest': '#2ca02c'}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, group in enumerate(groups):
        part = by_gamma[(by_gamma.group == group) & (by_gamma.estimand == 'ate')]
        for method, f in part.groupby('method'):
            axes[0, j].plot(f.gamma, f.rmse, 'o-', color=colors[method], label=method)
            g = f[f.gamma > 1]  # reference width is 0 (up to float noise) at Gamma = 1
            axes[1, j].plot(g.gamma, g.width / g.ref_width, 'o-', color=colors[method], label=method)
        axes[0, j].set_title(f'{group}: ATE endpoint RMSE')
        axes[1, j].set_title(f'{group}: mean width / reference width')
        axes[1, j].axhline(1, color='grey', lw=0.8, ls='--')
        axes[1, j].set_xlabel('Gamma (Lambda)')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(export / 'ate_rmse_width_by_gamma.png', dpi=120)

def replot():
    plot(pd.read_csv(REPORTS / 'dorn_guo' / 'by_gamma.csv'), REPORTS / 'dorn_guo')

LAMBDAS = (1.0, 1.4, 2.0, 3.0, 5.0)

def sample(n, rng):
    x = rng.uniform(-1, 1, size=(n, 5))
    e = expit(x.sum(axis=1) / np.sqrt(5))
    z = rng.binomial(1, e)
    mu = x.sum(axis=1)
    return x, z, mu + rng.normal(size=n), e, mu

def sharp_truth(rng, m, k=1024):
    x = rng.uniform(-1, 1, size=(m, 5))
    e = expit(x.sum(axis=1) / np.sqrt(5))
    grid = norm.ppf((np.arange(k) + 0.5) / k)
    draws = x.sum(axis=1)[:, None] + grid[None, :]
    out = {}
    for arm, pi in ((1, e), (0, 1 - e)):
        lo, up = msm_reference_from_samples(draws, pi, np.array(LAMBDAS))
        out[arm] = (lo.mean(axis=0), up.mean(axis=0))
    return pd.DataFrame(dict(gamma=LAMBDAS, apo1_lower=out[1][0], apo1_upper=out[1][1],
                             apo0_lower=out[0][0], apo0_upper=out[0][1],
                             ate_lower=out[1][0] - out[0][1], ate_upper=out[1][1] - out[0][0]))

def oracle_bounds(n, rng):
    x, z, y, e, mu = sample(n, rng)
    rows = []
    for lam in LAMBDAS:
        tau = tau_of(lam)
        qh, ql = mu + norm.ppf(tau), mu + norm.ppf(1 - tau)
        t_up = psi_T_lp(y, z, e, qh, lam, True)
        t_lo = psi_T_lp(y, z, e, ql, lam, False)
        c_up = psi_T_lp(y, 1 - z, 1 - e, qh, lam, True)
        c_lo = psi_T_lp(y, 1 - z, 1 - e, ql, lam, False)
        rows.append(dict(gamma=lam, apo1_lower=t_lo, apo1_upper=t_up, apo0_lower=c_lo, apo0_upper=c_up,
                         ate_lower=t_lo - c_up, ate_upper=t_up - c_lo))
    return pd.DataFrame(rows)

def validate(smoke):
    run = Run('e09_dorn_guo', 'validate_smoke' if smoke else 'validate', smoke=smoke, gpu=False, threads=4)
    rng = np.random.default_rng(20261107)
    cols = ['apo1_lower', 'apo1_upper', 'apo0_lower', 'apo0_upper', 'ate_lower', 'ate_upper']
    try:
        truth = sharp_truth(rng, 2000 if smoke else 100000)
        checks = {}
        # 1-2. LP vs Lemma 1 and Lambda = 1 collapse on an estimated-nuisance sample.
        x, z, y, _, _ = sample(2000, rng)
        e = nominal_propensity(x, z)
        diffs = []
        for lam in LAMBDAS[1:]:
            for upper, level in ((True, tau_of(lam)), (False, 1 - tau_of(lam))):
                q = crossfit_quantiles(x, y, z == 1, [level], 'linear')[:, 0]
                diffs.append(abs(psi_T_lp(y, z, e, q, lam, upper) - psi_T_lemma(y, z, e, q, lam, upper)))
        checks['lp_vs_lemma_max_abs'] = max(diffs)
        ipw = np.sum(y * z / e) / np.sum(z / e)
        checks['lambda1_minus_stabilized_ipw'] = abs(psi_T_lp(y, z, e, np.zeros_like(y), 1.0, True) - ipw)
        assert checks['lp_vs_lemma_max_abs'] < 1e-6, checks
        assert checks['lambda1_minus_stabilized_ipw'] < 1e-12, checks
        # 3. Oracle nuisances at large n.
        oracle = oracle_bounds(20000 if smoke else 200000, rng)
        err = (oracle[cols] - truth[cols]).abs()
        checks['oracle_max_abs_error'] = float(err.to_numpy().max())
        oracle.assign(kind='oracle_nuisance').to_csv(run.path / 'oracle_vs_truth.csv', index=False)
        # 4. Estimated nuisances, QB-Linear, n = 1000.
        reps = 3 if smoke else 40
        est = []
        for r in range(reps):
            x, z, y, _, _ = sample(1000, rng)
            rows, _ = qb_bounds(x, z, y, LAMBDAS, 'linear', seed=r)
            est.append(pd.DataFrame(rows).assign(rep=r))
        est = pd.concat(est, ignore_index=True)
        mean = est.groupby('gamma')[cols].mean().reset_index()
        sd = est.groupby('gamma')[cols].std().reset_index()
        bias = (mean[cols] - truth[cols]).abs()
        checks['qb_linear_n1000_max_abs_bias'] = float(bias.to_numpy().max())
        checks['qb_linear_n1000_max_sd'] = float(sd[cols].to_numpy().max())
        pd.concat([truth.assign(kind='sharp_truth'), oracle.assign(kind='oracle_nuisance_n200k'),
                   mean.assign(kind=f'qb_linear_n1000_mean_{reps}reps'), sd.assign(kind=f'qb_linear_n1000_sd_{reps}reps')]
                  ).to_csv(run.path / 'validation_table.csv', index=False)
        write(run.path / 'checks.json', checks)
        print(pd.concat([truth.assign(kind='truth'), oracle.assign(kind='oracle'), mean.assign(kind='qb_lin')]).round(4).to_string())
        print(checks)
        assert checks['oracle_max_abs_error'] < (0.05 if smoke else 0.02), checks
        run.done(**checks)
    except BaseException:
        run.done('failed')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'report', 'replot', 'validate'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', default='default', choices=['default', 'ihdp'])
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.smoke, args.cohort)
    else:
        assert args.cohort == 'default', 'report/replot/validate cover the default cohort only'
        if args.stage == 'validate':
            validate(args.smoke)
        elif args.stage == 'report':
            report(args.smoke)
        else:
            assert not args.smoke, 'replot reads the retained report'
            replot()


if __name__ == '__main__':
    main()
