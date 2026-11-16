"""B-Learner MSM bounds: authors implementation, independent control and matched evaluation.

The authors estimator runs in its separate compatible environment. The independent control
reuses Gamma-independent nuisances; the authors estimator cold-fits each Gamma.
Unclipped propensity is primary; --clip is an explicitly recorded sensitivity control."""

from __future__ import annotations

from experiments.shared.paths import REPORTS

import sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import time
import pandas as pd
from experiments.shared.runtime import Run
from experiments.comparisons.benchmark import GAMMAS, load_tasks
from experiments.shared.runtime import completed, read
from experiments.comparisons.benchmark import group_of, join, references, row_metrics, summarize
from experiments.comparisons.benchmark import pfn_outputs
from scipy.special import expit
from scipy.stats import norm
from experiments.shared.msm_reference import msm_reference_from_samples

N_ESTIMATORS = 200

MAX_DEPTH = 6

MIN_SAMPLES_LEAF = 0.01

def rf(seed, n_jobs):
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                 n_jobs=n_jobs, random_state=seed)

def propensity_model():
    return LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)

class ForestKernel:
    """Meinshausen quantile-forest weights for a fitted squared-error random forest."""

    def __init__(self, x, y, seed, n_jobs):
        self.forest = rf(seed, n_jobs).fit(x, y)
        self.train_leaves = self.forest.apply(x)              # (n, trees)
        order = np.argsort(y, kind='stable')
        self.order = order
        self.sorted_y = y[order]

    def weights(self, x):
        """(m, n) weights over training rows, sorted by training outcome; rows sum to 1."""
        leaves = self.forest.apply(x)
        n, trees = self.train_leaves.shape
        out = np.zeros((len(x), n))
        for t in range(trees):
            train = self.train_leaves[:, t]
            ids, inverse, counts = np.unique(train, return_inverse=True, return_counts=True)
            position = np.searchsorted(ids, leaves[:, t])
            same = train[None, :] == leaves[:, t][:, None]
            out += same / counts[inverse][None, :]
            assert np.all(ids[np.clip(position, 0, len(ids) - 1)] == leaves[:, t])
        out /= trees
        return out[:, self.order]

def weighted_quantile(sorted_weights, sorted_y, tau):
    cumulative = np.cumsum(sorted_weights, axis=1)
    index = np.argmax(cumulative >= tau - 1e-12, axis=1)
    return sorted_y[index]

def r_targets(y, q_plus, q_minus, gamma):
    tau = gamma / (1 + gamma)
    r_plus = y / gamma + (1 - 1 / gamma) * (q_plus + np.maximum(y - q_plus, 0) / (1 - tau))
    r_minus = y / gamma + (1 - 1 / gamma) * (q_minus + np.minimum(y - q_minus, 0) / (1 - tau))
    return r_plus, r_minus

@dataclass
class FoldBase:
    train: np.ndarray
    test: np.ndarray
    prop: object
    mu: list
    kernels: list                 # ForestKernel per arm
    w_in: list                    # in-sample kernel weights per arm (train rows of that arm)
    w_test: list                  # out-of-fold kernel weights per arm
    rho_cache: dict = field(default_factory=dict)

class BLearner:
    def __init__(self, cv=5, seed=0, n_jobs=4, clip=None):
        """clip=None mirrors the authors (no propensity clipping); clip=c is a declared sensitivity variant."""
        self.cv, self.seed, self.n_jobs, self.clip = cv, seed, n_jobs, clip

    def _e(self, e):
        return e if self.clip is None else np.clip(e, self.clip, 1 - self.clip)

    def fit_base(self, x, a, y):
        """Gamma-independent nuisances: folds, propensity, mu_a, quantile-forest kernels."""
        self.x, self.a, self.y = x, a, y
        n = len(y)
        self.folds = []
        self.e = np.full(n, np.nan)
        self.mu = np.full((n, 2), np.nan)
        for k, (train, test) in enumerate(KFold(self.cv, shuffle=True, random_state=self.seed).split(x)):
            prop = propensity_model().fit(x[train], a[train])
            mus, kernels, w_in, w_test = [], [], [], []
            for arm in (0, 1):
                rows = train[a[train] == arm]
                mus.append(rf(self.seed + 10 * k + arm, self.n_jobs).fit(x[rows], y[rows]))
                kernel = ForestKernel(x[rows], y[rows], self.seed + 100 + 10 * k + arm, self.n_jobs)
                kernels.append(kernel)
                w_in.append(kernel.weights(x[rows]))
                w_test.append(kernel.weights(x[test]))
                self.mu[test, arm] = mus[-1].predict(x[test])
            self.e[test] = prop.predict_proba(x[test])[:, 1]
            self.folds.append(FoldBase(train, test, prop, mus, kernels, w_in, w_test))
        assert np.isfinite(self.e).all() and np.isfinite(self.mu).all()
        return self

    def fit_gamma(self, gamma):
        """Gamma-specific rho models, pseudo-outcomes and final-stage bound forests."""
        from econml.grf import RegressionForest
        tau = gamma / (1 + gamma)
        x, a, y = self.x, self.a, self.y
        n = len(y)
        q = {key: np.full(n, np.nan) for key in ('plus0', 'plus1', 'minus0', 'minus1')}
        rho = {key: np.full(n, np.nan) for key in ('plus0', 'plus1', 'minus0', 'minus1')}
        rho_models = []
        for k, fold in enumerate(self.folds):
            models = {}
            for arm in (0, 1):
                rows = fold.train[a[fold.train] == arm]
                kernel = fold.kernels[arm]
                q_plus_in = weighted_quantile(fold.w_in[arm], kernel.sorted_y, tau)
                q_minus_in = weighted_quantile(fold.w_in[arm], kernel.sorted_y, 1 - tau)
                r_plus, r_minus = r_targets(y[rows], q_plus_in, q_minus_in, gamma)
                for side, target in (('plus', r_plus), ('minus', r_minus)):
                    model = rf(self.seed + 1000 + 10 * k + arm + (5 if side == 'minus' else 0), self.n_jobs).fit(x[rows], target)
                    models[f'{side}{arm}'] = model
                    rho[f'{side}{arm}'][fold.test] = model.predict(x[fold.test])
                q[f'plus{arm}'][fold.test] = weighted_quantile(fold.w_test[arm], kernel.sorted_y, tau)
                q[f'minus{arm}'][fold.test] = weighted_quantile(fold.w_test[arm], kernel.sorted_y, 1 - tau)
            rho_models.append(models)
        e = self._e(self.e)
        phi = {}
        for side in ('plus', 'minus'):
            r1 = r_targets(y, q[f'{side}1'], q[f'{side}1'], gamma)[0 if side == 'plus' else 1]
            r0 = r_targets(y, q[f'{side}0'], q[f'{side}0'], gamma)[0 if side == 'plus' else 1]
            phi[f'{side}1'] = rho[f'{side}1'] + (a / e) * (r1 - rho[f'{side}1']) + a * y - a * r1
            phi[f'{side}0'] = rho[f'{side}0'] + (1 - a) / (1 - e) * (r0 - rho[f'{side}0']) + (1 - a) * y - (1 - a) * r0
        upper_target = phi['plus1'] - phi['minus0']
        lower_target = phi['minus1'] - phi['plus0']
        assert np.isfinite(upper_target).all() and np.isfinite(lower_target).all()
        final = {}
        for name, target in (('upper', upper_target), ('lower', lower_target)):
            final[name] = RegressionForest(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                           n_jobs=self.n_jobs, random_state=self.seed + (1 if name == 'upper' else 2)).fit(x, target)
        return GammaFit(gamma, self, rho_models, final)

@dataclass
class GammaFit:
    gamma: float
    base: BLearner
    rho_models: list
    final: dict

    def effect(self, x):
        """B-Learner CATE bounds (lower, upper)."""
        return self.final['lower'].predict(x).ravel(), self.final['upper'].predict(x).ravel()

    def plugin_capo(self, x):
        """Plug-in per-arm CAPO bounds from fold-averaged nuisances: dict arm -> (lower, upper)."""
        folds = self.base.folds
        e = self.base._e(np.mean([f.prop.predict_proba(x)[:, 1] for f in folds], axis=0))
        mu = [np.mean([f.mu[arm].predict(x) for f in folds], axis=0) for arm in (0, 1)]
        rho = {key: np.mean([m[key].predict(x) for m in self.rho_models], axis=0)
               for key in ('plus0', 'plus1', 'minus0', 'minus1')}
        return {1: (e * mu[1] + (1 - e) * rho['minus1'], e * mu[1] + (1 - e) * rho['plus1']),
                0: ((1 - e) * mu[0] + e * rho['minus0'], (1 - e) * mu[0] + e * rho['plus0'])}

WORKERS = 3

JOBS_PER_WORKER = 4

def reimplementation_task(args):
    task, n_jobs, clip = args
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    x, a, y = task.observed()
    query_ids, xq = task.query_x()
    start = time.perf_counter()
    learner = BLearner(cv=5, seed=0, n_jobs=n_jobs, clip=clip).fit_base(x, a, y)
    base_seconds = time.perf_counter() - start
    cate, capo, timing = [], [], []
    for g in GAMMAS:
        start = time.perf_counter()
        fit = learner.fit_gamma(g)
        fitted = time.perf_counter()
        lo, up = fit.effect(xq)
        plug = fit.plugin_capo(xq)
        done = time.perf_counter()
        cate.append(pd.DataFrame(dict(task_id=task.task_id, query_id=query_ids, gamma=g, lower=lo, upper=up)))
        for arm in (0, 1):
            capo.append(pd.DataFrame(dict(task_id=task.task_id, query_id=query_ids, a_query=arm, gamma=g,
                                          lower=plug[arm][0], upper=plug[arm][1])))
        timing.append(dict(task_id=task.task_id, gamma=g, base_fit_seconds=base_seconds, gamma_fit_seconds=fitted - start,
                           query_seconds=done - fitted, n_context=len(y), d=x.shape[1],
                           min_e=float(learner.e.min()), max_e=float(learner.e.max())))
    return pd.concat(cate), pd.concat(capo), pd.DataFrame(timing)

def run_reimplementation(smoke, clip):
    name = ('cohort_smoke' if smoke else 'cohort') + ('' if clip is None else f'_clip{clip:g}')
    run = Run('e08_blearner', name, smoke=smoke, gpu=False, propensity_clip=clip,
              threads=WORKERS * JOBS_PER_WORKER, workers=WORKERS, jobs_per_worker=JOBS_PER_WORKER, gammas=GAMMAS,
              implementation='independent reimplementation (authors code not executed)')
    tasks = load_tasks(smoke=smoke)
    if smoke:
        tasks = [t for t in tasks if t.task_id in ('syn_1', 'acic_full_rep0')]
    try:
        out = []
        with ProcessPoolExecutor(WORKERS) as pool:
            for task, result in zip(tasks, pool.map(reimplementation_task, [(t, JOBS_PER_WORKER, clip) for t in tasks])):
                out.append(result)
                run.manifest.setdefault('completed_tasks', []).append(task.task_id)
                run.save()
                print(task.task_id, f"{result[2].base_fit_seconds.iloc[0] + result[2].gamma_fit_seconds.sum():.1f}s", flush=True)
        for k, name in enumerate(('cate', 'capo_plugin')):
            pd.concat([o[k] for o in out], ignore_index=True).to_parquet(run.path / f'{name}.parquet', index=False)
        timing = pd.concat([o[2] for o in out], ignore_index=True)
        timing.to_csv(run.path / 'timing.csv', index=False)
        run.done(tasks=len(out), median_task_seconds=float(timing.groupby('task_id').apply(
            lambda t: t.base_fit_seconds.iloc[0] + t.gamma_fit_seconds.sum() + t.query_seconds.sum()).median()))
    except BaseException:
        run.done('failed')
        raise

AUTHORS = Path(__file__).resolve().parent / 'vendor' / 'BLearner'

# Vendored from https://github.com/CausalML/BLearner (MIT), unchanged runtime package.
AUTHORS_COMMIT = '734fd7639da54554e41d944bfc2187a24e75cf81'

JOBS = 4


CLIP = None

def clipped_logistic(clip):
    """The authors' elastic-net logistic propensity with predict_proba clipped to [clip, 1 - clip].
    Wraps (does not modify) their configuration; sklearn clone keeps the subclass and parameters."""
    from sklearn.linear_model import LogisticRegression

    class ClippedLogisticRegression(LogisticRegression):
        def predict_proba(self, X):
            import numpy as np
            p = np.clip(super().predict_proba(X), clip, 1 - clip)
            return p / p.sum(axis=1, keepdims=True)
    return ClippedLogisticRegression

def authors_estimator(gamma, seed):
    """compute_intervals_BLearner configuration (use_rho=True, cv=5 default)."""
    sys.path.insert(0, str(AUTHORS))
    from econml.grf import RegressionForest
    from sklearn import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn_quantile import RandomForestQuantileRegressor
    from models.blearner import BLearner
    from models.blearner.nuisance import KernelSuperquantileRegressor, RFKernel
    tau = gamma / (1 + gamma)
    forest = dict(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=JOBS)
    mu_model = RandomForestRegressor(**forest, random_state=seed)
    propensity_cls = LogisticRegression if CLIP is None else clipped_logistic(CLIP)
    return BLearner(
        propensity_model=propensity_cls(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000),
        quantile_plus_model=RandomForestQuantileRegressor(**forest, q=tau, random_state=seed + 1),
        quantile_minus_model=RandomForestQuantileRegressor(**forest, q=1 - tau, random_state=seed + 2),
        mu_model=mu_model,
        cvar_plus_model=KernelSuperquantileRegressor(kernel=RFKernel(clone(mu_model)), tau=tau, tail='right'),
        cvar_minus_model=KernelSuperquantileRegressor(kernel=RFKernel(clone(mu_model)), tau=1 - tau, tail='left'),
        cate_bounds_model=RegressionForest(**forest, random_state=seed + 3),
        use_rho=True, gamma=gamma, cv=5, random_state=seed)

def plugin_capo(est, x):
    """Per-arm plug-in bounds from the authors' fold-averaged nuisances (use_rho=True column layout:
    0 e, 5 mu0, 6 mu1, 7 rho+0, 8 rho+1, 9 rho-0, 10 rho-1), as in their effect_plugin."""
    n = est.predict_nuisances(x)
    e, mu0, mu1 = n[:, 0], n[:, 5], n[:, 6]
    return {1: (e * mu1 + (1 - e) * n[:, 10], e * mu1 + (1 - e) * n[:, 8]),
            0: ((1 - e) * mu0 + e * n[:, 9], (1 - e) * mu0 + e * n[:, 7])}

def authors_task(task):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    x, a, y = task.observed()
    query_ids, xq = task.query_x()
    cate, capo, timing = [], [], []
    for g in GAMMAS:
        start = time.perf_counter()
        est = authors_estimator(g, seed=0).fit(x, a, y)
        fitted = time.perf_counter()
        lo, up = est.effect(xq)
        plug = plugin_capo(est, xq)
        done = time.perf_counter()
        cate.append(pd.DataFrame(dict(task_id=task.task_id, query_id=query_ids, gamma=g, lower=lo, upper=up)))
        for arm in (0, 1):
            capo.append(pd.DataFrame(dict(task_id=task.task_id, query_id=query_ids, a_query=arm, gamma=g,
                                          lower=plug[arm][0], upper=plug[arm][1])))
        e = est.predict_nuisances(x)[:, 0]
        # base_fit_seconds = 0: the authors' class refits every nuisance at each Gamma (cold per Gamma).
        timing.append(dict(task_id=task.task_id, gamma=g, base_fit_seconds=0.0, gamma_fit_seconds=fitted - start,
                           query_seconds=done - fitted, n_context=len(y), d=x.shape[1],
                           min_e=float(e.min()), max_e=float(e.max())))
    return pd.concat(cate), pd.concat(capo), pd.DataFrame(timing)

def run_authors(smoke, cohort_set='default', clip=None):
    global CLIP
    CLIP = clip
    commit = AUTHORS_COMMIT  # Upstream snapshot is vendored in the project repository.
    suffix = ('' if cohort_set == 'default' else f'_{cohort_set}') + ('' if clip is None else f'_clip{clip:g}')
    run = Run('e08_blearner', ('authors_smoke' if smoke else 'authors') + suffix, smoke=smoke, gpu=False, cohort_set=cohort_set,
              threads=WORKERS * JOBS, workers=WORKERS, jobs_per_worker=JOBS, gammas=GAMMAS,
              implementation=f'authors code CausalML/BLearner@{commit}', deviations=['n_jobs=4 (authors: -2)', 'fixed random_state'] + ([] if clip is None else [f'SENSITIVITY CONTROL: propensity clipped to [{clip}, {1 - clip}]']))
    tasks = load_tasks(smoke=smoke, cohort_set=cohort_set)
    if smoke:
        tasks = [t for t in tasks if t.task_id in ('syn_1', 'acic_full_rep0', 'ihdp_rep0')]
    try:
        out = []
        with ProcessPoolExecutor(WORKERS) as pool:
            for task, result in zip(tasks, pool.map(authors_task, tasks)):
                out.append(result)
                run.manifest.setdefault('completed_tasks', []).append(task.task_id)
                run.save()
                print(task.task_id, f'{result[2].gamma_fit_seconds.sum():.1f}s', flush=True)
        for k, name in enumerate(('cate', 'capo_plugin')):
            pd.concat([o[k] for o in out], ignore_index=True).to_parquet(run.path / f'{name}.parquet', index=False)
        pd.concat([o[2] for o in out], ignore_index=True).to_csv(run.path / 'timing.csv', index=False)
        run.done(tasks=len(out))
    except BaseException:
        run.done('failed')
        raise

def report(smoke, e8_run='authors'):
    """Report authors' code by default; a regenerated reimplementation is an explicit control."""
    name = f'{e8_run}_smoke' if smoke else e8_run
    label = 'blearner_authors' if e8_run.startswith('authors') else 'blearner'
    run = completed('e08_blearner', name)
    assert run is not None
    tasks = load_tasks(smoke=smoke)
    ids = set(read(run / 'manifest.json')['completed_tasks'])
    tasks = [t for t in tasks if t.task_id in ids]
    ref_capo, ref_cate, _ = references(tasks)
    pfn_capo, pfn_cate, _ = pfn_outputs(tasks)
    bl_cate = pd.read_parquet(run / 'cate.parquet')
    bl_capo = pd.read_parquet(run / 'capo_plugin.parquet')
    frames = {
        ('cate', label): join(bl_cate, ref_cate, ['task_id', 'query_id', 'gamma']),
        ('cate', 'pifm'): join(pfn_cate, ref_cate, ['task_id', 'query_id', 'gamma']),
        ('capo', f'{label}_plugin'): join(bl_capo, ref_capo, ['task_id', 'query_id', 'a_query', 'gamma']),
        ('capo', 'pifm'): join(pfn_capo, ref_capo, ['task_id', 'query_id', 'a_query', 'gamma']),
    }
    reps = 200 if smoke else 2000
    out = REPORTS / ('smoke/blearner' if smoke else 'blearner') / e8_run
    out.mkdir(parents=True, exist_ok=True)
    summaries, by_gamma = [], []
    rows = []
    for (target, method), frame in frames.items():
        f = row_metrics(frame).assign(target=target, method=method)
        rows.append(f)
        summaries.append(summarize(f, ('group',), reps).assign(target=target, method=method))
        by_gamma.append(summarize(f, ('group', 'gamma'), reps).assign(target=target, method=method))
    summary = pd.concat(summaries, ignore_index=True)
    gamma = pd.concat(by_gamma, ignore_index=True)
    summary.to_csv(out / 'summary.csv', index=False)
    gamma.to_csv(out / 'by_gamma.csv', index=False)
    allrows = pd.concat(rows, ignore_index=True)
    per_task = allrows.groupby(['target', 'method', 'group', 'task_id']).apply(lambda f: pd.Series(dict(
        rmse=np.sqrt(np.mean(np.r_[f.err_lower, f.err_upper] ** 2)), covers_ref=f.covers_ref.mean(),
        true_in=f.true_in.astype(float).mean()))).reset_index()
    per_task.to_csv(out / 'per_task.csv', index=False)
    timing = pd.read_csv(run / 'timing.csv')
    per_task_time = timing.groupby('task_id').apply(lambda t: pd.Series(dict(
        base_fit=t.base_fit_seconds.iloc[0], all_gamma_total=t.base_fit_seconds.iloc[0] + t.gamma_fit_seconds.sum() + t.query_seconds.sum(),
        single_gamma_mean=t.base_fit_seconds.iloc[0] + (t.gamma_fit_seconds + t.query_seconds).mean(),
        min_e=t.min_e.iloc[0], max_e=t.max_e.iloc[0]))).reset_index()
    per_task_time['group'] = per_task_time.task_id.map(group_of)
    per_task_time.to_csv(out / 'timing_per_task.csv', index=False)
    tsum = per_task_time.groupby('group')[['base_fit', 'single_gamma_mean', 'all_gamma_total']].median()
    tsum.to_csv(out / 'timing_summary.csv')
    cols = ['target', 'method', 'group', 'tasks', 'rows', 'rmse_lower', 'rmse_upper', 'covers_ref', 'width', 'ref_width', 'true_in']
    print(summary[[c for c in cols if c in summary]].round(3).to_string())
    print(tsum.round(1).to_string())
    plot(gamma, out)

def plot(gamma, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = gamma[gamma.target.eq('cate')]
    groups = sorted(g.group.unique())
    fig, axes = plt.subplots(2, len(groups), figsize=(4.2 * len(groups), 6.5), squeeze=False)
    colors = {'blearner': '#c0504d', 'blearner_authors': '#c0504d', 'pifm': '#1f4e79'}
    for j, group in enumerate(groups):
        for method, part in g[g.group.eq(group)].groupby('method'):
            part = part.sort_values('gamma')
            rmse = np.sqrt((part.rmse_lower ** 2 + part.rmse_upper ** 2) / 2)
            axes[0, j].plot(part.gamma, rmse, 'o-', color=colors[method], label=method)
            axes[1, j].plot(part.gamma, part.width, 'o-', color=colors[method], label=f'{method} width')
            if method == 'pifm':
                axes[1, j].plot(part.gamma, part.ref_width, 'k--', label='reference width')
        axes[0, j].set_title(f'{group}: CATE endpoint RMSE')
        axes[1, j].set_title(f'{group}: mean CATE-bound width')
        for ax in axes[:, j]:
            ax.set_xlabel('Gamma')
            ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(out / 'cate_rmse_width_by_gamma.png', dpi=130)


def oracle_mapping(rng, draws):
    rows = []
    families = {'normal': rng.normal(0.3, 1.2, draws), 'lognormal': rng.lognormal(0, 0.8, draws),
                'bimodal': np.r_[rng.normal(-2, 0.5, draws // 2), rng.normal(1, 1, draws - draws // 2)]}
    for name, y in families.items():
        for pi in (0.1, 0.5, 0.85):
            lo, up = msm_reference_from_samples(y[None, :], np.array([pi]), np.array(GAMMAS))
            for k, g in enumerate(GAMMAS):
                tau = g / (1 + g)
                r_plus, r_minus = r_targets(y, np.quantile(y, tau), np.quantile(y, 1 - tau), g)
                psi_plus = pi * y.mean() + (1 - pi) * r_plus.mean()
                psi_minus = pi * y.mean() + (1 - pi) * r_minus.mean()
                rows.append(dict(family=name, pi=pi, gamma=g, closed_lower=lo[0, k], closed_upper=up[0, k],
                                 rho_lower=psi_minus, rho_upper=psi_plus,
                                 abs_diff=max(abs(psi_minus - lo[0, k]), abs(psi_plus - up[0, k]))))
    return pd.DataFrame(rows)

def end_to_end(rng, n, n_test, n_jobs):
    x = rng.uniform(-1, 1, size=(n, 3))
    m1 = lambda z: z[:, 1]
    m0 = lambda z: -0.5 * z[:, 1]
    e = expit(1.0 * x[:, 0])
    a = rng.binomial(1, e)
    y = np.where(a == 1, m1(x), m0(x)) + rng.normal(size=n)
    xt = rng.uniform(-0.8, 0.8, size=(n_test, 3))
    learner = BLearner(cv=5, seed=0, n_jobs=n_jobs).fit_base(x, a, y)
    rows = []
    for g in GAMMAS:
        tau = g / (1 + g)
        c = (1 - 1 / g) * norm.pdf(norm.ppf(tau)) / (1 - tau)
        fit = learner.fit_gamma(g)
        lo, up = fit.effect(xt)
        cap = fit.plugin_capo(xt)
        true_lo, true_up = m1(xt) - m0(xt) - c, m1(xt) - m0(xt) + c
        et = expit(xt[:, 0])
        ref1_up = et * m1(xt) + (1 - et) * (m1(xt) + c)
        rows.append(dict(gamma=g, c=c, ref_width=2 * c, rmse_lower=np.sqrt(np.mean((lo - true_lo) ** 2)),
                         rmse_upper=np.sqrt(np.mean((up - true_up) ** 2)), mean_width=np.mean(up - lo),
                         bias_lower=np.mean(lo - true_lo), bias_upper=np.mean(up - true_up),
                         plugin_rmse_upper_arm1=np.sqrt(np.mean((cap[1][1] - ref1_up) ** 2))))
    return pd.DataFrame(rows)

def validate(smoke):
    run = Run('e08_blearner', 'validate_smoke' if smoke else 'validate', smoke=smoke, gpu=False, threads=8)
    rng = np.random.default_rng(20261107)
    mapping = oracle_mapping(rng, 20000 if smoke else 400000)
    mapping.to_csv(run.path / 'oracle_mapping.csv', index=False)
    tol = 0.02 if smoke else 0.005
    assert mapping.abs_diff.max() < tol, mapping.abs_diff.max()
    e2e = end_to_end(rng, 2000 if smoke else 6000, 400, n_jobs=8)
    e2e.to_csv(run.path / 'end_to_end.csv', index=False)
    print(mapping.groupby('family').abs_diff.max().to_string())
    print(e2e.round(4).to_string())
    run.done(max_oracle_mapping_diff=float(mapping.abs_diff.max()),
             max_e2e_rmse=float(e2e[['rmse_lower', 'rmse_upper']].max().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['authors', 'reimplementation', 'report', 'validate'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cohort', default='default', choices=['default', 'ihdp'])
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--e8-run', default='authors', choices=['authors', 'authors_clip0.01', 'cohort', 'cohort_clip0.01'])
    args = parser.parse_args()
    if args.stage == 'authors':
        run_authors(args.smoke, args.cohort, args.clip)
    else:
        assert args.cohort == 'default', 'only the authors stage supports the IHDP cohort'
        if args.stage == 'reimplementation':
            run_reimplementation(args.smoke, args.clip)
        elif args.stage == 'report':
            report(args.smoke, args.e8_run)
        else:
            validate(args.smoke)


if __name__ == '__main__':
    main()
