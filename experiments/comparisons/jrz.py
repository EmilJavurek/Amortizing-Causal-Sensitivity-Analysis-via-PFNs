"""JRZ KL bounds and matched PI-FM aggregate comparisons.

JRZ targets a one-sided KL relaxation, an outer bound of the two-sided PI-FM target.
The independent primal/dual checks quantify the difference on non-Gaussian laws.
Reference: github.com/ying531/f-sens-paper@62001a69afeaed299f2d689458717173db51bacb.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
import argparse
import time
import pandas as pd
from experiments.shared.runtime import Run, completed, sha
from experiments.comparisons.benchmark import load_tasks
from scipy.optimize import minimize
from scipy.special import expit
from experiments.shared.runtime import write

EPS_ALPHA = 0.01

PROPENSITY_CLIP = (0.01, 0.99)

@dataclass(frozen=True)
class JRZConfig:
    folds: int = 3
    ridge: float = 1e-3
    lbfgs_iterations: int = 200
    rf_trees: int = 100
    rf_jobs: int = 4
    seed: int = 0
    select_sieve: bool = True                     # inner-CV choice of basis kind and ridge
    ridge_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    kinds: tuple = ('linear', 'spline')
    inner_folds: int = 3

class SplineBasis:
    """Additive cubic truncated-power spline basis fitted on one fold's covariates."""

    def __init__(self, x_fit, kind='spline'):
        assert kind in ('linear', 'spline')
        self.low, self.high = x_fit.min(axis=0), x_fit.max(axis=0)
        self.columns = []
        for j in range(x_fit.shape[1]):
            values = x_fit[:, j]
            if kind == 'spline' and np.unique(values).size > 5:
                knots = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
                self.columns.append((j, knots))
            else:
                self.columns.append((j, None))
        raw = self._raw(x_fit)
        self.mean = raw.mean(axis=0)
        self.std = np.where(raw.std(axis=0) < 1e-8, 1.0, raw.std(axis=0))

    def _raw(self, x):
        parts = []
        for j, knots in self.columns:
            v = x[:, j]
            if knots is None:
                parts.append(v[:, None])
            else:
                parts.append(np.column_stack([v, v ** 2, v ** 3] + [np.maximum(v - k, 0) ** 3 for k in knots]))
        return np.hstack(parts)

    def __call__(self, x):
        x = np.clip(x, self.low, self.high)        # no cubic extrapolation beyond the fit fold
        z = (self._raw(x) - self.mean) / self.std
        return np.hstack([np.ones((len(x), 1)), z])

def kl_dual_h(alpha, eta, y, rho):
    """H(x, y; alpha, eta) for KL; works for numpy or torch inputs."""
    lib = torch if isinstance(y, torch.Tensor) else np
    return alpha * lib.exp(-(y + eta) / alpha - 1.0) + eta + alpha * rho

def _fit_sieve(x, y, rho, kind, ridge, iterations):
    """Penalized ERM of Eq. (16) on one basis; returns (params callable, final penalized loss)."""
    basis = SplineBasis(x, kind)
    design = torch.tensor(basis(x), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    scale = float(np.std(y)) if np.std(y) > 1e-8 else 1.0
    k = design.shape[1]
    theta_a = torch.zeros(k, dtype=torch.float64)
    theta_e = torch.zeros(k, dtype=torch.float64)
    theta_a[0] = float(np.log(np.expm1(scale)))          # alpha ~ sd(y) at start
    theta_e[0] = -float(np.mean(y)) - scale              # exponent ~ 0 at start
    theta_a.requires_grad_(True)
    theta_e.requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta_a, theta_e], lr=1.0, max_iter=iterations,
                                  tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def penalized():
        alpha = EPS_ALPHA + torch.nn.functional.softplus(design @ theta_a)
        eta = design @ theta_e
        return kl_dual_h(alpha, eta, yt, rho).mean() + ridge * (theta_a[1:].square().sum() + theta_e[1:].square().sum())

    def objective():
        optimizer.zero_grad()
        loss = penalized()
        loss.backward()
        return loss

    optimizer.step(objective)
    with torch.no_grad():
        final = penalized().item()
    ta, te = theta_a.detach().numpy(), theta_e.detach().numpy()

    def params(x_new):
        b = basis(x_new)
        alpha = EPS_ALPHA + np.logaddexp(0.0, b @ ta)
        return alpha, b @ te

    return params, final

def held_out_dual(params, x, y, rho):
    alpha, eta = params(x)
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.mean(kl_dual_h(alpha, eta, y, rho)))
    return value if np.isfinite(value) else np.inf

def fit_dual(x, y, rho, config: JRZConfig, rng=None, record=None):
    """Per-x ERM of Eq. (16): returns a callable x -> (alpha(x), eta(x)).

    With `select_sieve`, the basis kind and ridge are chosen by inner K-fold cross-validation of
    the unpenalized dual risk on the ERM fold only (a deviation; the authors fix a spline sieve).
    """
    kind, ridge = 'spline', config.ridge
    if config.select_sieve:
        rng = rng or np.random.default_rng(config.seed)
        split = np.array_split(rng.permutation(len(y)), config.inner_folds)
        scores = {}
        for cand_kind in config.kinds:
            for cand_ridge in config.ridge_grid:
                risk = []
                for f in range(config.inner_folds):
                    fit = np.concatenate([split[g] for g in range(config.inner_folds) if g != f])
                    params, _ = _fit_sieve(x[fit], y[fit], rho, cand_kind, cand_ridge, config.lbfgs_iterations)
                    risk.append(held_out_dual(params, x[split[f]], y[split[f]], rho))
                scores[(cand_kind, cand_ridge)] = float(np.mean(risk))
        kind, ridge = min(scores, key=scores.get)
        assert np.isfinite(scores[(kind, ridge)]), 'no sieve candidate with finite held-out dual risk'
        if record is not None:
            record.append(dict(kind=kind, ridge=ridge, cv_risk=scores[(kind, ridge)]))
    params, final = _fit_sieve(x, y, rho, kind, ridge, config.lbfgs_iterations)
    assert np.isfinite(final), 'non-finite KL dual objective'
    return params

def _folds(index, folds, rng):
    perm = rng.permutation(index)
    return np.array_split(perm, folds)

def counterfactual_mean_lower(x, t, y, rho, config: JRZConfig, rng):
    """Algorithm 1 lower bound on E[Y(1) | T=0] using the T=1 sample's outcomes y[t == 1].

    Follows the authors' fold rotation (code/simu.py): propensity and ERM on fold j,
    h-regression on fold j+1, debiased average on fold j+2. Returns (estimate, standard error)
    with the SE from the Theorem 3 influence values sd(phi)/sqrt(n), as in their code's
    `inf_low` (treated: r (H - h) / p1, control: h / p0).
    """
    treated, control = np.where(t == 1)[0], np.where(t == 0)[0]
    p1 = t.mean()
    tf, cf = _folds(treated, config.folds, rng), _folds(control, config.folds, rng)
    estimates = []
    influence = np.zeros(len(t))
    for j in range(config.folds):
        a, b, c = j, (j + 1) % config.folds, (j + 2) % config.folds
        fit_rows = np.r_[tf[a], cf[a]]
        e_model = RandomForestRegressor(n_estimators=config.rf_trees, n_jobs=config.rf_jobs,
                                        random_state=config.seed + j).fit(x[fit_rows], t[fit_rows])
        params = fit_dual(x[tf[a]], y[tf[a]], rho, config, rng=rng)
        alpha_b, eta_b = params(x[tf[b]])
        h_values = kl_dual_h(alpha_b, eta_b, y[tf[b]], rho)
        h_model = RandomForestRegressor(n_estimators=config.rf_trees, n_jobs=config.rf_jobs,
                                        random_state=config.seed + 10 + j).fit(x[tf[b]], h_values)
        alpha_c, eta_c = params(x[tf[c]])
        h_adj = kl_dual_h(alpha_c, eta_c, y[tf[c]], rho)
        e_c = np.clip(e_model.predict(x[tf[c]]), *PROPENSITY_CLIP)
        ratio = (1 - e_c) * p1 / (e_c * (1 - p1))
        h_treated, h_control = h_model.predict(x[tf[c]]), h_model.predict(x[cf[c]])
        influence[tf[c]] = ratio * (h_adj - h_treated) / p1
        influence[cf[c]] = h_control / (1 - p1)
        estimates.append(np.mean(ratio * (h_adj - h_treated)) + np.mean(h_control))
    return -float(np.mean(estimates)), float(np.std(influence) / np.sqrt(len(t)))

def counterfactual_mean_bounds(x, t, y, rho, config: JRZConfig, seed, return_se=False):
    """[lower, upper] on E[Y(1) | T=0]; upper = -lower(-Y). Same folds for both directions."""
    lower, se_lower = counterfactual_mean_lower(x, t, y, rho, config, np.random.default_rng(seed))
    neg, se_upper = counterfactual_mean_lower(x, t, -y, rho, config, np.random.default_rng(seed))
    if return_se:
        return lower, -neg, se_lower, se_upper
    return lower, -neg

def apo_bounds(x, a, y, rho, config: JRZConfig, seed=0):
    """Bounds on E[Y(arm)] over the sample's covariate distribution for arm in {0, 1}.

    E[Y(1)] = p1 E[Y | T=1] + p0 E[Y(1) | T=0]; arm 0 swaps the roles of the groups (paper Eq. 14).
    """
    out = {}
    for arm in (0, 1):
        t = (a == arm).astype(float)          # 'observed' group for this arm
        lo, up = counterfactual_mean_bounds(x, t, y, rho, config, seed + arm)
        p = t.mean()
        observed = y[t == 1].mean()
        out[arm] = (p * observed + (1 - p) * lo, p * observed + (1 - p) * up)
    return out

PACKAGE = 'e11_jrz_fsens'

RHOS = (0.05, 0.1, 0.25, 0.5, 1.0)

def run_jrz(smoke, tag=''):
    config = JRZConfig(rf_jobs=4)
    run = Run(PACKAGE, ('jrz_smoke' if smoke else 'jrz') + tag, smoke=smoke, gpu=False, threads=4,
              rhos=RHOS, jrz_config=config.__dict__)
    rows, timing = [], []
    try:
        for task in load_tasks(smoke=smoke):
            x, a, y = task.observed()
            start = time.perf_counter()
            for rho in RHOS:
                bounds = apo_bounds(x, a, y, rho, config, seed=0)
                for arm, (lo, up) in bounds.items():
                    # Diagnostic flag only (values are kept): non-finite, crossed, or far outside
                    # the outcome scale (|bound - mean(y)| > 50 sd(y)).
                    invalid = not (np.isfinite(lo) and np.isfinite(up)) or up < lo or\
                        max(abs(lo - y.mean()), abs(up - y.mean())) > 50 * y.std()
                    rows.append(dict(task_id=task.task_id, a_query=arm, gamma=rho, lower=lo, upper=up,
                                     invalid=bool(invalid)))
            timing.append(dict(task_id=task.task_id, fit_seconds=time.perf_counter() - start, query_seconds=0.0,
                               rhos=len(RHOS)))
            print(task.task_id, f"{timing[-1]['fit_seconds']:.1f}s", flush=True)
        pd.DataFrame(rows).to_parquet(run.path / 'apo.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(tasks=len(timing), invalid=int(sum(r['invalid'] for r in rows)))
    except BaseException:
        run.done('failed')
        raise

def run_pfn_kl(smoke, tag=''):
    import torch
    from experiments.shared.evaluation import KL_CHECKPOINT, load_pifm, pifm_predict_frame
    run = Run(PACKAGE, 'pfn_kl_smoke' if smoke else 'pfn_kl', smoke=smoke, gpu=True, threads=4,
              rhos=RHOS, kl_checkpoint_sha256=sha(KL_CHECKPOINT))
    device = torch.device('cuda:0')
    model = load_pifm(KL_CHECKPOINT, device)[0]
    rows, timing = [], []
    try:
        for task in load_tasks(smoke=smoke):
            xcols = task.xcols
            grid = []
            for arm in (0, 1):
                for rho in RHOS:
                    frame = task.context[['row'] + xcols].copy()
                    frame['a_query'], frame['gamma'] = arm, rho
                    grid.append(frame)
            query = pd.concat(grid, ignore_index=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            pred = pifm_predict_frame(model, task.context, xcols, query, device, chunk_size=4096)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            apo = pred.groupby(['a_query', 'gamma'], as_index=False)[['lower_mean', 'upper_mean']].mean()
            rows.append(apo.rename(columns={'lower_mean': 'lower', 'upper_mean': 'upper'}).assign(task_id=task.task_id))
            timing.append(dict(task_id=task.task_id, fit_seconds=0.0, query_seconds=seconds, rows=len(query)))
        pd.concat(rows, ignore_index=True).to_parquet(run.path / 'apo.parquet', index=False)
        pd.DataFrame(timing).to_csv(run.path / 'timing.csv', index=False)
        run.done(tasks=len(timing))
    except BaseException:
        run.done('failed')
        raise

def compare(smoke, tag=''):
    suffix = '_smoke' if smoke else ''
    run = Run(PACKAGE, 'compare' + suffix + tag, smoke=smoke, gpu=False, threads=2)
    tasks = {t.task_id: t for t in load_tasks(smoke=smoke)}
    jrz_apo = pd.read_parquet(completed(PACKAGE, 'jrz' + suffix + tag) / 'apo.parquet')
    pfn_apo = pd.read_parquet(completed(PACKAGE, 'pfn_kl' + suffix) / 'apo.parquet')
    keys = ['task_id', 'a_query', 'gamma']
    out = jrz_apo.merge(pfn_apo, on=keys, suffixes=('_jrz', '_pfn'), validate='one_to_one')
    truth = []
    for task_id, t in tasks.items():
        apo = t.apo
        for arm in (0, 1):
            value = apo.loc[apo.a_query.eq(arm), 'apo_true'].iloc[0] if 'apo_true' in apo else np.nan
            truth.append(dict(task_id=task_id, a_query=arm, apo_true=value,
                              apo_gamma1=apo.loc[apo.a_query.eq(arm) & apo.gamma.eq(1.0), 'apo_lower'].iloc[0]))
    out = out.merge(pd.DataFrame(truth), on=['task_id', 'a_query'], validate='many_to_one')
    for m in ('jrz', 'pfn'):
        out[f'width_{m}'] = out[f'upper_{m}'] - out[f'lower_{m}']
        out[f'true_in_{m}'] = np.where(out.apo_true.notna(),
                                       (out[f'lower_{m}'] <= out.apo_true) & (out.apo_true <= out[f'upper_{m}']), np.nan)
        out[f'point_in_{m}'] = (out[f'lower_{m}'] <= out.apo_gamma1) & (out.apo_gamma1 <= out[f'upper_{m}'])
    out['jrz_contains_pfn'] = (out.lower_jrz <= out.lower_pfn) & (out.upper_pfn <= out.upper_jrz)
    out.to_parquet(run.path / 'comparison.parquet', index=False)
    summary = out.groupby('gamma').agg(width_jrz=('width_jrz', 'mean'), width_pfn=('width_pfn', 'mean'),
                                       point_in_jrz=('point_in_jrz', 'mean'), point_in_pfn=('point_in_pfn', 'mean'),
                                       jrz_contains_pfn=('jrz_contains_pfn', 'mean'), invalid_jrz=('invalid', 'sum')).reset_index()
    summary.to_csv(run.path / 'summary.csv', index=False)
    print(out.round(3).to_string())
    run.done(rows=len(out))

def scalar_dual_lower(y, w, rho):
    """-min_{alpha>0, eta} sum_k w_k H(alpha, eta, y_k) for an (empirical or discrete) law."""
    def objective(z):
        alpha = np.exp(z[0])
        return float(np.sum(w * kl_dual_h(alpha, z[1], y, rho)))
    best = None
    for a0 in (np.log(y.std() + 1e-3), 0.0, -1.0):
        res = minimize(objective, x0=[a0, -np.sum(w * y)], method='Nelder-Mead',
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
        best = res if best is None or res.fun < best.fun else best
    return -best.fun

def dual_vs_closed_form():
    rng = np.random.default_rng(0)
    mu, sigma = 0.7, 1.3
    y = mu + sigma * rng.standard_normal(400_000)
    w = np.full(len(y), 1 / len(y))
    rows = []
    for rho in RHOS:
        lo = scalar_dual_lower(y, w, rho)
        up = -scalar_dual_lower(-y, w, rho)
        exact = sigma * np.sqrt(2 * rho)
        rows.append(dict(rho=rho, lower=lo, upper=up, exact_lower=mu - exact, exact_upper=mu + exact,
                         err=max(abs(lo - mu + exact), abs(up - mu - exact))))
    worst = max(r['err'] for r in rows)
    assert worst < 0.02, rows
    return dict(rows=rows, max_abs_error=worst, mc_draws=len(y))

def primal(y, phi, rho, sense, two_sided):
    """Generic constrained solver on the probability simplex via nu = softmax(z) (no dual used)."""
    from scipy.special import log_softmax
    log_phi = np.log(phi)

    def kl_nu_phi(z):
        log_nu = log_softmax(z)
        return float(np.sum(np.exp(log_nu) * (log_nu - log_phi)))

    def kl_phi_nu(z):
        return float(np.sum(phi * (log_phi - log_softmax(z))))

    cons = [dict(type='ineq', fun=lambda z: rho - kl_nu_phi(z))]
    if two_sided:
        cons.append(dict(type='ineq', fun=lambda z: rho - kl_phi_nu(z)))
    sign = -1.0 if sense == 'max' else 1.0
    best = None
    for start in (0.0, 0.5, -0.5):
        z0 = log_phi + start * sign * (y - y.mean()) / y.std()
        res = minimize(lambda z: sign * float(np.exp(log_softmax(z)) @ y), x0=z0, method='SLSQP', constraints=cons,
                       options=dict(ftol=1e-14, maxiter=5000))
        feasible = all(c['fun'](res.x) >= -1e-8 for c in cons)
        if feasible and (best is None or res.fun < best.fun):
            best = res
    assert best is not None, 'no feasible primal solution'
    return float(np.exp(log_softmax(best.x)) @ y)

def discrete_primal():
    k = 40
    levels = (np.arange(k) + 0.5) / k
    y = np.exp(0.9 * np.sqrt(2) * __import__('scipy').special.erfinv(2 * levels - 1))   # lognormal quantiles
    phi = np.full(k, 1 / k)
    rows = []
    for rho in RHOS:
        dual_lo, dual_up = scalar_dual_lower(y, phi, rho), -scalar_dual_lower(-y, phi, rho)
        p_lo, p_up = primal(y, phi, rho, 'min', False), primal(y, phi, rho, 'max', False)
        t_lo, t_up = primal(y, phi, rho, 'min', True), primal(y, phi, rho, 'max', True)
        rows.append(dict(rho=rho, dual_lower=dual_lo, dual_upper=dual_up, primal_lower=p_lo, primal_upper=p_up,
                         two_sided_lower=t_lo, two_sided_upper=t_up,
                         dual_primal_err=max(abs(dual_lo - p_lo), abs(dual_up - p_up)),
                         width_one_sided=p_up - p_lo, width_two_sided=t_up - t_lo))
    worst = max(r['dual_primal_err'] for r in rows)
    assert worst < 1e-4, rows
    assert all(r['two_sided_lower'] >= r['primal_lower'] - 1e-7 and r['two_sided_upper'] <= r['primal_upper'] + 1e-7 for r in rows)
    # Gaussian discretization: one-sided and two-sided coincide up to discretization error.
    gauss = np.sqrt(2) * __import__('scipy').special.erfinv(2 * ((np.arange(200) + 0.5) / 200) - 1)
    pg = np.full(200, 1 / 200)
    g = [dict(rho=r, one_sided_upper=primal(gauss, pg, r, 'max', False), two_sided_upper=primal(gauss, pg, r, 'max', True),
              exact=np.sqrt(2 * r)) for r in (0.1, 0.5)]
    return dict(rows=rows, max_dual_primal_error=worst, gaussian=g)

def dgp(n, rng):
    """JRZ-style heteroscedastic Gaussian outcome; covariates U[0,1]^4."""
    x = rng.uniform(size=(n, 4))
    gamma = np.array([-0.531, 0.126, -0.312, 0.018])
    beta1 = np.array([0.531, 1.126, -0.312, 0.671])
    e = expit((x - 0.5) @ (4 * gamma))
    t = rng.binomial(1, e)
    mu1 = x @ beta1
    sigma = np.sqrt(1 + 1.25 * x[:, 0] ** 2)
    y1 = mu1 + sigma * rng.standard_normal(n)
    return x, t, y1, e, mu1, sigma

def estimator_recovers_truth(sizes, replications):
    """Replicated Algorithm 1 runs: bias, spread and influence-function CI coverage of the truth.

    Assertion only at the largest n (finite-sample behaviour at n = 1,024 is reported, not
    asserted; invalid runs with upper < lower are counted).
    """
    config = JRZConfig(rf_jobs=4)
    big = np.random.default_rng(99)
    xb, tb, _, eb, mub, sb = dgp(2_000_000, big)
    control = tb == 0
    rows, summary = [], []
    for n in sizes:
        for rho in (0.1, 0.5):
            truth_lo = float(np.mean(mub[control] - sb[control] * np.sqrt(2 * rho)))
            truth_up = float(np.mean(mub[control] + sb[control] * np.sqrt(2 * rho)))
            for r in range(replications):
                rng = np.random.default_rng(1000 * n + r)
                x, t, y1, *_ = dgp(n, rng)
                start = time.perf_counter()
                lo, up, se_lo, se_up = counterfactual_mean_bounds(x, t.astype(float), y1, rho, config, seed=r, return_se=True)
                rows.append(dict(n=n, rho=rho, rep=r, lower=lo, upper=up, se_lower=se_lo, se_upper=se_up,
                                 truth_lower=truth_lo, truth_upper=truth_up, seconds=time.perf_counter() - start))
            part = [q for q in rows if q['n'] == n and q['rho'] == rho]
            err_lo = np.array([q['lower'] - truth_lo for q in part])
            err_up = np.array([q['upper'] - truth_up for q in part])
            valid = np.array([q['upper'] >= q['lower'] for q in part])
            cover = np.mean([abs(q['lower'] - truth_lo) <= 1.96 * q['se_lower'] for q in part] +
                            [abs(q['upper'] - truth_up) <= 1.96 * q['se_upper'] for q in part])
            summary.append(dict(n=n, rho=rho, replications=len(part), invalid_runs=int((~valid).sum()),
                                median_err_lower=float(np.median(err_lo)), median_err_upper=float(np.median(err_up)),
                                mean_err_lower=float(err_lo.mean()), mean_err_upper=float(err_up.mean()),
                                sd_lower=float(err_lo.std()), sd_upper=float(err_up.std()), ci95_coverage=float(cover),
                                median_seconds=float(np.median([q['seconds'] for q in part]))))
            print(summary[-1], flush=True)
    large = [q for q in summary if q['n'] == max(sizes)]
    for q in large:
        se_mean = max(q['sd_lower'], q['sd_upper']) / np.sqrt(q['replications'])
        assert q['invalid_runs'] == 0, q
        assert abs(q['mean_err_lower']) < 3 * se_mean + 0.02 and abs(q['mean_err_upper']) < 3 * se_mean + 0.02, q
        assert q['ci95_coverage'] >= 0.75, q
    return dict(rows=rows, summary=summary)

def checks(quick, tag):
    run = Run('e11_jrz_fsens', ('checks_quick' if quick else 'checks') + tag, smoke=quick, gpu=False, threads=4,
              jrz_config=JRZConfig().__dict__)
    try:
        result = dict(dual_vs_closed_form=dual_vs_closed_form(), discrete_primal=discrete_primal())
        print('dual/primal checks passed', flush=True)
        result['estimator'] = estimator_recovers_truth((1024, 6000), 4 if quick else 12)
        write(run.path / 'checks.json', result)
        run.done(passed=True)
    except BaseException:
        run.done('failed')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'pfn_kl', 'compare', 'checks'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.stage == 'pfn_kl':
        run_pfn_kl(args.smoke)
    elif args.stage == 'checks':
        checks(args.quick, args.tag)
    else:
        {'run': run_jrz, 'compare': compare}[args.stage](args.smoke, args.tag)


if __name__ == '__main__':
    main()
