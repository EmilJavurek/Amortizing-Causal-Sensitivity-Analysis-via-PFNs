"""KL optimizer and PI-FM labels against exact Gaussian/linear CAPO bounds."""


from __future__ import annotations

from experiments.shared.paths import REPORTS
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import yaml
from experiments.shared.inputs import cohort_ids, input_path, normalized_root
from experiments.shared.paths import CODE, RUNS
from experiments.shared.runtime import Run, cluster_bootstrap, read
from scipy.optimize import minimize
from scipy.stats import norm


# Run


PROD_YAML = CODE / 'experiments/configs/paper/optimization_kl/03_frontier.yaml'
PIS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIGMAS = [0.25, 1.0, 3.0]
MUS = [0.0, 0.0, 1.5, -2.0]
PFN_GAMMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

class ExactLinearOutcome(torch.nn.Module):
    """h(x, a, u) = x_0 + x_1 * u: exact linear outcome map for the sequential forward path."""

    def forward(self, x, a, u):
        assert x.shape[-1] == 2 and u.shape[-1] == 1
        return x[:, :1] + x[:, 1:2] * u

def solver_configs(budget):
    from setting_standard.frontier_optimization import FlowConfig, SolverConfig, LambdaGridConfig, make_lambda_grid
    c = yaml.safe_load(PROD_YAML.read_text())
    assert c['sensitivity_model'] == 'kl'
    flow = FlowConfig(num_bins=c['flow_num_bins'], tail_bound=c['flow_tail_bound'], min_bin_width=c['flow_min_bin_width'],
                      min_bin_height=c['flow_min_bin_height'], min_derivative=c['flow_min_derivative'])
    solver = dict(num_mc_samples_train=c['num_mc_samples_train'], num_mc_samples_eval=c['num_mc_samples_eval'],
                  max_steps=c['max_steps'], learning_rate=1e-3, lr_lambda_schedule=c['lr_lambda_schedule'],
                  lr_lambda_ref=c['lr_lambda_ref'], lr_lambda_min_mult=c['lr_lambda_min_mult'],
                  max_steps_lambda_schedule=c['max_steps_lambda_schedule'],
                  max_steps_lambda_max_mult=c['max_steps_lambda_max_mult'], loss_reduction=c['loss_reduction'],
                  early_stop=c['early_stop'], early_stop_min_steps=c['early_stop_min_steps'],
                  early_stop_check_every=c['early_stop_check_every'], early_stop_patience=c['early_stop_patience'],
                  early_stop_abs_tol=c['early_stop_abs_tol'], early_stop_rel_tol=c['early_stop_rel_tol'])
    grid = LambdaGridConfig(lambda_min=c['lambda_min'], lambda_max=c['lambda_max'], lambda_n=c['lambda_n'],
                            lambda_spacing=c['lambda_spacing'])
    m = 2048
    if budget == 'smoke':
        solver.update(max_steps=40, early_stop=False); grid.lambda_n = 4; m = 30
    elif budget == 'high':
        solver.update(num_mc_samples_train=512, num_mc_samples_eval=8192, max_steps=1000, early_stop=False)
    else:
        assert budget == 'prod'
    return flow, SolverConfig(**solver), make_lambda_grid(grid), m

def toy_queries(m):
    """Per-query (mu, sigma, pi, config index); configs cycle so every config gets ~m/15 replicates."""
    configs = [(pi, sg) for pi in PIS for sg in SIGMAS]
    idx = np.arange(m) % len(configs)
    pi = np.array([configs[i][0] for i in idx]); sg = np.array([configs[i][1] for i in idx])
    return idx, pi, sg

def kl_state_quadrature(params, flow, zq, wq):
    """Attained (KL(nu||phi), KL(phi||nu), E_nu U) of saved spline states by quadrature over z."""
    from setting_standard.frontier_optimization import batched_spline_forward, standard_normal_logprob
    u, log_p = batched_spline_forward(zq[None, None].expand(1, params.shape[0], -1).contiguous(), params[None], flow)
    u, log_r = u[0].double(), (log_p - standard_normal_logprob(u))[0].double()
    kl_nu_phi = log_r @ wq
    kl_phi_nu = (-log_r * torch.exp(-log_r)) @ wq
    return kl_nu_phi, kl_phi_nu, u @ wq

def toy(budget):
    from setting_standard.frontier_optimization import OutcomeForwardPlan, sweep_lambda_grid_batch
    flow, solver, lambdas, m = solver_configs(budget)
    run = Run('e04', f'toy_{budget}', smoke=budget == 'smoke', gpu=True, threads=2, flow=vars(flow),
              solver=vars(solver), lambda_grid=lambdas, queries_per_dgp=m, mus=MUS, pis=PIS, sigmas=SIGMAS)
    dev = 'cuda:0'
    torch.manual_seed(20261107); np.random.seed(20261107)
    idx, pi, sg = toy_queries(m)
    B = len(MUS)
    x = torch.tensor(np.stack([np.stack([np.full(m, mu), sg], 1) for mu in MUS]), dtype=torch.float32, device=dev)
    a = torch.ones(B, m, 1, device=dev)
    pit = torch.tensor(np.tile(pi, (B, 1)), dtype=torch.float32, device=dev)
    module = ExactLinearOutcome()
    # Adapter validation: exact values through the solver's own sequential forward.
    from setting_standard.frontier_optimization import sequential_outcome_forward
    u_test = torch.randn(B, m, 7, device=dev)
    y_test = sequential_outcome_forward([module] * B, x, a, u_test)
    assert torch.equal(y_test, x[..., :1] + x[..., 1:2] * u_test), 'adapter is not exact'
    plan = OutcomeForwardPlan(use_bucket_forward=False, depth_buckets=None, sequential_f_bnns=[module] * B)
    try:
        results = {}
        for head in ('upper', 'lower'):
            t = time.time()
            results[head] = sweep_lambda_grid_batch(
                outcome_plan=plan, x_queries_batch=x, a_queries_batch=a, pi_queries_batch=pit,
                dataset_ids=list(range(B)), query_specs_batch=[[None] * m] * B, bound_type=head,
                lambda_grid=lambdas, sensitivity_model='kl', flow_config=flow, solver_config=solver, device=dev,
                warm_start_lambdas=True)
            run.manifest[f'{head}_sweep_seconds'] = time.time() - t
            print(f'{head} sweep done in {time.time()-t:.1f}s', flush=True)
        zq = torch.linspace(-8, 8, 6401, device=dev)
        wq = torch.exp(-0.5 * zq.double() ** 2); wq = wq / wq.sum()
        rows = []
        for head, res in results.items():
            s = 1.0 if head == 'upper' else -1.0
            for li, r in enumerate(res):
                for b in range(B):
                    with torch.no_grad():
                        k1, k2, m_nu = kl_state_quadrature(torch.tensor(r.spline_params_final[b], device=dev), flow, zq, wq)
                    rows.append(pd.DataFrame(dict(
                        toy_dgp=b, query=np.arange(m), config=idx, pi_query=pi, sigma=sg, mu=MUS[b], head=head,
                        lambda_index=li, lambda_value=r.lambda_value, steps=r.num_steps, stop=r.stop_reason,
                        gamma_star=r.gamma_star[b], theta_star=r.theta_star[b], kl_nu_phi=k1.cpu().numpy(),
                        kl_phi_nu=k2.cpu().numpy(), mean_nu=m_nu.cpu().numpy(), sign=s)))
        f = pd.concat(rows, ignore_index=True)
        f['gamma_true'] = np.maximum(f.kl_nu_phi, f.kl_phi_nu).clip(lower=0)
        scale = (1 - f.pi_query) * f.sigma
        exact = lambda g: f.mu + f.sign * scale * np.sqrt(2 * np.maximum(g, 0))
        f['theta_exact_star'] = exact(f.gamma_star)
        f['theta_exact_true'] = exact(f.gamma_true)
        f['theta_nu'] = f.mu + scale * f.mean_nu
        f['label_error'] = f.sign * (f.theta_star - f.theta_exact_star)
        f['mc_error'] = f.sign * (f.theta_star - f.theta_nu)
        f['objective_error'] = f.sign * (f.theta_nu - f.theta_exact_true)
        f['sensitivity_shift'] = f.sign * (f.theta_exact_true - f.theta_exact_star)
        f['label_error_rel'] = f.label_error / scale
        f.to_parquet(run.path / 'points.parquet', index=False)
        run.done(points=len(f), max_objective_error=float(f.objective_error.max()))
    except BaseException:
        run.done('failed'); raise

def linearised_dataset(dataset_id, device):
    """Observed context + exact references for one fresh DGP linearised in U (see module doc)."""
    from setting_standard.causaldataset import load_outcome_model_bundle, build_outcome_model_from_bundle
    pool = input_path('raw_root')
    raw = pd.read_csv(pool / 'datasets_raw' / f'synthetic_continous_dataset_{dataset_id}.csv')
    norm = pd.read_csv(normalized_root() / 'datasets' / f'synthetic_continous_dataset_{dataset_id}.csv')
    xraw = [c for c in raw.columns if c.startswith('x') and c[1:].isdigit()]
    assert np.array_equal(raw.treatment.to_numpy(), norm.treatment.to_numpy())
    model = build_outcome_model_from_bundle(load_outcome_model_bundle(str(pool / 'models_raw' / f'outcome_bnn_{dataset_id}.pt')), device=device)
    X = torch.tensor(raw[xraw].to_numpy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        f = {(arm, u): model(X, torch.full((len(X), 1), float(arm), device=device),
                             torch.full((len(X), 1), float(u), device=device)).reshape(-1).double().cpu().numpy()
             for arm in (0, 1) for u in (-1, 0, 1)}
    floor = 0.05 * raw.outcome.std(ddof=0)
    mu = {arm: f[(arm, 0)] for arm in (0, 1)}
    sigma = {arm: np.maximum(np.abs(f[(arm, 1)] - f[(arm, -1)]) / 2, floor) for arm in (0, 1)}
    rng = np.random.default_rng(20261107 + dataset_id)
    A = raw.treatment.to_numpy().astype(int)
    eps = rng.standard_normal(len(A))
    y = np.where(A == 1, mu[1] + sigma[1] * eps, mu[0] + sigma[0] * eps).astype(np.float32)
    y_mean = float(y.mean()); y_std = float(y.std())
    frame = norm.copy()
    frame['outcome'] = (y - y_mean) / y_std
    return frame, dict(mu=mu, sigma=sigma, y_mean=y_mean, y_std=y_std, floor=floor,
                       floor_rate=float(np.mean([np.mean(sigma[k] == floor) for k in (0, 1)])))

def pfn(smoke):
    from experiments.shared.evaluation import load_pifm, make_query_grid, pifm_predict_frame, add_gmm_diagnostics, propensity_stratified_row_indices, KL_CHECKPOINT
    ids = cohort_ids('msm')[:2 if smoke else 64]
    run = Run('e04', 'pfn_smoke' if smoke else 'pfn', smoke=smoke, gpu=True, threads=2, ids=ids,
              gammas=PFN_GAMMAS, checkpoint=str(KL_CHECKPOINT), rows_per_dgp=8)
    dev = torch.device('cuda:0')
    model, _ = load_pifm(KL_CHECKPOINT, dev)
    quantiles = np.array([0.02, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.98])
    out = []
    try:
        for i in ids:
            frame, d = linearised_dataset(i, 'cuda:0')
            assert abs(frame.outcome.mean()) < 1e-5 and abs(frame.outcome.std(ddof=0) - 1) < 1e-4
            from setting_standard.causaldataset import _normalize_x_columns, _get_x_columns
            frame = _normalize_x_columns(frame); xcols = _get_x_columns(frame)
            rows = propensity_stratified_row_indices(frame, 8, quantiles)
            q = make_query_grid(frame, xcols, rows, PFN_GAMMAS)
            r = q.source_row_index.to_numpy(); arm = q.a_query.to_numpy()
            mu = np.where(arm == 1, d['mu'][1][r], d['mu'][0][r]); sg = np.where(arm == 1, d['sigma'][1][r], d['sigma'][0][r])
            half = (1 - q.pi_query.to_numpy()) * sg * np.sqrt(2 * q.gamma.to_numpy())
            q['theta_upper'] = (mu + half - d['y_mean']) / d['y_std']
            q['theta_lower'] = (mu - half - d['y_mean']) / d['y_std']
            q['q0'] = (mu - d['y_mean']) / d['y_std']
            pred = pifm_predict_frame(model, frame, xcols, q, dev)
            pred = add_gmm_diagnostics(pred, 'theta_upper')
            from experiments.shared.evaluation import gmm_cdf, gmm_arrays
            pred['lower_pit'] = gmm_cdf(pred.theta_lower.to_numpy(), *gmm_arrays(pred, 'lower'))
            pred['dataset_id'] = i; pred['sigma_obs'] = d['y_std']; pred['sigma_floor_rate'] = d['floor_rate']
            out.append(pred)
            print(f'pfn dgp {i} done', flush=True)
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(run.path / 'predictions.parquet', index=False)
        run.done(rows=len(res))
    except BaseException:
        run.done('failed'); raise


# Verify formula


GRID = np.linspace(-14, 14, 28001)
DZ = GRID[1] - GRID[0]
PHI = norm.pdf(GRID)

def kl_pair(nu):
    """(KL(nu||phi), KL(phi||nu)) of a density on GRID by quadrature (natural log)."""
    nu = nu / (nu.sum() * DZ)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(nu > 0, nu * np.log(nu / PHI), 0.0).sum() * DZ
        b = np.where(PHI > 0, PHI * np.log(PHI / np.maximum(nu, 1e-300)), 0.0).sum() * DZ
    return a, b, (GRID * nu).sum() * DZ

def discrete_program(gamma, n=161):
    """Maximize E_nu U over nu supported on n points, s.t. both discrete KL constraints <= gamma."""
    z = np.linspace(-9, 9, n)
    p = norm.pdf(z); p /= p.sum()
    def unpack(w):
        q = np.exp(w - w.max()); return q / q.sum()
    def kl12(w):
        q = unpack(w)
        return (q * np.log(q / p)).sum(), (p * np.log(p / q)).sum()
    cons = [dict(type='ineq', fun=lambda w: gamma - kl12(w)[0]),
            dict(type='ineq', fun=lambda w: gamma - kl12(w)[1])]
    w0 = np.log(p) + 0.5 * np.sqrt(2 * gamma) * z
    res = minimize(lambda w: -(unpack(w) * z).sum(), w0, constraints=cons, method='SLSQP',
                   options=dict(maxiter=500, ftol=1e-12))
    q = unpack(res.x)
    return float((q * z).sum()), *map(float, kl12(res.x)), bool(res.success)

def verify_formula():
    rows = []
    for gamma in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        s = np.sqrt(2 * gamma)
        k1, k2, m = kl_pair(norm.pdf(GRID, loc=s))
        rows.append(dict(check='shifted_normal', gamma=gamma, bound=s, mean=m, kl_nu_phi=k1, kl_phi_nu=k2,
                         passed=abs(k1 - gamma) < 1e-8 and abs(k2 - gamma) < 1e-8 and abs(m - s) < 1e-8))
        m_opt, d1, d2, ok = discrete_program(gamma)
        # SLSQP's success flag is recorded but not required: the check is that a feasible
        # arbitrary discrete nu gets within 0.1% of the bound and never exceeds it.
        rows.append(dict(check='discrete_convex_program', gamma=gamma, bound=s, mean=m_opt, kl_nu_phi=d1,
                         kl_phi_nu=d2, solver_success=ok, passed=max(d1, d2) <= gamma * (1 + 1e-5) + 1e-9
                         and 0.999 * s <= m_opt <= s * (1 + 1e-6)))
    rng = np.random.default_rng(20261107)
    worst = -np.inf
    for _ in range(20000):
        kind = rng.integers(3)
        if kind == 0:
            w = rng.dirichlet(np.ones(3)); locs = rng.normal(0, 2, 3); scales = rng.uniform(0.3, 2.5, 3)
            nu = sum(wi * norm.pdf(GRID, l, sc) for wi, l, sc in zip(w, locs, scales))
        elif kind == 1:
            from scipy.stats import skewnorm
            nu = skewnorm.pdf(GRID, rng.normal(0, 5), rng.normal(0, 1.5), rng.uniform(0.4, 2.0))
        else:
            from scipy.stats import t as student
            nu = student.pdf(GRID, rng.uniform(2.5, 30), rng.normal(0, 1.5), rng.uniform(0.5, 1.5))
        k1, k2, m = kl_pair(nu)
        g = max(k1, k2)
        worst = max(worst, abs(m) / np.sqrt(2 * g))
    rows.append(dict(check='random_non_gaussian_nu', gamma=np.nan, bound=1.0, mean=worst, kl_nu_phi=np.nan,
                     kl_phi_nu=np.nan, passed=worst <= 1 + 1e-6))
    frame = pd.DataFrame(rows)
    (REPORTS / 'kl_gaussian').mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS / 'kl_gaussian/formula_checks.csv', index=False)
    print(frame.to_string())
    assert frame.passed.all(), 'Toy-formula verification failed'


# Verify optimized




# Report


EXPORT = REPORTS / 'kl_gaussian'

def done(name):
    p = RUNS / 'e04' / name / 'manifest.json'
    return p.exists() and read(p)['status'] == 'completed'

def toy_tables():
    out, cfg, frames, gbin = [], [], {}, []
    for b in ('prod', 'high'):
        if not done(f'toy_{b}'):
            continue
        f = pd.read_parquet(RUNS / 'e04' / f'toy_{b}' / 'points.parquet')
        f['budget'] = b
        frames[b] = f
        for (head,), g in f.groupby(['head']):
            e = g.label_error
            out.append(dict(budget=b, head=head, points=len(g), gamma_star_median=g.gamma_star.median(),
                            gamma_star_p99=g.gamma_star.quantile(.99), label_rmse=np.sqrt((e ** 2).mean()),
                            label_rel_rmse=np.sqrt((g.label_error_rel ** 2).mean()), label_bias=e.mean(),
                            label_abs_p95=e.abs().quantile(.95), label_abs_max=e.abs().max(),
                            mc_error_sd=g.mc_error.std(), objective_error_mean=g.objective_error.mean(),
                            objective_error_p01=g.objective_error.quantile(.01),
                            objective_error_max=g.objective_error.max(),
                            gamma_true_over_star_median=(g.gamma_true / g.gamma_star).median(),
                            gamma_star_rel_error_p95=((g.gamma_star - g.gamma_true).abs() / g.gamma_true.clip(lower=1e-4)).quantile(.95),
                            stop_early_rate=(g.stop == 'early_stop').mean()))
        for keys, g in f.groupby(['head', 'pi_query', 'sigma']):
            cfg.append(dict(budget=b, head=keys[0], pi=keys[1], sigma=keys[2], points=len(g),
                            gamma_star_median=g.gamma_star.median(),
                            label_rel_rmse=np.sqrt((g.label_error_rel ** 2).mean()),
                            label_rel_bias=g.label_error_rel.mean(),
                            objective_rel_mean=(g.objective_error / ((1 - g.pi_query) * g.sigma)).mean(),
                            mc_rel_sd=(g.mc_error / ((1 - g.pi_query) * g.sigma)).std()))
        f['gamma_bin'] = pd.cut(f.gamma_star, [0, 0.05, 0.25, 1.0, 2.5, 10.0, np.inf], right=False).astype(str)
        for keys, g in f.groupby(['head', 'gamma_bin']):
            e = g.label_error
            gbin.append(dict(budget=b, head=keys[0], gamma_bin=keys[1], points=len(g),
                             label_rmse=np.sqrt((e ** 2).mean()), label_bias=e.mean(),
                             label_abs_p95=e.abs().quantile(.95), label_rel_rmse=np.sqrt((g.label_error_rel ** 2).mean()),
                             objective_error_mean=g.objective_error.mean(), objective_error_max=g.objective_error.max(),
                             objective_rel_mean=(g.objective_error / ((1 - g.pi_query) * g.sigma)).mean(),
                             mc_error_sd=g.mc_error.std(), sensitivity_shift_mean=g.sensitivity_shift.mean()))
    return pd.DataFrame(out), pd.DataFrame(cfg), frames, pd.DataFrame(gbin)

def pfn_tables():
    if not done('pfn'):
        return None, None
    p = pd.read_parquet(RUNS / 'e04' / 'pfn' / 'predictions.parquet')
    p['width_ref'] = p.theta_upper - p.theta_lower
    rows = []
    for gamma, g in [('all', p)] + list(p.groupby('gamma')):
        def stat(d):
            r = {}
            for side in ('lower', 'upper'):
                e = d[f'{side}_mean'] - d[f'theta_{side}']
                r[f'{side}_rmse'] = float(np.sqrt((e ** 2).mean()))
                r[f'{side}_bias'] = float(e.mean())
                r[f'{side}_cov90'] = float(((d[f'{side}_q05'] <= d[f'theta_{side}']) & (d[f'theta_{side}'] <= d[f'{side}_q95'])).mean())
            r['outward_joint_containment'] = float(((d.lower_q05 <= d.theta_lower) & (d.upper_q95 >= d.theta_upper)).mean())
            r['mean_abs_err_over_mean_width'] = float(((d.lower_mean - d.theta_lower).abs().mean() + (d.upper_mean - d.theta_upper).abs().mean()) / 2 / d.width_ref.mean())
            return r
        ci = cluster_bootstrap(g[['dataset_id', 'lower_mean', 'upper_mean', 'theta_lower', 'theta_upper', 'lower_q05', 'lower_q95', 'upper_q05', 'upper_q95', 'width_ref']],
                               'dataset_id', stat, replicates=1000)
        rows.append(dict(gamma=gamma, rows=len(g), dgps=g.dataset_id.nunique(), **ci))
    return pd.DataFrame(rows), p

def report():
    EXPORT.mkdir(parents=True, exist_ok=True)
    s, c, frames, gb = toy_tables()
    gb.to_csv(EXPORT / 'toy_by_gamma_bin.csv', index=False)
    s.to_csv(EXPORT / 'toy_summary.csv', index=False); c.to_csv(EXPORT / 'toy_by_config.csv', index=False)
    if len(s):
        print(s.round(4).to_string()); print(c[c['head'] == 'upper'].round(3).to_string())
    pt, p = pfn_tables()
    if pt is not None:
        pt.to_csv(EXPORT / 'pfn_summary.csv', index=False)
        print(pt[['gamma', 'rows', 'lower_rmse', 'upper_rmse', 'lower_cov90', 'upper_cov90', 'outward_joint_containment', 'mean_abs_err_over_mean_width']].round(3).to_string())
    runtime = [dict(run=n, **{k: read(RUNS / 'e04' / n / 'manifest.json').get(k) for k in ('status', 'wall_seconds')})
               for n in ('toy_smoke', 'toy_prod', 'toy_high', 'pfn_smoke', 'pfn') if (RUNS / 'e04' / n / 'manifest.json').exists()]
    pd.DataFrame(runtime).to_csv(EXPORT / 'runtime.csv', index=False)

    if frames:
        fig, axes = plt.subplots(1, len(frames), figsize=(5.2 * len(frames), 3.8), squeeze=False, sharey=True)
        for ax, (b, f) in zip(axes[0], frames.items()):
            for (sg, color) in zip(sorted(f.sigma.unique()), ('#1b9e77', '#7570b3', '#d95f02')):
                g = f[(f.sigma == sg) & (f.pi_query == 0.5) & (f['head'] == 'upper') & (f.toy_dgp == 0)]
                xs = np.linspace(0, g.gamma_star.max(), 200)
                ax.plot(xs, 0.5 * sg * np.sqrt(2 * xs), color=color, lw=1)
                ax.scatter(g.gamma_star, g.theta_star - g.mu, s=2, alpha=.25, color=color, label=f'sigma={sg}')
            ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_xlabel('reported Gamma* (KL)'); ax.set_title(f'{b}: upper, pi=0.5, mu=0')
        axes[0][0].set_ylabel('theta* - mu (line: exact)'); axes[0][0].legend(fontsize=7, markerscale=4)
        fig.tight_layout(); fig.savefig(EXPORT / 'toy_frontiers.png', dpi=130); plt.close(fig)
    if pt is not None:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for side, color in (('lower', '#d62728'), ('upper', '#1f77b4')):
            ax.scatter(p[f'theta_{side}'], p[f'{side}_mean'], s=3, alpha=.3, color=color, label=side)
        lim = [p[['theta_lower', 'theta_upper']].min().min(), p[['theta_lower', 'theta_upper']].max().max()]
        ax.plot(lim, lim, 'k', lw=.8); ax.set_xlabel('exact KL bound (std. units)'); ax.set_ylabel('KL PI-FM mean')
        ax.legend(fontsize=7, markerscale=4); fig.tight_layout(); fig.savefig(EXPORT / 'pfn_vs_exact.png', dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['toy', 'pfn', 'verify-formula', 'report'])
    parser.add_argument('--budget', default='prod')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'toy': toy(args.budget)
    elif args.stage == 'pfn': pfn(args.smoke)
    elif args.stage == 'verify-formula': verify_formula()
    else: report()


if __name__ == '__main__':
    main()
