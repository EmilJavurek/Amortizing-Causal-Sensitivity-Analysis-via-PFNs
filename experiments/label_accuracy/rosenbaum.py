"""Sharp Rosenbaum CAPO references, independent checks, optimizer sweeps and dense relabeling."""


from __future__ import annotations

from experiments.shared.paths import REPORTS
import argparse
import base64
import json
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time
import torch
import yaml
from experiments.label_accuracy.msm import IDS, production_config, spline_states
from experiments.shared.paths import CODE, OBSERVED_DATA, RUNS
from experiments.shared.runtime import Run, read
from experiments.shared.inputs import cohort_artifacts
from experiments.shared.msm_reference import msm_reference_from_samples
from pathlib import Path
from scipy.optimize import brentq, linprog
from scipy.special import ndtri
from scipy.stats import norm


# Rosenbaum


def rosenbaum_nu_mean_from_draws(draws, gamma, upper=True):
    """Max (upper) or min (lower) of E_nu[Y] over Rosenbaum(gamma), from equal-weight draws.

    Exact for the empirical law: on each segment between breakpoints alpha = k/n the objective is a
    linear-fractional function of alpha, hence monotone, so the optimum is attained at a breakpoint.
    draws: (rows, n) array. Returns (value, alpha_star) arrays of shape (rows,).
    """
    y = np.asarray(draws, dtype=np.float64)
    if not upper:
        value, alpha = rosenbaum_nu_mean_from_draws(-y, gamma, upper=True)
        return -value, alpha
    rows, n = y.shape
    gamma = float(gamma)
    assert gamma >= 1.0
    desc = -np.sort(-y, axis=1)
    top = np.concatenate([np.zeros((rows, 1)), np.cumsum(desc, axis=1)], axis=1) / n   # T(k/n)
    mu = top[:, -1]
    alpha = np.arange(n + 1) / n
    value = (mu[:, None] + (gamma - 1.0) * top) / (1.0 + (gamma - 1.0) * alpha[None, :])
    k = np.argmax(value, axis=1)
    return value[np.arange(rows), k], alpha[k]

def rosenbaum_capo_from_draws(draws, pi_query, gammas):
    """(lower, upper) CAPO bounds, arrays (rows, len(gammas)); pi_query = queried-arm propensity."""
    draws = np.asarray(draws, dtype=np.float64)
    pi = np.asarray(pi_query, dtype=np.float64)
    assert pi.shape == (draws.shape[0],)
    mu = draws.mean(axis=1)
    lower = np.empty((draws.shape[0], len(gammas)))
    upper = np.empty_like(lower)
    for j, g in enumerate(gammas):
        hi, _ = rosenbaum_nu_mean_from_draws(draws, g, upper=True)
        lo, _ = rosenbaum_nu_mean_from_draws(draws, g, upper=False)
        upper[:, j] = pi * mu + (1 - pi) * hi
        lower[:, j] = pi * mu + (1 - pi) * lo
    assert np.all(lower <= upper + 1e-12)
    return lower, upper

def gaussian_z(gamma):
    """Root z* of z = (Gamma - 1)(phi(z) - z (1 - Phi(z))); z* = 0 at Gamma = 1."""
    if gamma == 1.0:
        return 0.0
    f = lambda z: z - (gamma - 1.0) * (norm.pdf(z) - z * norm.sf(z))
    return brentq(f, 0.0, 50.0, xtol=1e-14)

def gaussian_capo(mu, sigma, pi, gamma):
    z = gaussian_z(gamma)
    return mu - (1 - pi) * sigma * z, mu + (1 - pi) * sigma * z

def gaussian_msm_capo(mu, sigma, pi, gamma):
    """MSM closed form for Y ~ N(mu, sigma^2): r = Gamma on the top 1/(1+Gamma) tail, 1/Gamma below."""
    tau = gamma / (1.0 + gamma)
    q = norm.ppf(tau)
    cvar = norm.pdf(q) / (1 - tau)                      # E[Z | Z > q]
    nu_shift = (1 - 1 / gamma) * cvar                   # (1/Gamma) E[Z] + (1 - 1/Gamma) CVaR, E[Z] = 0
    return mu - (1 - pi) * sigma * nu_shift, mu + (1 - pi) * sigma * nu_shift

def rosenbaum_delta_from_log_r(log_r, include_identity=True):
    """True Delta = sup r / inf r on a dense grid; include r = 1 where the flow is the identity."""
    log_r = np.asarray(log_r, dtype=np.float64)
    hi, lo = log_r.max(axis=-1), log_r.min(axis=-1)
    if include_identity:
        hi, lo = np.maximum(hi, 0.0), np.minimum(lo, 0.0)
    return np.exp(hi - lo)


# Checks

EXPORT = REPORTS / 'rosenbaum_labels'

def lp_nu_mean(y, gamma):
    """Independent LP: max (1/n) sum r_i y_i s.t. (1/n) sum r_i = 1, m <= r_i <= gamma m, m >= 0."""
    n = len(y)
    c = np.r_[-y / n, 0.0]
    a_eq = np.r_[np.ones(n) / n, 0.0][None]
    a_ub = np.zeros((2 * n, n + 1))
    a_ub[:n, :n] = -np.eye(n); a_ub[:n, n] = 1.0              # m - r_i <= 0
    a_ub[n:, :n] = np.eye(n); a_ub[n:, n] = -gamma            # r_i - gamma m <= 0
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(2 * n), A_eq=a_eq, b_eq=[1.0], bounds=[(0, None)] * (n + 1),
                  method='highs')
    assert res.status == 0, res.message
    r = res.x[:n]
    return -res.fun, r

def check_bathtub(rng, cases):
    rows = []
    for case in range(cases):
        n = int(rng.integers(5, 60))
        y = rng.choice([rng.normal(size=n), rng.lognormal(size=n), np.r_[rng.normal(-2, .3, n // 2), rng.normal(1, 1, n - n // 2)]])
        for g in (1.0, 1.3, 2.0, 4.0, 9.0, 25.0):
            closed, alpha = rosenbaum_nu_mean_from_draws(y[None], g)
            lp, r = lp_nu_mean(y, g)
            ratio = r.max() / r.min()
            distinct = len(np.unique(np.round(r / r.min(), 6)))
            rows.append(dict(case=case, n=n, gamma=g, closed=closed[0], lp=lp, abs_diff=abs(closed[0] - lp),
                             lp_ratio=ratio, lp_distinct_levels=distinct, alpha=alpha[0]))
            assert abs(closed[0] - lp) < 1e-7 * (1 + abs(lp)), (case, g, closed, lp)
            assert ratio <= g + 1e-6
    return pd.DataFrame(rows)

def check_gaussian():
    grid = ndtri((np.arange(65536) + 0.5) / 65536)
    rows = []
    for mu, sigma, pi in ((0.0, 1.0, 0.5), (1.3, 0.4, 0.2), (-0.7, 2.0, 0.85)):
        y = (mu + sigma * grid)[None]
        for g in (1.0, 1.5, 2.0, 4.0, 9.0, 25.0, 100.0):
            lo_c, up_c = gaussian_capo(mu, sigma, pi, g)
            lo_e, up_e = rosenbaum_capo_from_draws(y, np.array([pi]), [g])
            msm_lo, msm_up = gaussian_msm_capo(mu, sigma, pi, g)
            msm_lo_e, msm_up_e = msm_reference_from_samples(y, np.array([pi]), np.array([g]))
            rows.append(dict(mu=mu, sigma=sigma, pi=pi, gamma=g, z_star=gaussian_z(g),
                             upper_closed=up_c, upper_empirical=up_e[0, 0], lower_closed=lo_c, lower_empirical=lo_e[0, 0],
                             msm_upper_closed=msm_up, msm_upper_empirical=msm_up_e[0, 0]))
            tol = 2e-4 * sigma * max(1.0, gaussian_z(g))
            assert abs(up_c - up_e[0, 0]) < tol and abs(lo_c - lo_e[0, 0]) < tol, (mu, sigma, pi, g, up_c, up_e)
            assert abs(msm_up - msm_up_e[0, 0]) < 5e-4 * sigma, (g, msm_up, msm_up_e)
    return pd.DataFrame(rows)

def check_structure(rng):
    rows = []
    laws = {'normal': rng.normal(size=20000), 'lognormal': rng.lognormal(0, 0.8, 20000),
            'bimodal': np.r_[rng.normal(-2, .4, 8000), rng.normal(1.5, 1, 12000)],
            'discrete': rng.choice([-1.0, 0.0, 2.0], size=20000, p=[.3, .5, .2])}
    gammas = np.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    for name, y in laws.items():
        for pi in (0.1, 0.5, 0.9):
            d = y[None]
            lo, up = rosenbaum_capo_from_draws(d, np.array([pi]), list(gammas))
            lo2, up2 = rosenbaum_capo_from_draws(d, np.array([pi]), list(gammas ** 2))
            mlo, mup = msm_reference_from_samples(d, np.array([pi]), gammas)
            mu = y.mean()
            assert abs(lo[0, 0] - mu) < 1e-10 and abs(up[0, 0] - mu) < 1e-10           # Gamma = 1
            assert np.all(np.diff(up[0]) >= -1e-12) and np.all(np.diff(lo[0]) <= 1e-12)  # monotone
            assert np.all(up[0] <= mup[0] + 1e-9) and np.all(mup[0] <= up2[0] + 1e-9)    # nesting
            assert np.all(lo[0] >= mlo[0] - 1e-9) and np.all(mlo[0] >= lo2[0] - 1e-9)
            for j, g in enumerate(gammas):
                t, _ = rosenbaum_nu_mean_from_draws(d, g)
                fp = (g - 1) * np.mean(np.maximum(y - t[0], 0)) - (t[0] - mu)
                rows.append(dict(law=name, pi=pi, gamma=g, ros_lower=lo[0, j], ros_upper=up[0, j],
                                 msm_lower=mlo[0, j], msm_upper=mup[0, j], ros_sq_lower=lo2[0, j],
                                 ros_sq_upper=up2[0, j], fixed_point_residual=fp))
                assert abs(fp) < 5e-3 * (1 + np.std(y)), (name, g, fp)   # breakpoint discretization
    return pd.DataFrame(rows)

def check_frontier_estimator(rng, smoke):
    import torch
    from setting_standard.frontier_optimization import FlowConfig, batched_spline_forward, estimate_divergence_batched, standard_normal_logprob
    rows = []
    # (a) analytic nu: r(u) = c (1 + (G0 - 1) sigmoid(2u)), sup/inf -> G0 (limits, not attained).
    for g0 in (2.0, 5.0, 25.0):
        ugrid = np.linspace(-40, 40, 400001)
        s = 1 / (1 + np.exp(-2 * ugrid))
        w = np.exp(-ugrid ** 2 / 2) / np.sqrt(2 * np.pi)
        c = 1 / np.trapezoid((1 + (g0 - 1) * s) * w, ugrid)
        for S in (128, 1024, 4096):
            est = []
            for rep in range(20 if smoke else 200):
                z = rng.normal(size=8 * S)
                rr = c * (1 + (g0 - 1) / (1 + np.exp(-2 * z)))
                keep = rng.uniform(size=z.size) < rr / (c * g0)          # rejection sampling from nu
                u = z[keep][:S]
                assert len(u) == S
                log_r = np.log(c * (1 + (g0 - 1) / (1 + np.exp(-2 * u))))
                lt = torch.tensor(log_r, dtype=torch.float64)[None, None]
                e = float(estimate_divergence_batched(lt, 'rosenbaum')[0, 0])
                assert abs(e - np.exp(log_r.max() - log_r.min())) < 1e-9 * e   # implementation == numpy
                est.append(e)
            est = np.array(est)
            rows.append(dict(source='analytic_sigmoid', gamma_true=g0, samples=S, est_median=np.median(est),
                             ratio_q05=np.quantile(est, .05) / g0, ratio_q95=np.quantile(est, .95) / g0,
                             median_ratio=np.median(est) / g0))
    # (b) saved MSM-sweep spline states: true sup/inf on a dense grid (identity outside the tail
    #     bound) versus the frontier_optimization estimator on S fresh nu-samples, per state.
    candidates = [RUNS / 'e03' / b / 'frontier' / 'spline_params' / 'spline_params_1.npz'
                  for b in ('sweep_prod_msmgrid', 'sweep_prod', 'sweep_smoke')]
    candidates = [p for p in candidates if p.exists() and (p.parents[2] / 'manifest.json').exists()]
    if candidates:
        state_path = candidates[0]
        state = np.load(state_path)
        flow = FlowConfig(num_bins=8, tail_bound=6.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3)
        params = torch.tensor(state['upper_spline_params'][:, :32], dtype=torch.float32)  # [L, 32 queries, d]
        params = params.reshape(-1, params.shape[-1])
        z = torch.linspace(-6, 6, 20001)[None, None].expand(1, params.shape[0], -1)
        u, log_p = batched_spline_forward(z, params[None], flow)
        true_log_r = (log_p - standard_normal_logprob(u))[0].double().numpy()
        true_delta = rosenbaum_delta_from_log_r(true_log_r)
        for S in (128, 1024):
            base = torch.randn(1, params.shape[0], S)
            us, lp = batched_spline_forward(base, params[None], flow)
            est = estimate_divergence_batched(lp - standard_normal_logprob(us), 'rosenbaum')[0].double().numpy()
            ratio = est / true_delta
            rows.append(dict(source=f'e03_spline_states:{state_path.parents[2].name}', gamma_true=float(np.median(true_delta)), samples=S,
                             est_median=float(np.median(est)), ratio_q05=float(np.quantile(ratio, .05)),
                             ratio_q95=float(np.quantile(ratio, .95)), median_ratio=float(np.median(ratio)),
                             states=int(params.shape[0]), frac_underestimated=float(np.mean(ratio < 1 - 1e-6))))
            assert np.all(ratio <= 1 + 1e-4), ratio.max()                  # samples cannot exceed sup/inf
    return pd.DataFrame(rows)

def lambda_scale():
    rows = []
    gammas = np.array([1.001, 1.05, 1.2, 1.5, 2, 3, 5, 9, 16, 25, 50, 100])
    z = np.array([gaussian_z(g) for g in gammas])
    for g in gammas:
        h = 1e-4 * g
        slope = (gaussian_z(g + h) - gaussian_z(g - h)) / (2 * h)
        rows.append(dict(gamma=g, z_star=gaussian_z(g), dz_dgamma=slope))
    frame = pd.DataFrame(rows)
    second = np.diff(frame.dz_dgamma) / np.diff(gammas)
    frame['concave_so_far'] = np.r_[True, second <= 1e-9]
    assert np.all(second <= 1e-9), 'Gaussian Rosenbaum frontier not concave in Gamma'
    return frame

def checks(smoke):
    run = Run('e13_rosenbaum_oracle', 'checks_smoke' if smoke else 'checks', smoke=smoke, gpu=False, threads=4)
    rng = np.random.default_rng(20261107)
    try:
        out = dict(bathtub=check_bathtub(rng, 5 if smoke else 60), gaussian=check_gaussian(),
                   structure=check_structure(rng), frontier_estimator=check_frontier_estimator(rng, smoke),
                   lambda_scale=lambda_scale())
        if not smoke:
            EXPORT.mkdir(parents=True, exist_ok=True)
        for name, frame in out.items():
            frame.to_csv(run.path / f'checks_{name}.csv', index=False)
            if not smoke:
                frame.to_csv(EXPORT / f'checks_{name}.csv', index=False)
        summary = dict(bathtub_cases=len(out['bathtub']), bathtub_max_abs_diff=float(out['bathtub'].abs_diff.max()),
                       bathtub_lp_levels_max=int(out['bathtub'].lp_distinct_levels.max()),
                       gaussian_max_upper_err=float((out['gaussian'].upper_closed - out['gaussian'].upper_empirical).abs().max()),
                       fixed_point_max_residual=float(out['structure'].fixed_point_residual.abs().max()))
        print(summary)
        print(out['frontier_estimator'].round(4).to_string())
        print(out['lambda_scale'].round(4).to_string())
        run.done(**summary)
    except BaseException:
        run.done('failed')
        raise


# Pilot


PACKAGE = 'e13_rosenbaum_oracle'
SOURCE = OBSERVED_DATA / 'fresh_observed'
# Rosenbaum lambda grid, justified by checks_lambda_scale.csv: the Gaussian frontier slope
# d theta+/d Gamma = (1 - pi) sigma dz*/dGamma runs from 0.40 (1-pi)sigma at Gamma -> 1 to
# 0.0144 (1-pi)sigma at Gamma = 25; with (1-pi)sigma in ~[0.05, 1.5] (observed-standardized),
# lambda in [1e-3, 1] spans Gamma from ~1 to beyond 25.
ROSENBAUM_GRID = dict(lambda_min=0.001, lambda_max=1.0, lambda_n=50, lambda_spacing='log')
REFERENCE_GRID = 65536

def budget(name):
    cfg = production_config()
    cfg.update(sensitivity_model='rosenbaum', save_spline_params=True, **ROSENBAUM_GRID)
    if name == 'smoke':
        cfg.update(num_query_points=4, lambda_n=8, max_steps=100, early_stop=False, dgp_batch_size=2)
        return cfg, IDS[:2]
    if name == 'pilot':          # for the later phase: production budget on 8 DGPs, 64 query rows
        cfg.update(num_query_points=64, dgp_batch_size=8)
        return cfg, IDS[:8]
    if name == 'audit':          # exact production label grid/budget (paper Rosenbaum frontier recipe) on the pilot DGPs
        cfg.update(num_query_points=64, dgp_batch_size=8, lambda_min=0.0015, lambda_max=0.1, lambda_n=40)
        return cfg, IDS[:8]
    raise ValueError(name)

def sweep(name):
    cfg, ids = budget(name)
    run = Run(PACKAGE, f'sweep_{name}', smoke=name == 'smoke', gpu=True, threads=2, budget=name,
              dataset_ids=ids, frontier=cfg)
    run_dir = run.path / 'frontier'
    full = dict(source_run_dir=str(SOURCE), run_dir=str(run_dir), dataset_ids=list(ids), **cfg)
    config_bytes = yaml.safe_dump(full, sort_keys=False).encode()
    (run.path / 'input.yaml').write_bytes(config_bytes)

    from experiments.scripts._run import config_to_args
    cmd = [sys.executable, str(CODE / 'src/setting_standard/frontier_optimization.py')] + config_to_args(yaml.safe_load(config_bytes))
    env = os.environ.copy()
    env.update(PI_FM_PIPELINE_STEP='03_frontier', PI_FM_CONFIG_PATH=str(run.path / 'input.yaml'),
               PI_FM_CONFIG_BYTES_BASE64=base64.b64encode(config_bytes).decode(),
               PI_FM_LAUNCHER_ARGV_JSON=json.dumps([sys.executable, *sys.argv]),
               PI_FM_TARGET_ARGV_JSON=json.dumps(cmd),
               PI_FM_GIT_PROVENANCE_JSON=json.dumps({k: run.manifest['git'][k] for k in ('commit', 'dirty')}))
    started = time.time()
    code = subprocess.run(cmd, cwd=CODE, env=env).returncode
    if code != 0:
        run.done('failed', returncode=code)
        raise SystemExit(code)
    run.done(frontier_wall_seconds=time.time() - started, run_dir=str(run_dir))

def rosenbaum_bound_rows(top, mu, n, pi, gamma, upper):
    """Sharp Rosenbaum CAPO per row at per-row gamma; `top[r, k]` = T(k/n) of the bound direction."""
    alpha = torch.arange(n + 1, device=top.device, dtype=torch.float64) / n
    g = gamma[:, None]
    nu = ((mu[:, None] if upper else -mu[:, None]) + (g - 1.0) * top) / (1.0 + (g - 1.0) * alpha[None])
    best = nu.max(dim=1).values
    best = best if upper else -best
    return pi * mu + (1 - pi) * best

def analyze(name):
    sweep_dir = RUNS / PACKAGE / f'sweep_{name}'
    manifest = read(sweep_dir / 'manifest.json')
    assert manifest['status'] == 'completed', manifest['status']
    run = Run(PACKAGE, f'analyze_{name}', smoke=manifest['smoke'], gpu=True, threads=2, sweep=str(sweep_dir),
              reference_grid=REFERENCE_GRID, base_grid=4001, quadrature_grid=3201)
    from scipy.special import ndtri
    from setting_standard.frontier_optimization import FlowConfig
    from setting_standard.causaldataset import (load_outcome_model_bundle, build_outcome_model_from_bundle,
                                                load_propensity_model_bundle, build_propensity_model_from_bundle)
    frontier = sweep_dir / 'frontier'
    artifacts = cohort_artifacts('msm')
    flow = FlowConfig(**read(frontier / 'metadata_frontier/frontier_run_config.json')['config']['flow'])
    dev = torch.device('cuda:0')
    z = torch.linspace(-flow.tail_bound, flow.tail_bound, 4001, device=dev)
    zq = torch.linspace(-8.0, 8.0, 3201, device=dev, dtype=torch.float64)
    wq = torch.exp(-0.5 * zq ** 2) * (zq[1] - zq[0])
    wq[0] *= 0.5; wq[-1] *= 0.5
    wq = wq / wq.sum()
    grid_u = torch.tensor(ndtri((np.arange(REFERENCE_GRID) + 0.5) / REFERENCE_GRID), dtype=torch.float32, device=dev)
    (run.path / 'points').mkdir()
    try:
        for dataset_id in manifest['config']['dataset_ids']:
            artifact = artifacts[dataset_id]
            outcome = build_outcome_model_from_bundle(load_outcome_model_bundle(artifact['outcome_model_path']), device='cuda:0')
            propensity = build_propensity_model_from_bundle(load_propensity_model_bundle(artifact['propensity_model_path']), device='cuda:0')
            queries = pd.read_csv(frontier / 'queries' / f'queries_{dataset_id}.csv')
            state = np.load(frontier / 'spline_params' / f'spline_params_{dataset_id}.npz')
            xcols = [c for c in queries.columns if c.startswith('x_')]
            x = torch.tensor(queries[xcols].to_numpy(), dtype=torch.float32, device=dev)
            a = torch.tensor(queries.a_query.to_numpy(), dtype=torch.float32, device=dev)[:, None]
            m = len(queries)
            with torch.no_grad():
                p1 = propensity(x).reshape(-1).double()
                y = outcome(x[:, None, :].expand(-1, REFERENCE_GRID, -1).reshape(-1, x.shape[1]),
                            a[:, None, :].expand(-1, REFERENCE_GRID, -1).reshape(-1, 1),
                            grid_u[None].expand(m, -1).reshape(-1, 1)).reshape(m, REFERENCE_GRID).double()
            pi = torch.where(a[:, 0] == 1, p1, 1 - p1)
            asc = torch.sort(y, dim=1).values
            zero = torch.zeros(m, 1, device=dev, dtype=torch.float64)
            top_up = torch.cat([zero, torch.cumsum(asc.flip(1), 1)], 1) / REFERENCE_GRID
            top_lo = torch.cat([zero, torch.cumsum(-asc, 1)], 1) / REFERENCE_GRID
            mu = asc.mean(1)
            out = []
            for head in ('upper', 'lower'):
                params = torch.tensor(state[f'{head}_spline_params'], device=dev)[:, :m]
                theta_star = torch.tensor(state[f'{head}_theta_star'], device=dev).double()[:, :m]
                gamma_star = torch.tensor(state[f'{head}_gamma_star'], device=dev).double()[:, :m]
                up = head == 'upper'
                for li in range(params.shape[0]):
                    with torch.no_grad():
                        _, log_r = spline_states(params[li], flow, z)
                        log_r = log_r.double()
                        delta_true = torch.exp(log_r.max(1).values.clamp(min=0.0) - log_r.min(1).values.clamp(max=0.0))
                        u_q, _ = spline_states(params[li], flow, zq.float())
                        xe = x[:, None, :].expand(-1, len(zq), -1).reshape(-1, x.shape[1])
                        ae = a[:, None, :].expand(-1, len(zq), -1).reshape(-1, 1)
                        y_nu = outcome(xe, ae, u_q.reshape(-1, 1)).reshape(m, -1).double()
                        y_phi = outcome(xe, ae, zq.float()[None].expand(m, -1).reshape(-1, 1)).reshape(m, -1).double()
                        theta_nu = pi * (y_phi @ wq) + (1 - pi) * (y_nu @ wq)
                        top = top_up if up else top_lo
                        ref_true = rosenbaum_bound_rows(top, mu, REFERENCE_GRID, pi, delta_true, up)
                        ref_star = rosenbaum_bound_rows(top, mu, REFERENCE_GRID, pi, gamma_star[li].clamp(min=1.0), up)
                    out.append(pd.DataFrame(dict(
                        dataset_id=dataset_id, query_id=queries.query_id.to_numpy(), a_query=queries.a_query.to_numpy(),
                        pi_query=pi.cpu().numpy(), head=head, lambda_index=li, lambda_value=state[f'{head}_lambda_value'][li],
                        gamma_star=gamma_star[li].cpu().numpy(), gamma_true=delta_true.cpu().numpy(),
                        theta_star=theta_star[li].cpu().numpy(), theta_nu=theta_nu.cpu().numpy(),
                        theta_ref_star=ref_star.cpu().numpy(), theta_ref_true=ref_true.cpu().numpy(),
                        q0=mu.cpu().numpy(), sign=1.0 if up else -1.0)))
            frame = pd.concat(out, ignore_index=True)
            s = frame.sign
            frame['label_error'] = s * (frame.theta_star - frame.theta_ref_star)
            frame['mc_error'] = s * (frame.theta_star - frame.theta_nu)
            frame['objective_error'] = s * (frame.theta_nu - frame.theta_ref_true)
            frame['sensitivity_shift'] = s * (frame.theta_ref_true - frame.theta_ref_star)
            frame['over_budget'] = frame.gamma_true > frame.gamma_star * (1 + 1e-6)
            frame.to_parquet(run.path / 'points' / f'dgp_{dataset_id}.parquet', index=False)
            print(f'analyzed {dataset_id}: points={len(frame)}', flush=True)
        points = pd.concat([pd.read_parquet(p) for p in sorted((run.path / 'points').glob('*.parquet'))])
        summary = dict(points=len(points), over_budget_rate=float(points.over_budget.mean()),
                       gamma_true_over_star_median=float((points.gamma_true / points.gamma_star).median()),
                       label_error_median=float(points.label_error.median()),
                       label_error_abs_p95=float(points.label_error.abs().quantile(.95)),
                       objective_error_max=float(points.objective_error.max()),
                       gamma_star_range=[float(points.gamma_star.min()), float(points.gamma_star.max())])
        print(summary)
        run.done(**summary)
    except BaseException:
        run.done('failed')
        raise


# Nonconcavity

NONCONCAVITY_GRID = np.exp(np.linspace(0.0, np.log(200.0), 2001))

def upper_hull_gap(g, t):
    """Max vertical gap between the concave majorant of (g, t) and t; exposed mask (hull vertices)."""
    hull = []
    for i in range(len(g)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            # keep concave: remove i2 if it lies below the chord i1 -> i
            if (t[i2] - t[i1]) * (g[i] - g[i1]) <= (t[i] - t[i1]) * (g[i2] - g[i1]):
                hull.pop()
            else:
                break
        hull.append(i)
    majorant = np.interp(g, g[hull], t[hull])
    exposed = np.zeros(len(g), bool)
    exposed[hull] = True
    gap = majorant - t
    # Gamma measure (log scale) strictly inside non-exposed hull segments with positive gap
    missed = 0.0
    for a, b in zip(hull[:-1], hull[1:]):
        if b > a + 1 and gap[a + 1:b].max() > 1e-12:
            missed += np.log(g[b]) - np.log(g[a])
    return gap.max(), missed, exposed

def example():
    y = np.r_[np.zeros(49), np.ones(50), np.full(1, 10.0)]           # exact equal-weight representation
    t_oracle = np.array([rosenbaum_nu_mean_from_draws(y[None], g)[0][0] for g in NONCONCAVITY_GRID])
    left = 0.6 * NONCONCAVITY_GRID / (0.49 + 0.51 * NONCONCAVITY_GRID)
    right = (0.5 + 0.1 * NONCONCAVITY_GRID) / (0.99 + 0.01 * NONCONCAVITY_GRID)
    closed = np.where(left <= 1.0, left, right)
    lp = np.array([lp_nu_mean(y, g)[0] for g in NONCONCAVITY_GRID[::100]])
    g0 = 49.0 / 9.0
    slope_l = 0.6 * 0.49 / (0.49 + 0.51 * g0) ** 2
    slope_r = (0.1 * 0.99 - 0.5 * 0.01) / (0.99 + 0.01 * g0) ** 2
    eps = 1e-6
    f = lambda g: rosenbaum_nu_mean_from_draws(y[None], g)[0][0]
    num_l = (f(g0) - f(g0 - eps)) / eps
    num_r = (f(g0 + eps) - f(g0)) / eps
    gap, missed, exposed = upper_hull_gap(NONCONCAVITY_GRID, t_oracle)
    lo = NONCONCAVITY_GRID[~exposed & (NONCONCAVITY_GRID < g0)].min() if (~exposed & (NONCONCAVITY_GRID < g0)).any() else np.nan
    hi = NONCONCAVITY_GRID[~exposed & (NONCONCAVITY_GRID > g0)].max() if (~exposed & (NONCONCAVITY_GRID > g0)).any() else np.nan
    out = dict(max_abs_oracle_minus_closed=float(np.abs(t_oracle - closed).max()),
               max_abs_lp_minus_oracle=float(np.abs(lp - t_oracle[::100]).max()),
               gamma_kink=g0, slope_left_closed=slope_l, slope_right_closed=slope_r,
               slope_left_numeric=num_l, slope_right_numeric=num_r,
               hull_gap_max=float(gap), missed_log_gamma=float(missed),
               missed_gamma_interval=[float(lo), float(hi)])
    assert out['max_abs_oracle_minus_closed'] < 1e-9 and out['max_abs_lp_minus_oracle'] < 1e-7
    assert abs(num_l - slope_l) < 1e-4 and abs(num_r - slope_r) < 1e-4 and slope_r > slope_l
    return out

def continuous():
    rng = np.random.default_rng(0)
    n = 200000
    comp = rng.uniform(size=n) < 0.02
    y = np.where(comp, rng.normal(8, 0.5, n), rng.normal(0, 1, n))
    t = np.array([rosenbaum_nu_mean_from_draws(y[None], g)[0][0] for g in NONCONCAVITY_GRID[::4]])
    gap, missed, _ = upper_hull_gap(NONCONCAVITY_GRID[::4], t)
    return dict(hull_gap_max=float(gap), missed_log_gamma=float(missed),
                concave=bool(gap < 1e-9), sd=float(y.std()))

def prior(run):
    import torch
    from scipy.special import ndtri
    from setting_standard.causaldataset import (load_outcome_model_bundle, build_outcome_model_from_bundle,
                                                load_propensity_model_bundle, build_propensity_model_from_bundle)
    frontier = RUNS / 'e13_rosenbaum_oracle' / 'sweep_audit' / 'frontier'
    artifacts = cohort_artifacts('msm')
    dev = torch.device('cuda:0')
    n = 4096
    grid_u = torch.tensor(ndtri((np.arange(n) + 0.5) / n), dtype=torch.float32, device=dev)
    g = torch.tensor(NONCONCAVITY_GRID[::2], device=dev, dtype=torch.float64)
    alpha = torch.arange(n + 1, device=dev, dtype=torch.float64) / n
    rows = []
    for dataset_id in range(1, 9):
        artifact = artifacts[dataset_id]
        outcome = build_outcome_model_from_bundle(load_outcome_model_bundle(artifact['outcome_model_path']), device='cuda:0')
        propensity = build_propensity_model_from_bundle(load_propensity_model_bundle(artifact['propensity_model_path']), device='cuda:0')
        q = pd.read_csv(frontier / 'queries' / f'queries_{dataset_id}.csv')
        xcols = [c for c in q.columns if c.startswith('x_')]
        x = torch.tensor(q[xcols].to_numpy(), dtype=torch.float32, device=dev)
        a = torch.tensor(q.a_query.to_numpy(), dtype=torch.float32, device=dev)[:, None]
        m = len(q)
        with torch.no_grad():
            p1 = propensity(x).reshape(-1).double()
            y = outcome(x[:, None, :].expand(-1, n, -1).reshape(-1, x.shape[1]), a[:, None, :].expand(-1, n, -1).reshape(-1, 1),
                        grid_u[None].expand(m, -1).reshape(-1, 1)).reshape(m, n).double()
        pi = torch.where(a[:, 0] == 1, p1, 1 - p1)
        asc = torch.sort(y, dim=1).values
        mu = asc.mean(1)
        zero = torch.zeros(m, 1, device=dev, dtype=torch.float64)
        for head, top in (('upper', torch.cat([zero, torch.cumsum(asc.flip(1), 1)], 1) / n),
                          ('lower', torch.cat([zero, torch.cumsum(-asc, 1)], 1) / n)):
            sgn_mu = mu if head == 'upper' else -mu
            vals = []
            for gi in g:
                nu = (sgn_mu[:, None] + (gi - 1.0) * top) / (1.0 + (gi - 1.0) * alpha[None])
                vals.append(nu.max(1).values)
            best = torch.stack(vals, 1)                          # (m, G) in the head's ascending sign
            theta = (pi[:, None] * sgn_mu[:, None] + (1 - pi[:, None]) * best).cpu().numpy()
            for k in range(m):
                gap, missed, _ = upper_hull_gap(NONCONCAVITY_GRID[::2], theta[k])
                rows.append(dict(dataset_id=dataset_id, query_id=int(q.query_id.iloc[k]), head=head,
                                 pi_query=float(pi[k]), hull_gap=gap, missed_log_gamma=missed,
                                 width_at_25=float(theta[k][np.searchsorted(NONCONCAVITY_GRID[::2], 25.0)] - theta[k][0])))
        print(f'prior dgp {dataset_id} done', flush=True)
    return pd.DataFrame(rows)

def nonconcavity():
    run = Run('e13_rosenbaum_oracle', 'nonconcavity', gpu=True, threads=4)
    try:
        ex, cont = example(), continuous()
        frame = prior(run)
        EXPORT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(EXPORT / 'nonconcavity_prior_queries.csv', index=False)
        summary = dict(example=ex, continuous=cont, prior=dict(
            query_heads=len(frame), nonconcave_fraction=float((frame.hull_gap > 1e-6).mean()),
            hull_gap_quantiles={q: float(frame.hull_gap.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)},
            hull_gap_rel_width25_p99=float((frame.hull_gap / frame.width_at_25.abs().clip(lower=1e-9)).quantile(0.99)),
            missed_log_gamma_quantiles={q: float(frame.missed_log_gamma.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)}))
        pd.Series(summary).to_json(EXPORT / 'nonconcavity_summary.json', indent=2)
        print(summary)
        run.done(**summary['prior'])
    except BaseException:
        run.done('failed')
        raise


def label_qa(run_dir, expected=10000):
    run_dir = Path(run_dir).resolve()
    run = Run('e13_rosenbaum_oracle', f'label_qa_{run_dir.name}', gpu=False, threads=2, run_dir=str(run_dir))
    try:
        ids = sorted(int(p.stem.split('_')[-1]) for p in (run_dir / 'frontiers').glob('frontier_points_*.csv'))
        n_queries = len(list((run_dir / 'queries').glob('queries_*.csv')))
        assert len(ids) == expected and n_queries == expected, (len(ids), n_queries)
        assert (run_dir / 'diagnostics_postprocess' / 'postprocess_summary.json').exists()
        relabel = json.loads((run_dir / 'relabel_dense' / 'summary.json').read_text())
        assert relabel['dgps'] == expected and max(relabel['upper_ratio_median'], relabel['lower_ratio_median']) <= 1.05, relabel
        rng = np.random.default_rng(20261108)
        sample = sorted(rng.choice(ids, size=min(300, len(ids)), replace=False))
        maxg, ming, allg, rows_per = [], [], [], []
        for i in sample:
            f = pd.read_csv(run_dir / 'frontiers' / f'frontier_points_{i}.csv')
            assert np.isfinite(f[['gamma_star', 'theta_star']].to_numpy()).all(), i
            assert (f.gamma_star >= 1.0 - 1e-9).all() and (f.theta_star.abs() < 100).all(), i
            for (q, head), g in f.groupby(['query_id', 'bound_type']):
                g = g.sort_values('gamma_star', kind='stable')
                d = np.diff(g.theta_star.to_numpy())
                assert (d >= -1e-6).all() if head == 'upper' else (d <= 1e-6).all(), (i, q, head)
                rows_per.append(len(g)); maxg.append(g.gamma_star.max()); ming.append(g.gamma_star.min())
            allg.append(f.gamma_star.to_numpy())
        allg = np.concatenate(allg)
        assert min(rows_per) >= 10
        out = dict(dgps=len(ids), sampled_dgps=len(sample), relabel=relabel,
                   rows_per_query_head=[int(min(rows_per)), int(np.median(rows_per)), int(max(rows_per))],
                   label_gamma_quantiles={q: float(np.quantile(allg, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                   per_query_head_max_gamma_quantiles={q: float(np.quantile(maxg, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                   per_query_head_min_gamma_q95=float(np.quantile(ming, 0.95)),
                   fraction_reaching={str(v): float(np.mean(np.array(maxg) >= v)) for v in (5, 10, 25, 50)})
        (run.path / 'label_qa.json').write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2), flush=True)
        run.done(**{k: v for k, v in out.items() if k in ('dgps', 'sampled_dgps')})
    except BaseException:
        run.done('failed')
        raise

# Relabel dense


DENSE_GRID = 4001

def dense_delta(params, flow, z):
    _, log_r = spline_states(params, flow, z)
    log_r = log_r.double()
    return np.exp((log_r.max(1).values.clamp(min=0.0) - log_r.min(1).values.clamp(max=0.0)).cpu().numpy())

def identity_q0(dataset_id, src, dev, cache={}):
    """Point-identified queried-arm CAPO E_phi[f(x, a, U)] per query (normalized units)."""
    import torch
    from setting_standard.causaldataset import load_outcome_model_bundle, build_outcome_model_from_bundle
    if 'zq' not in cache:
        zq = torch.linspace(-8.0, 8.0, 3201, device=dev, dtype=torch.float64)
        wq = torch.exp(-0.5 * zq ** 2)
        wq[0] *= 0.5; wq[-1] *= 0.5
        cache.update(zq=zq.float(), wq=(wq / wq.sum()))
    zq, wq = cache['zq'], cache['wq']
    meta = read(src / 'metadata' / f'dgp_{dataset_id}.json')
    outcome = build_outcome_model_from_bundle(load_outcome_model_bundle(meta['outcome_model_path']), device=str(dev))
    q = pd.read_csv(src / 'queries' / f'queries_{dataset_id}.csv')
    xcols = [c for c in q.columns if c.startswith('x_')]
    x = torch.tensor(q[xcols].to_numpy(), dtype=torch.float32, device=dev)
    a = torch.tensor(q.a_query.to_numpy(), dtype=torch.float32, device=dev)[:, None]
    m, k = len(q), len(zq)
    with torch.no_grad():
        y = outcome(x[:, None, :].expand(-1, k, -1).reshape(-1, x.shape[1]), a[:, None, :].expand(-1, k, -1).reshape(-1, 1),
                    zq[None].expand(m, -1).reshape(-1, 1)).reshape(m, k).double()
    return q.query_id.to_numpy(), (y @ wq).cpu().numpy()

def relabel_one(dataset_id, src, dest, flow, z, dev, identity_point=False):
    import torch
    state = np.load(src / 'spline_params' / f'spline_params_{dataset_id}.npz')
    raw = pd.read_csv(src / 'frontiers_raw' / f'frontier_points_{dataset_id}.csv')
    qid = state['query_id']
    tables, stats = [], {}
    for head in ('upper', 'lower'):
        params = torch.tensor(state[f'{head}_spline_params'], device=dev)
        g_star = state[f'{head}_gamma_star'].astype(np.float64)
        th_star = state[f'{head}_theta_star']
        with torch.no_grad():
            dense = np.stack([dense_delta(params[li], flow, z) for li in range(params.shape[0])])
        attained = np.maximum(g_star, dense)
        lam = np.repeat(np.arange(params.shape[0]), len(qid))
        tables.append(pd.DataFrame(dict(query_id=np.tile(qid, params.shape[0]), bound_type=head,
                                        gamma_star=attained.ravel(), theta_star=th_star.ravel(),
                                        _sampled=g_star.ravel().astype(np.float32), _lambda=lam)))
        ratio = attained / g_star
        stats[head] = dict(ratio_median=float(np.median(ratio)), ratio_p99=float(np.quantile(ratio, 0.99)),
                           ratio_max=float(ratio.max()), dense_below_sampled=float((dense < g_star).mean()))
    new = pd.concat(tables, ignore_index=True)
    # Exact correspondence with the raw Step-3 table: same multiset of (query, head, sampled gamma, theta).
    key_new = new[['query_id', 'bound_type', '_sampled', 'theta_star']].sort_values(['bound_type', 'query_id', '_sampled', 'theta_star']).to_numpy()
    key_raw = raw.assign(_sampled=raw.gamma_star.astype(np.float32))[['query_id', 'bound_type', '_sampled', 'theta_star']].sort_values(
        ['bound_type', 'query_id', '_sampled', 'theta_star']).to_numpy()
    assert key_new.shape == key_raw.shape and (key_new[:, :2] == key_raw[:, :2]).all(), dataset_id
    assert np.allclose(key_new[:, 2:].astype(float), key_raw[:, 2:].astype(float), rtol=0, atol=1e-6), dataset_id
    assert (new.gamma_star >= new._sampled - 1e-6).all() and np.isfinite(new.gamma_star).all()
    out = new.sort_values(['bound_type', 'query_id', '_lambda'], kind='stable')[['query_id', 'bound_type', 'gamma_star', 'theta_star']]
    if identity_point:
        qids, q0 = identity_q0(dataset_id, src, dev)
        assert set(qids) == set(qid) and np.isfinite(q0).all(), dataset_id
        ident = pd.concat([pd.DataFrame(dict(query_id=qids, bound_type=h, gamma_star=1.0, theta_star=q0)) for h in ('upper', 'lower')])
        out = pd.concat([ident, out], ignore_index=True).sort_values(['bound_type', 'query_id'], kind='stable')
        # Sanity: where the largest-lambda label already sits at Gamma <= 1.02, its MC theta* should be ~q0.
        lookup = pd.Series(q0, index=qids)
        near = new[new._lambda.eq(0) & (new.gamma_star <= 1.02)]
        diff = (near.theta_star - lookup.reindex(near.query_id).to_numpy()).abs()
        stats['identity'] = dict(q0_abs_max=float(np.abs(q0).max()), near_rows=int(len(near)),
                                 near_abs_diff_median=float(diff.median()) if len(near) else None)
    target = dest / 'frontiers_raw' / f'frontier_points_{dataset_id}.csv'
    tmp = target.with_suffix('.csv.tmp')
    out.to_csv(tmp, index=False)
    os.replace(tmp, target)                     # new inode: the source run's hardlinked file is untouched
    return stats

def relabel(source, dest, ids, identity_point=False, device="cuda:0"):
    import torch
    from setting_standard.frontier_optimization import FlowConfig
    source, dest = Path(source).resolve(), Path(dest).resolve()
    assert (source / 'spline_params').is_dir() and not dest.exists()
    run = Run('e13_rosenbaum_oracle', f'relabel_dense_{dest.name}', gpu=device.startswith('cuda'), threads=4,
              source=str(source), dest=str(dest), grid=DENSE_GRID, rule='gamma = max(sampled, dense-grid sup r / inf r)', identity_point=identity_point)
    try:
        subprocess.run([sys.executable, str(CODE / 'src/setting_standard/fork_run_dir.py'),
                        '--source_run_dir', str(source), '--dest_run_dir', str(dest), '--step', '4']
                       + (['--dataset_ids', *map(str, ids)] if ids else []), check=True, cwd=CODE)
        flow = FlowConfig(**read(source / 'metadata_frontier/frontier_run_config.json')['config']['flow'])
        dev = torch.device(device)
        z = torch.linspace(-flow.tail_bound, flow.tail_bound, DENSE_GRID, device=dev)
        files = sorted((dest / 'frontiers_raw').glob('frontier_points_*.csv'), key=lambda p: int(p.stem.split('_')[-1]))
        (dest / 'relabel_dense').mkdir()
        rows = []
        for k, path in enumerate(files):
            dataset_id = int(path.stem.split('_')[-1])
            stats = relabel_one(dataset_id, source, dest, flow, z, dev, identity_point)
            rows.append(dict(dataset_id=dataset_id, **{f'{h}_{k2}': v for h, s in stats.items() for k2, v in s.items()}))
            if k % 500 == 0:
                print(f'relabelled {k + 1}/{len(files)}', flush=True)
        summary = pd.DataFrame(rows)
        summary.to_csv(dest / 'relabel_dense' / 'per_dgp.csv', index=False)
        agg = dict(dgps=len(summary), upper_ratio_median=float(summary.upper_ratio_median.median()),
                   lower_ratio_median=float(summary.lower_ratio_median.median()),
                   ratio_p99_max=float(summary[['upper_ratio_p99', 'lower_ratio_p99']].max().max()),
                   ratio_max=float(summary[['upper_ratio_max', 'lower_ratio_max']].max().max()))
        (dest / 'relabel_dense' / 'summary.json').write_text(json.dumps(dict(agg, run=str(run.path)), indent=2))
        print(agg, flush=True)
        run.done(**agg)
    except BaseException:
        run.done('failed')
        raise


def report(name):
    """Summarize completed optimizer labels against their independent references."""
    source = RUNS / PACKAGE / f'analyze_{name}'
    assert read(source / 'manifest.json')['status'] == 'completed'
    files = sorted((source / 'points').glob('*.parquet'))
    assert files, source
    points = pd.concat([pd.read_parquet(p, columns=['gamma_true', 'label_error']) for p in files], ignore_index=True)
    rows = []
    for band, selected in [('all', points), ('Gamma<=1.1', points[points.gamma_true <= 1.1]),
                           ('25<=Gamma<=50', points[points.gamma_true.between(25, 50)])]:
        assert len(selected), band
        error = selected.label_error.to_numpy()
        rows.append(dict(band=band, points=len(error), rmse=float(np.sqrt(np.mean(error**2))),
                         median_signed_error=float(np.median(error))))
    EXPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXPORT / 'optimizer_audit.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sweep', 'analyze', 'checks', 'nonconcavity', 'relabel', 'label-qa', 'report'])
    parser.add_argument('--budget', choices=['smoke', 'pilot', 'audit'], default='audit')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--dest')
    parser.add_argument('--ids', type=int, nargs='*')
    parser.add_argument('--identity-point', '--identity_point', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected', type=int, default=10000)
    args = parser.parse_args()
    if args.stage in {'sweep', 'analyze'}: {'sweep': sweep, 'analyze': analyze}[args.stage](args.budget)
    elif args.stage == 'report': report(args.budget)
    elif args.stage == 'checks': checks(args.smoke)
    elif args.stage == 'nonconcavity': nonconcavity()
    elif args.stage == 'label-qa':
        assert args.source, 'label-qa requires --source'
        label_qa(args.source, args.expected)
    else:
        assert args.source and args.dest, 'relabel requires --source and --dest'
        relabel(args.source, args.dest, args.ids, args.identity_point, args.device)


if __name__ == '__main__':
    main()
