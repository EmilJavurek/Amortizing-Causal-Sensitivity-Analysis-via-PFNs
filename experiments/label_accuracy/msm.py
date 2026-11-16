"""MSM optimizer labels versus analytical bounds: sweep, error decomposition and reports."""


from __future__ import annotations

from experiments.shared.paths import REPORTS
import argparse
import base64
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time
import torch
import yaml
from experiments.shared.paths import CODE, RUNS
from experiments.shared.runtime import Run, cluster_bootstrap, read
from experiments.shared.inputs import cohort_ids, input_path, cohort_artifacts


# Run


EXPORT = REPORTS / 'msm_labels'
PROD_YAML = CODE / 'experiments/configs/paper/optimization_kl/03_frontier.yaml'
IDS = cohort_ids('msm')
SOURCE = input_path('normalized_root')

# Production solver block from the paper KL frontier config; only the model changes.
PRODUCTION_KEYS = ['device', 'save_diagnostics', 'num_query_points', 'lambda_min', 'lambda_max',
                   'lambda_n', 'lambda_spacing', 'num_mc_samples_train', 'num_mc_samples_eval',
                   'max_steps', 'flow_num_bins', 'flow_tail_bound', 'flow_min_bin_width',
                   'flow_min_bin_height', 'flow_min_derivative', 'loss_reduction', 'warm_start_lambdas',
                   'early_stop', 'use_bucket_forward', 'lr_lambda_schedule', 'lr_lambda_ref',
                   'lr_lambda_min_mult', 'max_steps_lambda_schedule', 'max_steps_lambda_max_mult',
                   'early_stop_min_steps', 'early_stop_check_every', 'early_stop_patience',
                   'early_stop_abs_tol', 'early_stop_rel_tol']

def production_config():
    cfg = yaml.safe_load(PROD_YAML.read_text())
    assert cfg['sensitivity_model'] == 'kl'
    return {key: cfg[key] for key in PRODUCTION_KEYS}

# Budgets. `prod` is the paper KL production budget with sensitivity_model switched to msm.
# `prod_msmgrid` replaces the lambda grid to cover the MSM sensitivity range.
# `high` raises Monte Carlo samples and steps and disables early stopping on fewer DGPs.
MSM_GRID = dict(lambda_min=0.002, lambda_max=2.0, lambda_n=50, lambda_spacing='log')

def budget(name):
    cfg = production_config()
    cfg.update(sensitivity_model='msm', save_spline_params=True, dgp_batch_size=16)
    if name == 'smoke':
        cfg.update(num_query_points=4, lambda_n=6, max_steps=60, early_stop=False, dgp_batch_size=2)
        return cfg, IDS[:2]
    if name == 'prod':
        return cfg, IDS[:16]
    if name == 'prod_msmgrid':
        cfg.update(MSM_GRID)
        return cfg, IDS[:16]
    if name == 'high_msmgrid':
        cfg.update(MSM_GRID)
        cfg.update(num_mc_samples_train=512, num_mc_samples_eval=8192, max_steps=1000, early_stop=False,
                   dgp_batch_size=4)
        return cfg, IDS[:4]
    raise ValueError(name)

def sweep(name):
    cfg, ids = budget(name)
    artifacts = cohort_artifacts('msm')
    assert set(ids) <= set(artifacts)
    run = Run('e03', f'sweep_{name}', smoke=name == 'smoke', gpu=True, threads=2, budget=name,
              dataset_ids=ids, frontier=cfg)
    run_dir = run.path / 'frontier'
    full = dict(source_run_dir=str(SOURCE), run_dir=str(run_dir), dataset_ids=list(ids), **cfg)
    config_bytes = yaml.safe_dump(full, sort_keys=False).encode()
    (run.path / 'input.yaml').write_bytes(config_bytes)

    from experiments.scripts._run import config_to_args
    script = CODE / 'src/setting_standard/frontier_optimization.py'
    cmd = [sys.executable, str(script)] + config_to_args(yaml.safe_load(config_bytes))
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

# ----------------------------------------------------------------------------------------------
# Analysis: pure computation helpers
# ----------------------------------------------------------------------------------------------

def msm_bound_at(sorted_desc_cumsum, total, n, pi, gamma, upper):
    """Fractional-threshold empirical MSM CAPO bound at per-row gamma (float64 torch).

    Mirrors run_shared_msm.msm_reference_from_samples: weight (1-Gamma)pi+Gamma on the top
    n/(Gamma+1) outcome draws (bottom draws for the lower bound) and (1-1/Gamma)pi+1/Gamma on the rest.
    `sorted_desc_cumsum[r]` is the prefix sum of the draws ordered towards the bound direction.
    """
    tail = n / (gamma + 1.0)
    full = torch.floor(tail).long().clamp(max=n - 1)
    frac = tail - full
    inv_high = (1.0 - gamma) * pi + gamma
    inv_low = (1.0 - 1.0 / gamma) * pi + 1.0 / gamma
    raw = tail * inv_high + (n - tail) * inv_low
    high, low = inv_high / raw, inv_low / raw
    prefix = torch.where(full > 0, sorted_desc_cumsum.gather(1, (full - 1).clamp(min=0)[:, None])[:, 0],
                         torch.zeros_like(total))
    nxt = sorted_desc_cumsum.gather(1, full[:, None])[:, 0] - prefix
    return total * low + (prefix + frac * nxt) * (high - low)

def spline_states(params, flow_cfg, z):
    """Push a dense base grid z through every saved spline state: returns u=T(z), log r(u)."""
    from setting_standard.frontier_optimization import batched_spline_forward, standard_normal_logprob
    P = params.shape[0]
    base = z[None, None, :].expand(1, P, -1)
    u, log_p_eta = batched_spline_forward(base, params[None], flow_cfg)
    log_r = log_p_eta - standard_normal_logprob(u)
    return u[0], log_r[0]

def analyze(name, max_queries=None):
    sweep_dir = RUNS / 'e03' / f'sweep_{name}'
    manifest = read(sweep_dir / 'manifest.json')
    assert manifest['status'] == 'completed', manifest['status']
    run = Run('e03', f'analyze_{name}', smoke=manifest['smoke'], gpu=True, threads=2, sweep=str(sweep_dir),
              reference_quantile_grid=65536, base_grid=4001, quadrature_grid=3201)
    from setting_standard.frontier_optimization import FlowConfig
    from setting_standard.causaldataset import (load_outcome_model_bundle, build_outcome_model_from_bundle,
                                                load_propensity_model_bundle, build_propensity_model_from_bundle)
    frontier = sweep_dir / 'frontier'
    artifacts = cohort_artifacts('msm')
    fcfg = read(frontier / 'metadata_frontier/frontier_run_config.json')['config']
    flow = FlowConfig(**fcfg['flow'])
    dev = torch.device('cuda:0')
    z = torch.linspace(-flow.tail_bound, flow.tail_bound, 4001, device=dev)
    # Quadrature grid for expectations under N(0,1): trapezoid on [-8, 8] (mass outside < 1e-15).
    zq = torch.linspace(-8.0, 8.0, 3201, device=dev, dtype=torch.float64)
    wq = torch.exp(-0.5 * zq ** 2) / np.sqrt(2 * np.pi) * (zq[1] - zq[0])
    wq[0] *= 0.5; wq[-1] *= 0.5
    wq = wq / wq.sum()
    rows = []
    fresh_out = run.path / 'points'
    fresh_out.mkdir()
    try:
        for dataset_id in manifest['config']['dataset_ids']:
            artifact = artifacts[dataset_id]
            outcome = build_outcome_model_from_bundle(load_outcome_model_bundle(artifact['outcome_model_path']), device='cuda:0')
            propensity = build_propensity_model_from_bundle(load_propensity_model_bundle(artifact['propensity_model_path']), device='cuda:0')
            queries = pd.read_csv(frontier / 'queries' / f'queries_{dataset_id}.csv')
            points = pd.read_csv(frontier / 'frontiers_raw' / f'frontier_points_{dataset_id}.csv')
            state = np.load(frontier / 'spline_params' / f'spline_params_{dataset_id}.npz')
            xcols = [c for c in queries.columns if c.startswith('x_')]
            if max_queries is not None:
                queries = queries.iloc[:max_queries]
            x = torch.tensor(queries[xcols].to_numpy(), dtype=torch.float32, device=dev)
            a = torch.tensor(queries.a_query.to_numpy(), dtype=torch.float32, device=dev)[:, None]
            with torch.no_grad():
                p1 = propensity(x).reshape(-1).double()
            pi = torch.where(a[:, 0] == 1, p1, 1 - p1)
            m = len(queries)
            # Reference latent: deterministic N(0,1) midpoint-quantile grid (n=65,536). The empirical
            # law of f(x,a,U) on this grid approximates Y|x,a with O(1/n) quadrature error instead of
            # O(n^-1/2) Monte Carlo error; the MSM formula is applied to it exactly as to MC draws.
            from scipy.special import ndtri
            grid_u = torch.tensor(ndtri((np.arange(65536) + 0.5) / 65536), dtype=torch.float32, device=dev)
            sorted_up, sorted_lo, totals = [], [], []
            with torch.no_grad():
                for s in range(0, m, 64):
                    xs, as_ = x[s:s + 64], a[s:s + 64]
                    lat = grid_u[None].expand(len(xs), -1)
                    y = outcome(xs[:, None, :].expand(-1, 65536, -1).reshape(-1, xs.shape[1]),
                                as_[:, None, :].expand(-1, 65536, -1).reshape(-1, 1),
                                lat.reshape(-1, 1)).reshape(len(xs), 65536).double()
                    asc = torch.sort(y, dim=1).values
                    sorted_lo.append(torch.cumsum(asc, 1)); sorted_up.append(torch.cumsum(asc.flip(1), 1))
                    totals.append(asc.sum(1))
            cum_up, cum_lo, total = torch.cat(sorted_up), torch.cat(sorted_lo), torch.cat(totals)
            out = []
            for head in ('upper', 'lower'):
                params = torch.tensor(state[f'{head}_spline_params'], device=dev)        # [L, m_all, d]
                theta_star = torch.tensor(state[f'{head}_theta_star'], device=dev).double()
                gamma_star = torch.tensor(state[f'{head}_gamma_star'], device=dev).double()
                lambdas = state[f'{head}_lambda_value']
                L = params.shape[0]
                params, theta_star, gamma_star = params[:, :m], theta_star[:, :m], gamma_star[:, :m]
                cum = cum_up if head == 'upper' else cum_lo
                for li in range(L):
                    with torch.no_grad():
                        _, log_r = spline_states(params[li], flow, z)
                        log_r = log_r.double()
                        # Outside [-B, B] the flow is the identity, so r = 1 there.
                        gamma_true = torch.maximum(torch.exp(log_r.max(1).values).clamp(min=1.0),
                                                   torch.exp(-log_r.min(1).values).clamp(min=1.0))
                        u_q, _ = spline_states(params[li], flow, zq.float())
                        y_nu = torch.empty(m, len(zq), device=dev, dtype=torch.float64)
                        y_phi = torch.empty_like(y_nu)
                        for s in range(0, m, 128):
                            xs, as_ = x[s:s + 128], a[s:s + 128]
                            k = len(xs)
                            xe = xs[:, None, :].expand(-1, len(zq), -1).reshape(-1, xs.shape[1])
                            ae = as_[:, None, :].expand(-1, len(zq), -1).reshape(-1, 1)
                            y_nu[s:s + k] = outcome(xe, ae, u_q[s:s + k].reshape(-1, 1)).reshape(k, -1).double()
                            y_phi[s:s + k] = outcome(xe, ae, zq.float()[None].expand(k, -1).reshape(-1, 1)).reshape(k, -1).double()
                        theta_nu = pi * (y_phi @ wq) + (1 - pi) * (y_nu @ wq)
                        up = head == 'upper'
                        ref_true = msm_bound_at(cum, total, 65536, pi, gamma_true, up)
                        ref_star = msm_bound_at(cum, total, 65536, pi, gamma_star[li].clamp(min=1.0), up)
                    sign = 1.0 if up else -1.0
                    out.append(pd.DataFrame(dict(
                        dataset_id=dataset_id, query_id=queries.query_id.to_numpy(), a_query=queries.a_query.to_numpy(),
                        pi_query=pi.cpu().numpy(), head=head, lambda_index=li, lambda_value=lambdas[li],
                        gamma_star=gamma_star[li].cpu().numpy(), gamma_true=gamma_true.cpu().numpy(),
                        theta_star=theta_star[li].cpu().numpy(), theta_nu=theta_nu.cpu().numpy(),
                        theta_ref_star=ref_star.cpu().numpy(), theta_ref_true=ref_true.cpu().numpy(),
                        q0=(total / 65536).cpu().numpy(), sign=sign)))
            frame = pd.concat(out, ignore_index=True)
            s = frame.sign
            frame['label_error'] = s * (frame.theta_star - frame.theta_ref_star)
            frame['mc_error'] = s * (frame.theta_star - frame.theta_nu)
            frame['objective_error'] = s * (frame.theta_nu - frame.theta_ref_true)
            frame['sensitivity_shift'] = s * (frame.theta_ref_true - frame.theta_ref_star)
            frame.to_parquet(fresh_out / f'dgp_{dataset_id}.parquet', index=False)
            print(f'analyzed {dataset_id}: points={len(frame)} elapsed={time.time()-run.started:.0f}s', flush=True)
        run.done()
    except BaseException:
        run.done('failed')
        raise


# Report


BUDGETS = ['prod', 'prod_msmgrid', 'high_msmgrid']
BINS = [1.0, 1.1, 1.5, 2.0, 3.0, 5.0, np.inf]

def load(budget):
    path = RUNS / 'e03' / f'analyze_{budget}'
    if not (path / 'manifest.json').exists() or read(path / 'manifest.json')['status'] != 'completed':
        return None
    f = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(path / 'points/*.parquet')))], ignore_index=True)
    f['budget'] = budget
    f['gamma_bin'] = pd.cut(f.gamma_star, BINS, right=False).astype(str)
    f['gamma_ratio'] = f.gamma_true / f.gamma_star
    return f

def q(s, p):
    return float(np.quantile(s, p))

def describe(g):
    e = g.label_error
    return pd.Series(dict(
        points=len(g), dgps=g.dataset_id.nunique(),
        label_rmse=float(np.sqrt((e ** 2).mean())), label_bias=float(e.mean()),
        label_abs_median=q(e.abs(), .5), label_abs_p95=q(e.abs(), .95), label_abs_p99=q(e.abs(), .99),
        label_abs_max=float(e.abs().max()), label_outward_rate=float((e > 0).mean()),
        mc_error_sd=float(g.mc_error.std()), objective_error_mean=float(g.objective_error.mean()),
        objective_error_p01=q(g.objective_error, .01), objective_error_max=float(g.objective_error.max()),
        objective_error_rate_below_m001=float((g.objective_error < -0.01).mean()),
        sensitivity_shift_mean=float(g.sensitivity_shift.mean()), sensitivity_shift_p99=q(g.sensitivity_shift, .99),
        gamma_true_over_star_median=q(g.gamma_ratio, .5), gamma_true_over_star_p99=q(g.gamma_ratio, .99),
        gamma_true_above_star_1pct_rate=float((g.gamma_ratio > 1.01).mean()),
        gamma_true_below_star_rate=float((g.gamma_ratio < 1 - 1e-6).mean())))

def report():
    EXPORT.mkdir(parents=True, exist_ok=True)
    frames = {b: load(b) for b in BUDGETS}
    frames = {b: f for b, f in frames.items() if f is not None}
    summary, by_bin, per_dgp, coverage, boot = [], [], [], [], []
    for b, f in frames.items():
        s = f.groupby('head').apply(describe).reset_index(); s.insert(0, 'budget', b); summary.append(s)
        s = f.groupby(['head', 'gamma_bin']).apply(describe).reset_index(); s.insert(0, 'budget', b); by_bin.append(s)
        s = f.groupby(['dataset_id', 'head']).apply(describe).reset_index(); s.insert(0, 'budget', b); per_dgp.append(s)
        cov = f.groupby('head').gamma_star.describe(percentiles=[.1, .5, .9, .99]).reset_index()
        cov.insert(0, 'budget', b); coverage.append(cov)
        for head, g in f.groupby('head'):
            ci = cluster_bootstrap(g[['dataset_id', 'label_error']], 'dataset_id',
                                   lambda d: dict(label_rmse=float(np.sqrt((d.label_error ** 2).mean())),
                                                  label_bias=float(d.label_error.mean())), replicates=1000)
            boot.append(dict(budget=b, head=head, **ci))
    pd.concat(summary).to_csv(EXPORT / 'summary.csv', index=False)
    pd.concat(by_bin).to_csv(EXPORT / 'by_gamma_bin.csv', index=False)
    pd.concat(per_dgp).to_csv(EXPORT / 'per_dgp.csv', index=False)
    pd.concat(coverage).to_csv(EXPORT / 'gamma_star_coverage.csv', index=False)
    pd.DataFrame(boot).to_csv(EXPORT / 'label_rmse_bootstrap.csv', index=False)
    runtimes = []
    for b in ['smoke'] + BUDGETS:
        for kind in ('sweep', 'analyze'):
            p = RUNS / 'e03' / f'{kind}_{b}' / 'manifest.json'
            if p.exists():
                m = read(p); runtimes.append(dict(run=f'{kind}_{b}', status=m['status'], wall_seconds=m.get('wall_seconds')))
    pd.DataFrame(runtimes).to_csv(EXPORT / 'runtime.csv', index=False)

    # Plot 1: binned label-error quantiles against reported gamma* per budget and head.
    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 3.6), sharey=True, squeeze=False)
    for ax, (b, f) in zip(axes[0], frames.items()):
        edges = np.unique(np.quantile(f.gamma_star, np.linspace(0, 1, 41)))
        for head, color in (('upper', '#1f77b4'), ('lower', '#d62728')):
            g = f[f['head'] == head]
            idx = np.clip(np.digitize(g.gamma_star, edges) - 1, 0, len(edges) - 2)
            stats = g.groupby(idx).label_error.quantile([.05, .5, .95]).unstack()
            centers = np.array([g.gamma_star[idx == k].median() for k in stats.index])
            ax.plot(centers, stats[0.5], color=color, label=f'{head} median')
            ax.fill_between(centers, stats[0.05], stats[0.95], color=color, alpha=.2, label=f'{head} 5-95%')
        ax.axhline(0, color='k', lw=.8); ax.set_xscale('log'); ax.set_title(b); ax.set_xlabel('reported Gamma*')
    axes[0][0].set_ylabel('label error vs analytical (outward +)'); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(EXPORT / 'label_error_vs_gamma.png', dpi=130); plt.close(fig)

    # Plot 2: example frontiers (reported labels vs analytical curve) for the MSM-grid production run.
    key = 'prod_msmgrid' if 'prod_msmgrid' in frames else next(iter(frames))
    f = frames[key]
    examples = f[(f.dataset_id == f.dataset_id.min())].query_id.drop_duplicates().iloc[[0, 700, 1400, 2047 % f.query_id.nunique()]]
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 3.3), squeeze=False)
    for ax, qid in zip(axes[0], examples):
        g = f[(f.dataset_id == f.dataset_id.min()) & (f.query_id == qid)].sort_values('gamma_star')
        for head, color in (('upper', '#1f77b4'), ('lower', '#d62728')):
            h = g[g['head'] == head]
            ax.plot(h.gamma_star, h.theta_ref_star, color=color, lw=1, label=f'analytical {head}')
            ax.scatter(h.gamma_star, h.theta_star, s=8, color=color, alpha=.7, label=f'sweep label {head}')
        ax.set_xlabel('Gamma'); ax.set_title(f'DGP {f.dataset_id.min()} query {qid} (pi={g.pi_query.iloc[0]:.2f})', fontsize=8)
    axes[0][0].set_ylabel('CAPO bound (std. units)'); axes[0][0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(EXPORT / 'example_frontiers.png', dpi=130); plt.close(fig)
    print(pd.concat(summary)[['budget', 'head', 'points', 'label_rmse', 'label_abs_p95', 'label_abs_max',
                              'objective_error_max', 'gamma_true_above_star_1pct_rate']].to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sweep', 'analyze', 'report'])
    parser.add_argument('--budget', default='prod')
    parser.add_argument('--max-queries', '--max_queries', dest='max_queries', type=int)
    args = parser.parse_args()
    if args.stage == 'sweep': sweep(args.budget)
    elif args.stage == 'analyze': analyze(args.budget, args.max_queries)
    else: report()


if __name__ == '__main__':
    main()
