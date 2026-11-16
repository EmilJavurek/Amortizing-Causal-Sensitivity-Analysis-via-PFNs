"""Generate high-budget KL optimization references on the observed evaluation cohort.

These references use the training frontier solver with larger budgets and fresh diagnostics;
they are numerical comparison curves, not an independent sharp-bound certificate. Each head
retains its own attained sensitivity levels. Scalers and source hashes accompany the table.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from setting_standard.causaldataset import build_outcome_model_from_bundle, load_outcome_model_bundle
from setting_standard.frontier_optimization import FlowConfig, OutcomeForwardPlan, SolverConfig, optimize_lambda_batch
from experiments.comparisons.neuralcsa import Stage2Config, _independent_gamma_diagnostics
from experiments.shared.evaluation import normalized_dataset, propensity_stratified_row_indices
from experiments.shared.inputs import cohort_artifacts, cohort_ids, evaluation_config, input_path, normalized_root
from experiments.shared.runtime import provenance, read, set_threads, sha, write

LAMBDA_GRID = np.geomspace(2.0, 0.08, 10)


def configs(smoke=False):
    flow = FlowConfig(num_bins=8, tail_bound=6.0, min_bin_width=1e-3,
                      min_bin_height=1e-3, min_derivative=1e-3)
    solver = SolverConfig(num_mc_samples_train=256, num_mc_samples_eval=4096, max_steps=1000,
                          learning_rate=1e-3, lr_lambda_schedule='sqrt', lr_lambda_ref=0.25,
                          lr_lambda_min_mult=0.4, max_steps_lambda_schedule='inverse_sqrt_lr',
                          max_steps_lambda_max_mult=2.0, loss_reduction='per_dgp_sum', early_stop=False)
    if smoke:
        solver = SolverConfig(num_mc_samples_train=64, num_mc_samples_eval=512, max_steps=40,
                              learning_rate=1e-3, loss_reduction='per_dgp_sum')
    return flow, solver


def load_shard(
    artifacts: dict[int, dict],
    dataset_ids: list[int],
    rows_per_dgp: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list,
    list[list[dict]],
]:
    x_batches: list[torch.Tensor] = []
    a_batches: list[torch.Tensor] = []
    pi_batches: list[torch.Tensor] = []
    models: list = []
    metadata_batches: list[list[dict]] = []
    for dataset_id in dataset_ids:
        frame, x_columns = normalized_dataset(artifacts[dataset_id])
        row_indices = propensity_stratified_row_indices(
            frame,
            rows_per_dgp,
            quantiles=np.linspace(0.25, 0.75, rows_per_dgp),
        )
        local_x: list[np.ndarray] = []
        local_a: list[float] = []
        local_pi: list[float] = []
        local_metadata: list[dict] = []
        for source_row_index in row_indices:
            row = frame.iloc[source_row_index]
            p_a1 = float(row["propensity"])
            for arm in (0, 1):
                local_x.append(row[x_columns].to_numpy(dtype=np.float32))
                local_a.append(float(arm))
                pi_query = p_a1 if arm == 1 else 1.0 - p_a1
                local_pi.append(pi_query)
                local_metadata.append(
                    {
                        "dataset_id": dataset_id,
                        "source_row_index": int(source_row_index),
                        "a_query": arm,
                        "p_a1": p_a1,
                        "pi_query": pi_query,
                        "counterfactual_weight": 1.0 - pi_query,
                        **{column: float(row[column]) for column in x_columns},
                    }
                )
        x_batches.append(torch.tensor(np.stack(local_x), device=device))
        a_batches.append(
            torch.tensor(local_a, dtype=torch.float32, device=device)[:, None]
        )
        pi_batches.append(torch.tensor(local_pi, dtype=torch.float32, device=device))
        bundle = load_outcome_model_bundle(
            artifacts[dataset_id]["outcome_model_path"], map_location="cpu"
        )
        models.append(
            build_outcome_model_from_bundle(bundle, device=str(device), freeze=True)
        )
        metadata_batches.append(local_metadata)
    return (
        torch.stack(x_batches),
        torch.stack(a_batches),
        torch.stack(pi_batches),
        models,
        metadata_batches,
    )


def run_scalarized_sweep(
    *,
    label: str,
    restart: int,
    dataset_ids: list[int],
    x_queries: torch.Tensor,
    a_queries: torch.Tensor,
    pi_queries: torch.Tensor,
    query_metadata: list[list[dict]],
    outcome_plan: OutcomeForwardPlan,
    flow_config: FlowConfig,
    solver_config: SolverConfig,
    lambda_grid: np.ndarray,
    diagnostic_mc: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records: list[dict] = []
    spline_payload: dict[str, np.ndarray] = {}
    for bound_type in ("upper", "lower"):
        warm_start = None
        for lambda_index, lambda_value in enumerate(lambda_grid):
            result = optimize_lambda_batch(
                outcome_plan=outcome_plan,
                x_queries=x_queries,
                a_queries=a_queries,
                pi_queries=pi_queries,
                lambda_value=float(lambda_value),
                bound_type=bound_type,
                sensitivity_model="kl",
                spline_params_init=warm_start,
                flow_config=flow_config,
                solver_config=solver_config,
                device=str(device),
                lambda_index=lambda_index,
                dataset_ids=dataset_ids,
                save_diagnostics=False,
            )
            warm_start = result.spline_params_final
            flattened = torch.tensor(
                result.spline_params_final.reshape(
                    -1, result.spline_params_final.shape[-1]
                ),
                dtype=torch.float32,
                device=device,
            )
            diagnostic_config = Stage2Config(
                num_bins=flow_config.num_bins,
                tail_bound=flow_config.tail_bound,
                min_bin_width=flow_config.min_bin_width,
                min_bin_height=flow_config.min_bin_height,
                min_derivative=flow_config.min_derivative,
                eval_mc_samples=diagnostic_mc,
                seed=seed + 100_000 + lambda_index,
            )
            with torch.no_grad():
                robust_gamma, forward_kl, reverse_kl = (
                    _independent_gamma_diagnostics(
                        flattened,
                        "kl",
                        diagnostic_config,
                        seed + 100_000 + lambda_index,
                        device,
                    )
                )
            robust_gamma_np = robust_gamma.cpu().numpy().reshape(
                result.gamma_star.shape
            )
            forward_np = forward_kl.cpu().numpy().reshape(result.gamma_star.shape)
            reverse_np = reverse_kl.cpu().numpy().reshape(result.gamma_star.shape)
            spline_payload[f"{bound_type}_lambda_{lambda_index}"] = (
                result.spline_params_final
            )
            sign = 1.0 if bound_type == "upper" else -1.0
            for dgp_index, dataset_id in enumerate(dataset_ids):
                for query_index, metadata in enumerate(query_metadata[dgp_index]):
                    gamma_value = float(robust_gamma_np[dgp_index, query_index])
                    theta_value = float(result.theta_star[dgp_index, query_index])
                    records.append(
                        {
                            **metadata,
                            "reference_label": label,
                            "restart": restart,
                            "bound_type": bound_type,
                            "lambda_index": lambda_index,
                            "lambda_value": float(lambda_value),
                            "theta_reference": theta_value,
                            "gamma_importance": float(
                                result.gamma_star[dgp_index, query_index]
                            ),
                            "gamma_reference": gamma_value,
                            "kl_nu_phi": float(forward_np[dgp_index, query_index]),
                            "kl_phi_nu_direct": float(
                                reverse_np[dgp_index, query_index]
                            ),
                            "scalarized_objective": (
                                sign * theta_value - float(lambda_value) * gamma_value
                            ),
                            "solver_steps": result.num_steps,
                            "solver_runtime_seconds": result.runtime_seconds,
                        }
                    )
            print(
                f"{label} restart={restart} bound={bound_type} "
                f"lambda={lambda_index + 1}/{len(lambda_grid)}",
                flush=True,
            )
    np.savez_compressed(
        output_dir / f"spline_params_{label}_restart_{restart}.npz",
        **spline_payload,
    )
    return pd.DataFrame(records)

def select_best_restart(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset_id",
        "source_row_index",
        "a_query",
        "bound_type",
        "lambda_index",
    ]
    positions = frame.groupby(keys)["scalarized_objective"].idxmax()
    selected = frame.loc[positions].sort_values(keys).reset_index(drop=True)
    assert len(selected) * frame["restart"].nunique() == len(frame)
    return selected


def build(smoke=False, device='cuda:0', output_dir=None):
    ids = cohort_ids('kl')[:2] if smoke else cohort_ids('kl')
    artifacts = cohort_artifacts('kl', ids)
    output = Path(output_dir) if output_dir else input_path('kl_reference_root').with_name(
        input_path('kl_reference_root').name + ('_smoke' if smoke else ''))
    output = output.resolve()
    assert not output.exists(), f'References require a fresh output directory: {output}'
    output.mkdir(parents=True)
    (output / 'scalers').mkdir()
    set_threads(4)
    flow, solver = configs(smoke)
    dev = torch.device(device)
    shard_size, restarts, rows_per_dgp = (1, 1, 2) if smoke else (8, 2, 8)
    diagnostic_mc = 512 if smoke else 8192
    grid = LAMBDA_GRID
    manifest = dict(status='running', smoke=smoke, base_seed=evaluation_config()['base_seed'],
                    y_fit_column='outcome', source_root=str(normalized_root()), dataset_ids=ids,
                    reference_kind='high_budget_training_solver', git=provenance(),
                    flow=asdict(flow), solver=asdict(solver), lambda_grid=grid.tolist(),
                    rows_per_dgp=rows_per_dgp, restarts=restarts, diagnostic_mc=diagnostic_mc,
                    seed=20261107, inputs={})
    write(output / 'manifest.json', manifest)
    try:
        selected = []
        for offset in range(0, len(ids), shard_size):
            shard = ids[offset:offset + shard_size]
            x, a, pi, models, metadata = load_shard(artifacts, shard, rows_per_dgp, dev)
            plan = OutcomeForwardPlan(use_bucket_forward=False, depth_buckets=None, sequential_f_bnns=models)
            shard_dir = output / f'shard_{offset // shard_size:03d}'
            shard_dir.mkdir()
            frames = []
            for restart in range(restarts):
                frames.append(run_scalarized_sweep(
                    label='reference', restart=restart, dataset_ids=shard,
                    x_queries=x, a_queries=a, pi_queries=pi, query_metadata=metadata,
                    outcome_plan=plan, flow_config=flow, solver_config=solver,
                    lambda_grid=grid, diagnostic_mc=diagnostic_mc,
                    seed=20261107 + 1000 * restart + offset, device=dev, output_dir=shard_dir))
            selected.append(select_best_restart(pd.concat(frames, ignore_index=True)))
            for i in shard:
                scaler = normalized_root() / 'scalers' / f'scaler_{i}.json'
                assert read(scaler)['y_fit_column'] == 'outcome'
                shutil.copy2(scaler, output / 'scalers' / scaler.name)
                sources = {key: Path(artifacts[i][key]) for key in
                           ('dataset_path', 'outcome_model_path', 'propensity_model_path', 'metadata_path')}
                sources['scaler_path'] = scaler
                manifest['inputs'][str(i)] = {key: dict(path=str(path), sha256=sha(path))
                                              for key, path in sources.items()}
            write(output / 'manifest.json', manifest)
        result = pd.concat(selected, ignore_index=True)
        assert np.isfinite(result[['theta_reference', 'gamma_reference']]).all().all()
        result.to_parquet(output / 'reference_selected.parquet', index=False)
        manifest['files'] = {str(p.relative_to(output)): dict(sha256=sha(p))
                             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        manifest.update(status='completed', rows=len(result))
        write(output / 'manifest.json', manifest)
        print(f'{output}: completed ({len(result)} rows)', flush=True)
    except BaseException:
        manifest['status'] = 'failed'
        write(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    build(args.smoke, args.device, args.output_dir)


if __name__ == '__main__':
    main()
