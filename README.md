# Amortizing Causal Sensitivity Analysis via Prior-Data Fitted Networks

Supplementary code for causal sensitivity analysis under MSM, KL and Rosenbaum
models. Built with [TabPFN](https://github.com/PriorLabs/TabPFN).

Use Python 3.10–3.12. Install with uv and run a small end-to-end example:

```bash
uv sync --frozen
bash experiments/scripts/run_pipeline.sh example analytical_msm
```

Replace `analytical_msm` with `optimization_kl` or `optimization_rosenbaum` for
other models. For paper-scale runs, replace `example` with `paper` and edit
[the configs](experiments/configs/paper/). Use CUDA for paper-scale optimization
and training. Add `--print-command` to preview a run.

[src/](src/) contains the implementation; [experiments/](experiments/) contains
study scripts and configs. Generated outputs stay under `artifacts/`.

Pretrained checkpoints are not included in this release. Evaluation expects them
at `artifacts/checkpoints/{msm,kl,rosenbaum}/model.pth`.

Licensed under [Apache-2.0](LICENSE); upstream attribution and third-party
license notices are in [NOTICE](NOTICE).
