"""Factual-context evaluation protocol on the common experiment runtime."""
from experiments.shared.runtime import Run, sha
from experiments.shared.inputs import CONFIG, evaluation_config


def observed_run(name, smoke=False, gpu=True):
    run = Run('observed', name, smoke=smoke, gpu=gpu, threads=2, y_fit_column='outcome',
              evaluation=evaluation_config(), evaluation_config_sha256=sha(CONFIG))
    run.manifest.update(y_fit_column='outcome', device='cuda:0' if gpu else 'cpu',
                        cpu_threads=2, completed_ids=[])
    run.save()
    return run
