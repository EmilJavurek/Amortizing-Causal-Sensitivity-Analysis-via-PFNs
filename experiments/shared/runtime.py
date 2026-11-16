"""Fresh experiment outputs, provenance, resources, and clustered bootstrap."""

import sys
from pathlib import Path
from experiments.shared.paths import CODE, DATA, OBSERVED_RUNS, ROOT, RUNS
import hashlib
import json
import os
import subprocess
import time
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + '\n')
    tmp.replace(path)

def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def provenance():
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                branch=git('branch', '--show-current'))

def check_resources(gpu):
    resources = dict(cpu_affinity=sorted(os.sched_getaffinity(0)))
    if gpu:
        import torch
        assert torch.cuda.is_available(), 'This experiment requires CUDA'
        device = torch.cuda.get_device_properties(0)
        resources.update(gpu_name=device.name, gpu_uuid=str(device.uuid),
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return resources

def set_threads(threads):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import torch
    torch.set_num_threads(threads)

def fresh(path):
    path = Path(path).resolve()
    assert any(path.is_relative_to(root) for root in (RUNS, DATA)), path
    assert not path.exists(), f'Refusing existing output: {path}'
    path.mkdir(parents=True)
    return path

class Run:
    """Fresh outputs; dataset builds keep minimal metadata with the dataset."""

    def __init__(self, package, name, smoke=False, gpu=False, threads=4, dataset=False, **config):
        self.resources = check_resources(gpu)
        set_threads(threads)
        self.dataset = dataset
        self.path = fresh(DATA / package / ('smoke' if smoke else 'retained') if dataset else RUNS / package / name)
        self.gpu = gpu
        self.started = time.time()
        self.manifest = dict(package=package, name=name, status='running', smoke=smoke,
                             started_unix=self.started, git=provenance(), argv=sys.argv,
                             threads=threads, resources=self.resources, config=config)
        self.save()

    def save(self):
        metadata = self.manifest
        if self.dataset:
            # Keep the scientific recipe, source identity, and completion with datasets.
            metadata = {k: metadata[k] for k in ('status', 'smoke', 'config', 'git', 'argv', 'summary') if k in metadata}
            metadata['git'] = {k: metadata['git'][k] for k in ('commit', 'dirty')}
        write(self.path / 'manifest.json', metadata)

    def done(self, status='completed', **summary):
        assert status in {'completed', 'failed', 'interrupted'}, status
        if self.gpu:
            import torch
            torch.cuda.synchronize()
        wall = time.time() - self.started
        self.manifest.update(status=status, wall_seconds=wall, finished_unix=time.time(), summary=summary)
        self.save()
        print(f'{self.path.relative_to(CODE)}: {status}, wall={wall:.1f}s', flush=True)

def dataset_root(package, smoke=False):
    """A built dataset owns its recipe and completion metadata."""
    path = DATA / package / ('smoke' if smoke else 'retained')
    assert read(path / 'manifest.json')['status'] == 'completed', f'Dataset build incomplete: {path}'
    return path

def completed(package, name):
    """Return the run directory if a completed manifest exists, else None."""
    path = RUNS / package / name / 'manifest.json'
    return path.parent if path.exists() and read(path)['status'] == 'completed' else None

def cluster_bootstrap(frame, cluster, statistic, replicates=2000, seed=20261107):
    """Whole-cluster percentile bootstrap of `statistic(frame) -> dict[str, float]`."""
    groups = {key: index for key, index in frame.groupby(cluster).indices.items()}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    point = statistic(frame)
    draws = {name: [] for name in point}
    for _ in range(replicates):
        sample = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([groups[key] for key in sample])
        value = statistic(frame.iloc[rows])
        for name in point:
            draws[name].append(value[name])
    out = {}
    for name, estimate in point.items():
        values = np.asarray(draws[name], dtype=float)
        out[name] = estimate
        out[f'{name}_lo'] = float(np.nanquantile(values, 0.025))
        out[f'{name}_hi'] = float(np.nanquantile(values, 0.975))
    return out

def canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def hardware_snapshot() -> list[str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,"
            "utilization.gpu,utilization.memory,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return output.strip().splitlines()
