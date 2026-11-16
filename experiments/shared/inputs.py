"""Current evaluation cohorts and scientific inputs."""
from pathlib import Path
import json
import yaml
from experiments.shared.paths import ROOT, EXPERIMENTS

CONFIG = EXPERIMENTS / 'configs/evaluation.yaml'

def evaluation_config():
    return yaml.safe_load(CONFIG.read_text())

def input_path(name):
    path = Path(evaluation_config()['inputs'][name])
    return path if path.is_absolute() else ROOT / path

def cohort_ids(name):
    cohort = evaluation_config()['cohorts'][name]
    first, count = cohort['first_id'], cohort['count']
    assert isinstance(first, int) and first > 0 and isinstance(count, int) and count > 0
    return list(range(first, first + count))

def overlap_edges():
    return evaluation_config()['overlap_quartile_edges']



def kl_reference(smoke=False):
    """Native, head-specific attained-Gamma rows and their original outcome scales."""
    import pandas as pd
    root = input_path('kl_reference_root')
    if smoke:
        root = root.with_name(root.name + '_smoke')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['y_fit_column'] in {'outcome', 'pooled_potential_outcomes'}
    from experiments.shared.runtime import sha
    for relative, identity in manifest['files'].items():
        assert sha(root / relative) == identity['sha256'], f'Changed KL reference input: {relative}'
    frame = pd.read_parquet(root / 'reference_selected.parquet')
    ids = cohort_ids('kl')[:2] if smoke else cohort_ids('kl')
    assert set(ids) <= set(frame.dataset_id)
    if 'base_seed' in manifest:
        assert manifest['base_seed'] == evaluation_config()['base_seed']
    for i in ids:
        scaler = json.loads((root / 'scalers' / f'scaler_{i}.json').read_text())
        assert scaler['y_fit_column'] == manifest['y_fit_column']
    if manifest.get('inputs'):
        for source in manifest['inputs'].values():
            for item in source.values():
                assert sha(item['path']) == item['sha256'], f"Changed KL source: {item['path']}"
    assert set(frame.bound_type) == {'lower', 'upper'}
    assert not frame.duplicated(['dataset_id', 'source_row_index', 'a_query', 'bound_type', 'lambda_index']).any()
    return frame

def normalized_root():
    return input_path('normalized_root')

def cohort_artifacts(name, ids=None):
    """Resolve current observed-only contexts and reject a different seed cohort."""
    from setting_standard.causaldataset import discover_causal_artifacts
    artifacts = discover_causal_artifacts(str(normalized_root()))
    indexed = {item['dataset_id']: item for item in artifacts}
    assert len(indexed) == len(artifacts)
    requested = cohort_ids(name)
    ids = requested if ids is None else list(ids)
    assert ids and set(ids) <= set(requested), (name, ids)
    assert set(ids) <= set(indexed), f'Missing current {name} DGPs: {set(ids)-set(indexed)}'
    base_seed = evaluation_config()['base_seed']
    for i in ids:
        metadata = indexed[i]['metadata']
        assert metadata['seed'] == base_seed + i, (i, metadata['seed'], base_seed + i)
        assert metadata['normalization']['y_fit_column'] == 'outcome', (i, 'Expected observed-only context scaling')
    return {i: indexed[i] for i in ids}
