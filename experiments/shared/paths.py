"""Repository inputs, runtime outputs, and report directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT
EXPERIMENTS = ROOT / 'experiments'
ARTIFACTS = ROOT / 'artifacts'
PRIORS = ARTIFACTS / 'priors'
CHECKPOINTS = ARTIFACTS / 'checkpoints'
RUNS = ARTIFACTS / 'experiment_outputs'
REPORTS = RUNS / 'reports'
RESULTS = REPORTS
DATA = ARTIFACTS / 'evaluation_data'
OBSERVED_RUNS = RUNS / 'observed'
OBSERVED_DATA = DATA / 'observed'
OBSERVED_RESULTS = REPORTS

OBSERVED_STUDIES = {
    'e1': 'kl_in_prior', 'e2': 'msm_in_prior', 'e3': 'neuralcsa',
    'e4': 'point_identification', 'e7_context': 'context_length',
    'e7_dimension': 'covariate_dimension', 'e9': 'out_of_prior',
}


def observed_report(study, smoke=False):
    root = REPORTS / 'smoke' if smoke else REPORTS
    return root / OBSERVED_STUDIES.get(study, study)
