#!/usr/bin/env bash
# Run a complete fresh recipe, including Rosenbaum dense relabeling.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != --print-command ) ]]; then
    echo "Usage: $0 <experiment_set> <analytical_msm|optimization_kl|optimization_rosenbaum> [--print-command]" >&2
    exit 2
fi
EXPERIMENT_SET="$1"
BRANCH="$2"
shift 2
[[ "$EXPERIMENT_SET" =~ ^[a-z0-9_]+$ && "$BRANCH" =~ ^[a-z0-9_]+$ ]] || exit 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_BIN="${PI_FM_PYTHON:-${CODE_DIR}/.venv/bin/python}"

# Check the entire recipe before Step 1 creates any artifacts.
"$PYTHON_BIN" - "$CODE_DIR" "$EXPERIMENT_SET" "$BRANCH" "$@" <<'PY'
from pathlib import Path
import sys
import yaml
code = Path(sys.argv[1])
config = code / 'experiments/configs' / sys.argv[2] / sys.argv[3]
steps = ['01_generate', '02_normalize', '03_frontier', '04_postprocess', '05_train']
recipes = []
for step in steps:
    path = config / f'{step}.yaml'
    assert path.is_file(), f'Incomplete pipeline recipe: {path}; use run_step.sh for individual stages'
    recipes.append(yaml.safe_load(path.read_bytes()))
run_dirs = [(code / recipe['run_dir']).resolve() for recipe in recipes[:4]]
assert len(set(run_dirs[:3])) == 1, 'Generation, normalization and frontier stages must share run_dir'
outputs = [run_dirs[0]]
if sys.argv[3] == 'optimization_rosenbaum':
    relabel_path = config / '03_relabel.yaml'
    assert relabel_path.is_file(), f'Missing dense relabel recipe: {relabel_path}'
    relabel = yaml.safe_load(relabel_path.read_bytes())
    assert (code / relabel['source']).resolve() == run_dirs[2], 'Dense relabel must consume Step 3 output'
    assert (code / relabel['dest']).resolve() == run_dirs[3], 'Step 4 must consume dense relabel output'
    assert recipes[2]['save_spline_params'], 'Dense relabel requires saved spline parameters'
    assert run_dirs[3] != run_dirs[2], 'Dense relabel destination must differ from source'
    outputs.append(run_dirs[3])
else:
    assert len(set(run_dirs)) == 1, 'Steps 1–4 must share run_dir'
assert (code / recipes[4]['data_path']).resolve() == run_dirs[3], 'Training must consume Step 4 output'
assert not any(recipe.get(key) for recipe in recipes for key in ('source_run_dir', 'resume', 'extend', 'resume_from')), 'Pipeline recipes must start fresh'
if '--print-command' not in sys.argv[4:]:
    for path in [*outputs, *(code / recipes[4][key] for key in ('save_dir', 'log_dir'))]:
        assert not path.exists(), f'Pipeline requires fresh outputs: {path}'
PY
STEPS=(01_generate 02_normalize 03_frontier)
if [[ "$BRANCH" == "optimization_rosenbaum" ]]; then
    STEPS+=(03_relabel)
fi
STEPS+=(04_postprocess 05_train)
for STEP in "${STEPS[@]}"; do
    bash "$SCRIPT_DIR/run_step.sh" "$EXPERIMENT_SET" "$BRANCH" "$STEP" "$@"
done
