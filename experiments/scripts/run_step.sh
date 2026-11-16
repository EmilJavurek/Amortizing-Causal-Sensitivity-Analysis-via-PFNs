#!/usr/bin/env bash
# Run one reviewed pipeline step from one experiment set and frontier branch.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <experiment_set> <analytical_msm|optimization_kl|optimization_rosenbaum> <01_generate|02_normalize|03_frontier|03_relabel|04_postprocess|05_train> [--print-command] [target-args...]" >&2
    exit 2
fi

EXPERIMENT_SET="$1"
BRANCH="$2"
STEP="$3"
shift 3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$(dirname "$EXPERIMENTS_DIR")"
CONFIG_DIR="${EXPERIMENTS_DIR}/configs/${EXPERIMENT_SET}/${BRANCH}"
RUNNER="${SCRIPT_DIR}/_run.py"

assert_name() {
    if [[ ! "$1" =~ ^[a-z0-9_]+$ ]]; then
        echo "ERROR: names may contain only lowercase letters, digits, and underscores: $1" >&2
        exit 2
    fi
}

assert_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file does not exist: $1" >&2
        exit 1
    fi
}

assert_name "$EXPERIMENT_SET"
assert_name "$BRANCH"

case "$BRANCH" in
    analytical_msm|optimization_kl|optimization_rosenbaum) ;;
    *)
        echo "ERROR: unknown frontier branch: ${BRANCH}" >&2
        exit 2
        ;;
esac

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: experiment configuration directory does not exist: $CONFIG_DIR" >&2
    exit 1
fi

TARGET_ARGS=()
case "$STEP" in
    01_generate)
        PYTHON_SCRIPT="${CODE_DIR}/src/setting_standard/gen_standard_syn.py"
        CONFIG_FILE="${CONFIG_DIR}/01_generate.yaml"
        ;;
    02_normalize)
        PYTHON_SCRIPT="${CODE_DIR}/src/setting_standard/normalize_data.py"
        CONFIG_FILE="${CONFIG_DIR}/02_normalize.yaml"
        ;;
    03_frontier)
        if [[ "$BRANCH" == "analytical_msm" ]]; then
            PYTHON_SCRIPT="${CODE_DIR}/src/setting_standard/frontier_msm.py"
        else
            PYTHON_SCRIPT="${CODE_DIR}/src/setting_standard/frontier_optimization.py"
        fi
        CONFIG_FILE="${CONFIG_DIR}/03_frontier.yaml"
        ;;
    03_relabel)
        [[ "$BRANCH" == "optimization_rosenbaum" ]] || { echo "ERROR: dense relabeling requires optimization_rosenbaum" >&2; exit 2; }
        PYTHON_SCRIPT="${CODE_DIR}/experiments/label_accuracy/rosenbaum.py"
        CONFIG_FILE="${CONFIG_DIR}/03_relabel.yaml"
        TARGET_ARGS=(relabel)
        ;;
    04_postprocess)
        PYTHON_SCRIPT="${CODE_DIR}/src/setting_standard/postprocess_frontiers.py"
        CONFIG_FILE="${CONFIG_DIR}/04_postprocess.yaml"
        ;;
    05_train)
        PYTHON_SCRIPT="${CODE_DIR}/src/tabpfn/train_standard/training_standard.py"
        CONFIG_FILE="${CONFIG_DIR}/05_train.yaml"
        ;;
    *)
        echo "ERROR: unknown step: ${STEP}" >&2
        exit 2
        ;;
esac

assert_file "$RUNNER"
assert_file "$PYTHON_SCRIPT"
assert_file "$CONFIG_FILE"

cd "$CODE_DIR"
PYTHON_BIN="${PI_FM_PYTHON:-${CODE_DIR}/.venv/bin/python}"
assert_file "$PYTHON_BIN"
"$PYTHON_BIN" "$RUNNER" "$EXPERIMENT_SET" "$BRANCH" "$STEP" "$PYTHON_SCRIPT" "$CONFIG_FILE" "${TARGET_ARGS[@]}" "$@"
