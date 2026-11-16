#!/usr/bin/env python3
"""Launch one YAML recipe; trailing target CLI arguments override recipe values.

Nulls omit an argument, booleans emit store_true flags, quoted 'True'/'False' pass
values, and dataset_ids_range expands an inclusive [first, last] ID interval.
--print-command prints the final command without importing or executing its target.
"""
import base64
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import yaml


def config_to_args(cfg: dict) -> list[str]:
    assert isinstance(cfg, dict), "Recipe must be a YAML mapping"
    cfg = {key: val for key, val in cfg.items() if val is not None}
    args: list[str] = []
    if "dataset_ids_range" in cfg:
        assert "dataset_ids" not in cfg, "Choose dataset_ids or dataset_ids_range"
        bounds = cfg.pop("dataset_ids_range")
        assert isinstance(bounds, list) and len(bounds) == 2, "Expected [first_id, last_id]"
        assert all(type(value) is int for value in bounds), bounds
        first, last = bounds
        assert 0 <= first <= last, bounds
        args += ["--dataset_ids"] + [str(i) for i in range(first, last + 1)]
    for key, val in cfg.items():
        assert isinstance(key, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", key), key
        if key.startswith("_"):
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
        elif isinstance(val, list):
            assert val and all(isinstance(v, (str, int, float)) for v in val), (key, val)
            args += [flag] + [str(v) for v in val]
        else:
            assert isinstance(val, (str, int, float)), (key, val)
            args += [flag, str(val)]
    return args


def main() -> None:
    assert len(sys.argv) >= 6, (
        "Usage: _run.py <set> <branch> <step> <script.py> <config.yaml> "
        "[--print-command] [target-args...]"
    )
    experiment_set, branch, step = sys.argv[1:4]
    target_script, config_file = map(pathlib.Path, sys.argv[4:6])
    extra_args = sys.argv[6:]
    print_command = "--print-command" in extra_args
    extra_args = [arg for arg in extra_args if arg != "--print-command"]
    assert target_script.is_file(), f"Script not found: {target_script}"
    assert config_file.is_file(), f"Config not found: {config_file}"
    config_bytes = config_file.read_bytes()
    cmd = [sys.executable, str(target_script)] + config_to_args(yaml.safe_load(config_bytes)) + extra_args
    if print_command:
        print(shlex.join(cmd))
        return
    project_root = pathlib.Path(__file__).resolve().parents[2]
    git_provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True
        ).strip()),
    }
    env = os.environ.copy()
    env.update({
        "PI_FM_PIPELINE_STEP": step,
        "PI_FM_CONFIG_PATH": str(config_file.resolve()),
        "PI_FM_CONFIG_BYTES_BASE64": base64.b64encode(config_bytes).decode("ascii"),
        "PI_FM_LAUNCHER_ARGV_JSON": json.dumps([
            "bash", "experiments/scripts/run_step.sh", experiment_set, branch, step, *extra_args,
        ]),
        "PI_FM_TARGET_ARGV_JSON": json.dumps(cmd),
        "PI_FM_GIT_PROVENANCE_JSON": json.dumps(git_provenance),
    })
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
