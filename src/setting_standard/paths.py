"""Resolve repository paths, including input paths saved before relocation."""
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
RELOCATED = {
    'training_data': 'artifacts/priors',
    'checkpoints': 'artifacts/checkpoints',
    'data': 'artifacts/evaluation_data',
    'runs': 'artifacts/experiment_outputs',
    'results': 'artifacts/experiment_outputs/reports',
}


def resolve_code_relative_path(value: str) -> str:
    path = Path(value).expanduser()
    text = path.as_posix()
    # Saved prior metadata uses this former package path.
    if 'Setting_standard/training_data/' in text:
        path = CODE_ROOT / 'training_data' / text.split('Setting_standard/training_data/', 1)[1]
    elif '/code/' in text:
        path = CODE_ROOT / text.split('/code/', 1)[1]
    if not path.is_absolute():
        if path.parts and path.parts[0] == 'code':
            path = Path(*path.parts[1:])
        path = CODE_ROOT / path
    if path.is_relative_to(CODE_ROOT):
        relative = path.relative_to(CODE_ROOT)
        if relative.parts and relative.parts[0] in RELOCATED:
            path = CODE_ROOT / RELOCATED[relative.parts[0]] / Path(*relative.parts[1:])
    return str(path.resolve())
