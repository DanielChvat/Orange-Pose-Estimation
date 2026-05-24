import json
from argparse import Namespace
from pathlib import Path


DEFAULT_CONFIG = "configs/gaussian_pipeline.json"


def repo_root():
    return Path(__file__).resolve().parent


def resolve_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def load_config(path=DEFAULT_CONFIG):
    path = resolve_config(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_section(path, section):
    config = load_config(path)
    values = config.get(section)
    if not isinstance(values, dict):
        raise KeyError(f"Config {resolve_config(path)} has no object section named '{section}'")
    return dict(values)


def namespace_from_config(path, section, overrides=None):
    values = load_section(path, section)
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
    return Namespace(**values)
