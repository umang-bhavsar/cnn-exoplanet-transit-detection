from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as f:
        config = yaml.safe_load(f)

    for key in ("cache_dir", "output_dir", "checkpoint_dir"):
        if key in config.get("download", {}):
            config["download"][key] = _resolve(config["download"][key])
        if key in config.get("preprocess", {}):
            config["preprocess"][key] = _resolve(config["preprocess"][key])
        if key in config.get("train", {}):
            config["train"][key] = _resolve(config["train"][key])

    return config


def _resolve(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p)
