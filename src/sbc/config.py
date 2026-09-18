from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "controller_sbc.yaml"


def load_controller_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load the controller's shared physical and tuning parameters."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Controller configuration must be a mapping: {path}")
    return config
