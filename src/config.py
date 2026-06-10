"""Load and validate the game configuration YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "clash_royale.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the game config and check the fields the pipeline depends on."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ("observation", "arena", "action", "timing", "ui_regions", "reward")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config {config_path} missing sections: {missing}")

    arena = config["arena"]
    action = config["action"]
    if action["num_cells"] != arena["grid_rows"] * arena["grid_cols"]:
        raise ValueError(
            "action.num_cells must equal arena.grid_rows * arena.grid_cols"
        )
    return config
