"""Map the config's `game:` key to its adapter class."""

from __future__ import annotations

from typing import Any

from src.games.base import GameAdapter


def make_adapter(config: dict[str, Any]) -> GameAdapter:
    game = config.get("game")
    if game == "clash_royale":
        from src.games.clash_royale.adapter import ClashRoyaleAdapter

        return ClashRoyaleAdapter(config)
    if game == "brawl_stars":
        from src.games.brawl_stars.adapter import BrawlStarsAdapter

        return BrawlStarsAdapter(config)
    raise ValueError(f"Unknown game in config: {game!r}")
