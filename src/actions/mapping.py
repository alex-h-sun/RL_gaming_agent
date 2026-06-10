"""Pure action-space math: agent action -> normalized screen coordinates.

Shared by Mac and ADB backends; no device dependencies.
"""

from __future__ import annotations

from typing import Any

NO_OP = 0


def cell_to_point(
    cell: int, grid_rows: int, grid_cols: int, arena_region: list[float]
) -> tuple[float, float]:
    """Center of a flattened grid cell as normalized (x, y) screen coords."""
    if not 0 <= cell < grid_rows * grid_cols:
        raise ValueError(f"cell {cell} out of range for {grid_rows}x{grid_cols} grid")
    row, col = divmod(cell, grid_cols)
    ax, ay, aw, ah = arena_region
    x = ax + (col + 0.5) / grid_cols * aw
    y = ay + (row + 0.5) / grid_rows * ah
    return x, y


def card_slot_point(slot: int, ui_regions: dict[str, Any]) -> tuple[float, float]:
    """Center of card slot 1-4 as normalized (x, y) screen coords."""
    if not 1 <= slot <= 4:
        raise ValueError(f"card slot must be 1-4, got {slot}")
    x, y, w, h = ui_regions["card_slots"][slot - 1]
    return x + w / 2.0, y + h / 2.0


def decode_action(
    action: tuple[int, int] | list[int], config: dict[str, Any]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Decode MultiDiscrete([5, 70]) action into (card_point, drop_point).

    Returns None for the no-op action (card == 0).
    """
    card, cell = int(action[0]), int(action[1])
    if card == NO_OP:
        return None
    arena = config["arena"]
    source = card_slot_point(card, config["ui_regions"])
    target = cell_to_point(cell, arena["grid_rows"], arena["grid_cols"], arena["region"])
    return source, target
