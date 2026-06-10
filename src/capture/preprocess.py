"""Pure frame preprocessing shared by all capture backends.

Kept free of any screen/device dependency so it is unit-testable anywhere.
"""

from __future__ import annotations

import numpy as np


def to_observation(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """Convert a raw captured frame to the (size, size, 3) uint8 observation.

    Accepts HxWx3 (RGB/BGR) or HxWx4 (BGRA from mss/Quartz); drops alpha.
    """
    import cv2

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 frame, got shape {frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def is_black_frame(frame: np.ndarray, threshold: float = 2.0) -> bool:
    """Detect the all-black frames produced by copy-protected windows."""
    return float(frame.mean()) < threshold
