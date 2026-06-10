"""Capture the iPhone Mirroring window on macOS.

Primary path: locate the window with Quartz, grab its rect with mss.
Fallback: CGWindowListCreateImage if mss returns a black (copy-protected)
frame. If both return black, the user must disable screen-recording
protection on the iPhone (Settings -> Privacy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.capture.preprocess import is_black_frame, to_observation


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int
    window_id: int = 0  # CGWindow number; 0 = unknown (screen-region grabs only)


def find_window(title: str = "iPhone Mirroring") -> WindowRect:
    """Locate the mirroring window via Quartz. Raises if not found."""
    import Quartz

    options = Quartz.kCGWindowListOptionOnScreenOnly
    windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    for window in windows:
        owner = window.get("kCGWindowOwnerName", "")
        name = window.get("kCGWindowName", "")
        if title in (owner or "") or title in (name or ""):
            bounds = window["kCGWindowBounds"]
            return WindowRect(
                left=int(bounds["X"]),
                top=int(bounds["Y"]),
                width=int(bounds["Width"]),
                height=int(bounds["Height"]),
                window_id=int(window.get("kCGWindowNumber", 0)),
            )
    raise RuntimeError(
        f"Window '{title}' not found. Is iPhone Mirroring open and on screen?"
    )


def grab_raw(rect: WindowRect) -> np.ndarray:
    """Grab the window contents.

    Primary: CGWindowListCreateImage with the window ID — captures the
    window itself, so other windows sitting on top do not corrupt the frame.
    Fallback: mss screen-region grab (requires the window unobstructed).
    """
    frame = None
    if rect.window_id:
        try:
            frame = _grab_window_image(rect.window_id)
        except Exception:
            frame = None

    if frame is None or is_black_frame(frame):
        frame = _grab_screen_region(rect)
        if is_black_frame(frame):
            raise RuntimeError(
                "Captured frame is black via both window-ID and screen grabs. "
                "Disable 'Prevent screen recording' on the iPhone and check "
                "the Screen Recording permission."
            )
    return frame


def _grab_screen_region(rect: WindowRect) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        monitor = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        return np.asarray(sct.grab(monitor))  # BGRA


def _grab_window_image(window_id: int) -> np.ndarray:
    """Capture one window's contents by ID, ignoring overlapping windows."""
    import Quartz

    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming,
    )
    if image is None:
        raise RuntimeError("CGWindowListCreateImage returned no image")
    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
    buffer = np.frombuffer(data, dtype=np.uint8)
    buffer = buffer.reshape((height, bytes_per_row // 4, 4))
    return np.ascontiguousarray(buffer[:, :width, :])


class MacCapture:
    """Stateful capture: resolves the window once, re-resolves on failure."""

    def __init__(self, window_title: str = "iPhone Mirroring", frame_size: int = 84):
        self._title = window_title
        self._frame_size = frame_size
        self._rect: WindowRect | None = None

    def capture_raw(self) -> np.ndarray:
        """Full-resolution BGRA frame for the detector."""
        if self._rect is None:
            self._rect = find_window(self._title)
        try:
            return grab_raw(self._rect)
        except Exception:
            self._rect = find_window(self._title)
            return grab_raw(self._rect)

    def capture(self) -> np.ndarray:
        """Downscaled (size, size, 3) uint8 observation frame."""
        return to_observation(self.capture_raw(), self._frame_size)

    @property
    def rect(self) -> WindowRect:
        if self._rect is None:
            self._rect = find_window(self._title)
        return self._rect


def capture_frame(window_title: str = "iPhone Mirroring", size: int = 84) -> np.ndarray:
    """One-shot helper used by the live smoke test in the README."""
    return MacCapture(window_title, size).capture()
