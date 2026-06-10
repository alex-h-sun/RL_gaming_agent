"""Calibration helper: overlay the config's UI regions on a captured frame.

Captures the game window (or loads a saved frame), draws every ui_region
rectangle and tap point with labels, saves calibration_overlay.png, and
prints the detected game state. Iterate: adjust the YAML, re-run, repeat
until the boxes sit on the real UI elements and the state reads correctly.

Usage:
    python -m scripts.calibrate                        # capture live (Clash Royale)
    python -m scripts.calibrate --config config/brawl_stars.yaml
    python -m scripts.calibrate --image calibration_frame.png
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.config import load_config
from src.games.registry import make_adapter

BOX_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 0)


def capture_live(config) -> np.ndarray:
    from src.capture.mac_capture import MacCapture

    raw = MacCapture(config["window"]["mac_title"]).capture_raw()
    return raw[:, :, 2::-1]  # BGRA -> RGB


def draw_overlay(rgb: np.ndarray, ui_regions: dict) -> np.ndarray:
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    for name, region in ui_regions.items():
        entries = region if isinstance(region[0], list) else [region]
        for i, (x, y, w, h) in enumerate(entries):
            label = f"{name}[{i}]" if len(entries) > 1 else name
            px, py = int(x * width), int(y * height)
            if w == 0.0 and h == 0.0:  # tap point
                cv2.circle(bgr, (px, py), 6, POINT_COLOR, 2)
            else:
                cv2.rectangle(
                    bgr, (px, py), (int((x + w) * width), int((y + h) * height)),
                    BOX_COLOR, 1,
                )
            cv2.putText(bgr, label, (px, max(py - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", default=None,
                        help="use a saved RGB frame instead of capturing live")
    parser.add_argument("--output", default="calibration_overlay.png")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.image:
        rgb = cv2.imread(args.image)[:, :, ::-1]
    else:
        rgb = capture_live(config)

    cv2.imwrite(args.output, draw_overlay(rgb, config["ui_regions"]))
    print(f"Overlay saved: {args.output}  (frame {rgb.shape[1]}x{rgb.shape[0]})")

    state = make_adapter(config).detect(rgb)
    print(f"Detected state: {state}")


if __name__ == "__main__":
    main()
