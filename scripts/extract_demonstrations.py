"""Extract action-free demonstration pairs from gameplay videos.

Samples frames at the env's step rate (4 FPS), crops to the gameplay
region, preprocesses to 84x84 RGB, builds 4-frame stacks, and saves
consecutive (obs_t, obs_{t+1}) pairs for IDM labeling.

Usage:
    .venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 \
        --out data/demos/pairs.npz [--fps 4] [--crop L,T,R,B]

--crop takes normalized fractions trimmed from each edge (left, top,
right, bottom), e.g. --crop 0.3,0.0,0.3,0.0 cuts pillarboxing from a
16:9 video that shows portrait gameplay in the center. Use
--preview-frame to dump one cropped frame as a PNG and check the crop
before extracting everything.

Output .npz contract:
    observations       (n, 84, 84, 12) uint8
    next_observations  (n, 84, 84, 12) uint8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_crop(text: str) -> tuple[float, float, float, float]:
    parts = [float(part) for part in text.split(",")]
    if len(parts) != 4 or not all(0.0 <= part < 0.5 for part in parts):
        raise argparse.ArgumentTypeError("crop must be 4 fractions in [0, 0.5)")
    return tuple(parts)  # type: ignore[return-value]


def crop_frame(frame: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top, right, bottom = crop
    return frame[
        int(height * top) : height - int(height * bottom),
        int(width * left) : width - int(width * right),
    ]


def extract_video(
    path: Path, fps: float, crop: tuple[float, float, float, float]
) -> list[np.ndarray]:
    """Return the per-step 84x84x12 observation stacks for one video."""
    import cv2

    from src.capture.preprocess import to_observation

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"Cannot open video: {path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(video_fps / fps))

    stacks: list[np.ndarray] = []
    recent: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            rgb = cv2.cvtColor(crop_frame(frame, crop), cv2.COLOR_BGR2RGB)
            recent.append(to_observation(rgb))
            recent = recent[-4:]
            if len(recent) == 4:
                stacks.append(np.concatenate(recent, axis=-1))
        index += 1
    capture.release()
    return stacks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="video files (.mp4)")
    parser.add_argument("--out", default="data/demos/pairs.npz")
    parser.add_argument("--fps", type=float, default=4.0, help="sample rate")
    parser.add_argument(
        "--crop", type=parse_crop, default=(0.0, 0.0, 0.0, 0.0),
        help="normalized trim: left,top,right,bottom",
    )
    parser.add_argument(
        "--preview-frame", action="store_true",
        help="save one cropped frame per video as <video>_crop_preview.png and exit",
    )
    args = parser.parse_args()

    if args.preview_frame:
        import cv2

        for video in args.videos:
            capture = cv2.VideoCapture(video)
            # Frame 300 skips intros; fall back to the start for short videos.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 300)
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            capture.release()
            if not ok:
                print(f"Could not read a frame from {video}")
                continue
            preview = Path(video).stem + "_crop_preview.png"
            cv2.imwrite(preview, crop_frame(frame, args.crop))
            print(f"Wrote {preview} — verify only the arena/gameplay is visible")
        return

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    for video in args.videos:
        stacks = extract_video(Path(video), args.fps, args.crop)
        # Pairs never span videos: each video contributes its own transitions.
        observations.extend(stacks[:-1])
        next_observations.extend(stacks[1:])
        print(f"{video}: {max(0, len(stacks) - 1)} pairs")

    if not observations:
        raise SystemExit("No pairs extracted — are the videos long enough?")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.stack(observations),
        next_observations=np.stack(next_observations),
    )
    print(f"Saved {len(observations)} pairs to {out}")


if __name__ == "__main__":
    main()
