"""Label demonstration pairs with the IDM and run behavioral cloning.

Usage:
    .venv/bin/python -m scripts.train_bc data/demos/pairs.npz \
        --idm checkpoints/idm.pt [--out checkpoints/bc_pretrained.zip] \
        [--epochs 5] [--config config/clash_royale.yaml]

Output is a standard SB3 PPO .zip — copy it to checkpoints/best.zip to
use it as the starting policy for live collection rounds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.agent.bc import downsample_noops, save_bc_checkpoint, train_bc
from src.agent.idm import label_pairs, load_idm
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", help="demonstration pairs .npz")
    parser.add_argument("--idm", required=True, help="trained IDM checkpoint (.pt)")
    parser.add_argument("--out", default="checkpoints/bc_pretrained.zip")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--config", default="config/clash_royale.yaml")
    parser.add_argument(
        "--keep-noop-fraction", type=float, default=0.25,
        help="downsample no-op-labeled pairs to this fraction (most video "
        "frames have no card play; cloning them all yields a do-nothing policy)",
    )
    args = parser.parse_args()

    with np.load(args.pairs) as data:
        observations = data["observations"]
        next_observations = data["next_observations"]

    idm = load_idm(args.idm)
    actions = label_pairs(idm, observations, next_observations)
    played = actions[:, 0] > 0
    print(
        f"Labeled {len(actions)} pairs: {played.sum()} card plays, "
        f"{(~played).sum()} no-ops"
    )

    observations, actions = downsample_noops(
        observations, actions, args.keep_noop_fraction
    )
    print(f"Training BC on {len(actions)} examples after no-op downsampling")

    config = load_config(args.config)
    model = train_bc(observations, actions, config, epochs=args.epochs)
    path = save_bc_checkpoint(model, Path(args.out))
    print(f"Saved BC-pretrained checkpoint: {path}")


if __name__ == "__main__":
    main()
