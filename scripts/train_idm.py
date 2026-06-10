"""Train the Inverse Dynamics Model from collected live rollouts.

Usage:
    .venv/bin/python -m scripts.train_idm rollouts/rollouts.npz \
        [more.npz ...] [--out checkpoints/idm.pt] [--epochs 10]

The rollouts carry real action labels, so the IDM learns
(obs_t, obs_{t+1}) -> action. More rollouts (especially from varied
policies) make a better IDM; random-policy rollouts are fine.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agent.idm import save_idm, train_idm
from src.agent.rollout_io import load_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", help="rollout .npz files")
    parser.add_argument("--out", default="checkpoints/idm.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Concatenating files creates one bogus pair per file boundary —
    # negligible noise relative to dataset size.
    all_obs, all_act = [], []
    for path in args.rollouts:
        arrays = load_rollouts(path)
        all_obs.append(arrays["observations"])
        all_act.append(arrays["actions"])
        print(f"{path}: {len(arrays['observations']) - 1} pairs")

    observations = np.concatenate(all_obs)
    actions = np.concatenate(all_act)
    model = train_idm(
        observations, actions, epochs=args.epochs, batch_size=args.batch_size
    )
    path = save_idm(model, args.out)
    print(f"Saved IDM: {path}")


if __name__ == "__main__":
    main()
