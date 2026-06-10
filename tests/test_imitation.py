"""Offline tests for the BCO pipeline: IDM, labeling, and BC training."""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import PPO

from scripts.extract_demonstrations import crop_frame, parse_crop
from src.agent.bc import downsample_noops, save_bc_checkpoint, train_bc
from src.agent.idm import (
    InverseDynamicsModel,
    label_pairs,
    load_idm,
    save_idm,
    train_idm,
    valid_pair_indices,
)
from src.config import load_config

N = 12


@pytest.fixture
def synthetic_rollout():
    rng = np.random.default_rng(0)
    observations = rng.integers(0, 256, size=(N, 84, 84, 12), dtype=np.uint8)
    actions = np.stack(
        [rng.integers(0, 5, size=N), rng.integers(0, 70, size=N)], axis=1
    ).astype(np.int64)
    return observations, actions


def test_idm_train_label_roundtrip(tmp_path, synthetic_rollout):
    observations, actions = synthetic_rollout
    model = train_idm(observations, actions, epochs=1, batch_size=4, device="cpu")
    assert isinstance(model, InverseDynamicsModel)

    path = save_idm(model, tmp_path / "idm.pt")
    reloaded = load_idm(path, device="cpu")

    labels = label_pairs(reloaded, observations[:-1], observations[1:], device="cpu")
    assert labels.shape == (N - 1, 2)
    assert labels.dtype == np.int64
    assert (labels[:, 0] >= 0).all() and (labels[:, 0] < 5).all()
    assert (labels[:, 1] >= 0).all() and (labels[:, 1] < 70).all()


def test_bc_produces_loadable_ppo_checkpoint(tmp_path, synthetic_rollout):
    observations, actions = synthetic_rollout
    config = load_config("config/clash_royale.yaml")
    model = train_bc(
        observations, actions, config, epochs=1, batch_size=4, device="cpu"
    )
    path = save_bc_checkpoint(model, tmp_path / "bc.zip")

    reloaded = PPO.load(path, device="cpu")
    obs = observations[0]
    action, _ = reloaded.predict(obs, deterministic=True)
    assert action.shape == (2,)


def test_valid_pair_indices_excludes_episode_boundaries():
    starts = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    # Pair (2, 3) spans a reset: obs[3] starts a new episode.
    assert valid_pair_indices(5, starts).tolist() == [0, 1, 3]
    assert valid_pair_indices(5).tolist() == [0, 1, 2, 3]


def test_idm_trains_with_episode_starts(synthetic_rollout):
    observations, actions = synthetic_rollout
    starts = np.zeros(N, dtype=np.float32)
    starts[0] = starts[N // 2] = 1.0
    model = train_idm(
        observations,
        actions,
        episode_starts=starts,
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    assert isinstance(model, InverseDynamicsModel)


def test_downsample_noops_keeps_all_plays(synthetic_rollout):
    observations, actions = synthetic_rollout
    kept_obs, kept_act = downsample_noops(observations, actions, keep_fraction=0.0)
    assert (kept_act[:, 0] > 0).all()
    assert kept_act.shape[0] == (actions[:, 0] > 0).sum()
    assert kept_obs.shape[0] == kept_act.shape[0]

    all_obs, all_act = downsample_noops(observations, actions, keep_fraction=1.0)
    assert all_act.shape == actions.shape


def test_train_bc_rejects_empty_dataset():
    config = load_config("config/clash_royale.yaml")
    empty_obs = np.zeros((0, 84, 84, 12), dtype=np.uint8)
    empty_act = np.zeros((0, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        train_bc(empty_obs, empty_act, config, epochs=1, device="cpu")


def test_crop_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_frame(frame, parse_crop("0.25,0.1,0.25,0.1"))
    assert cropped.shape == (80, 100, 3)


def test_parse_crop_rejects_bad_input():
    with pytest.raises(Exception):
        parse_crop("0.6,0,0,0")
    with pytest.raises(Exception):
        parse_crop("0.1,0.1,0.1")
