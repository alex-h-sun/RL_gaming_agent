"""collect.py output: .npz keys/shapes/dtypes correct, loadable into SB3,
and one full PPO gradient update runs on it (the Colab path, locally)."""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from src.agent.collect import collect_rollouts
from src.agent.rollout_io import REQUIRED_KEYS, load_rollouts
from src.env.mock_game_env import MockGameEnv

N_STEPS = 32


@pytest.fixture(scope="module")
def rollout_file(tmp_path_factory):
    env = MockGameEnv(min_episode_steps=8, max_episode_steps=12)
    path = tmp_path_factory.mktemp("rollouts") / "rollouts.npz"
    return collect_rollouts(env, model=None, n_steps=N_STEPS, output_path=path,
                            random_policy=True)


class TestNpzFormat:
    def test_required_keys_present(self, rollout_file):
        arrays = load_rollouts(rollout_file)
        assert set(REQUIRED_KEYS) <= set(arrays)

    def test_shapes_and_dtypes(self, rollout_file):
        arrays = load_rollouts(rollout_file)
        assert arrays["observations"].shape == (N_STEPS, 84, 84, 12)
        assert arrays["observations"].dtype == np.uint8
        assert arrays["actions"].shape == (N_STEPS, 2)
        assert arrays["actions"].dtype == np.int64
        for key in ("rewards", "episode_starts", "values", "log_probs"):
            assert arrays[key].shape == (N_STEPS,)
            assert arrays[key].dtype == np.float32
        assert arrays["last_value"].shape == ()
        assert arrays["last_done"].shape == ()

    def test_episode_starts_marks_resets(self, rollout_file):
        arrays = load_rollouts(rollout_file)
        starts = arrays["episode_starts"]
        assert starts[0] == 1.0
        # episodes are 8-12 steps, so 32 steps must contain several resets
        assert starts.sum() >= 2

    def test_actions_in_space(self, rollout_file):
        arrays = load_rollouts(rollout_file)
        actions = arrays["actions"]
        assert actions[:, 0].min() >= 0 and actions[:, 0].max() < 5
        assert actions[:, 1].min() >= 0 and actions[:, 1].max() < 70


class TestColabTrainingPath:
    def test_gradient_update_on_rollouts(self, rollout_file, tmp_path, config):
        """End-to-end learner step: load .npz -> PPO update -> save .zip."""
        from src.agent.train_colab import train_on_rollouts

        small = {**config, "ppo": {**config["ppo"], "n_steps": N_STEPS,
                                   "batch_size": 16, "n_epochs": 1}}
        output = tmp_path / "best.zip"
        model = train_on_rollouts(
            rollout_file, checkpoint_path=None, output_path=output,
            config=small, device="cpu",
        )
        assert output.exists()
        assert model.rollout_buffer.full

    def test_policy_collection_with_model(self, tmp_path, config):
        """Round-trip: a real policy collects (values/log_probs populated)."""
        from src.agent.model import make_ppo

        env = MockGameEnv(min_episode_steps=4, max_episode_steps=6)
        small = {**config, "ppo": {**config["ppo"], "n_steps": 8,
                                   "batch_size": 4, "n_epochs": 1}}
        model = make_ppo(env, small, device="cpu")
        path = collect_rollouts(env, model, n_steps=8,
                                output_path=tmp_path / "r.npz")
        arrays = load_rollouts(path)
        assert np.abs(arrays["log_probs"]).sum() > 0.0
