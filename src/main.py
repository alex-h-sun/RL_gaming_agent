"""CLI entry point.

Examples:
    python -m src.main --platform mac --mode collect --steps 2048
    python -m src.main --mode collect --steps 20 --random          # smoke test
    python -m src.main --mode collect --mock --steps 64            # no game
    python -m src.main --mode eval --model checkpoints/best.zip --episodes 20
    python -m src.main --mode play --model checkpoints/best.zip
"""

from __future__ import annotations

import argparse

from src.config import load_config


def build_env(args, config):
    if args.mock:
        from src.env.mock_game_env import MockGameEnv

        return MockGameEnv()
    from src.env.game_env import GameEnv

    return GameEnv(config, platform=args.platform, curriculum_round=args.round)


def build_model(args, env, config):
    if args.random:
        return None
    from src.agent.model import load_or_init

    return load_or_init(args.model, env, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["mac", "adb"], default="mac")
    parser.add_argument("--mode", choices=["collect", "play", "eval"], required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default="checkpoints/best.zip")
    parser.add_argument("--output", default="rollouts/rollouts.npz")
    parser.add_argument("--config", default=None)
    parser.add_argument("--random", action="store_true", help="use a random policy")
    parser.add_argument("--mock", action="store_true", help="use MockGameEnv (no game)")
    parser.add_argument(
        "--round", type=int, default=0, help="curriculum round index for reward blending"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(args, config)
    model = build_model(args, env, config)

    if args.mode == "collect":
        from src.agent.collect import collect_rollouts

        collect_rollouts(env, model, args.steps, args.output, random_policy=args.random)
    elif args.mode == "eval":
        from src.agent.evaluate import evaluate

        evaluate(env, model, episodes=args.episodes)
    elif args.mode == "play":
        from src.agent.evaluate import run_episode

        while True:
            result = run_episode(env, model)
            print(f"{result.outcome}: crowns {result.crowns_won}-{result.crowns_lost}")


if __name__ == "__main__":
    main()
