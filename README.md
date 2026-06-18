# RL Gaming Agent — Clash Royale

Purely for educational purposes

A reinforcement learning agent that plays Clash Royale through iPhone Mirroring
on macOS (or an Android emulator via ADB). It captures game frames, extracts
game state with CV, injects touch actions, and trains with PPO using a
decoupled actor-learner loop: the Mac collects experience, Google Colab runs
the gradient updates.

See `PLAN.md` for the full design (reward shaping, curriculum, action space)
and `TRAINING.md` for the step-by-step training workflow. `IMITATION.md`
covers optional BC pre-training from YouTube gameplay (BCO).

## Setup

Local (Mac, capture + collection):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
```

Colab (training): handled by `notebooks/train.ipynb` via `requirements-train.txt`.
For local pipeline tests you also need `pip install -r requirements-train.txt`.

macOS permissions: grant Terminal both Screen Recording and Accessibility in
System Settings -> Privacy & Security. If captures come back black, disable
screen-recording protection on the iPhone.

## Training loop

```
Mac (actor)                              Colab (learner)
1. python -m src.main --mode collect     4. open notebooks/train.ipynb
   --steps 2048 --round <k>              5. upload rollouts.npz (+ best.zip)
2. saves rollouts/rollouts.npz           6. run all cells -> PPO update
3. upload to Colab  ───────────────────> 7. download updated best.zip
        ^                                            │
        └────────── place in checkpoints/ ───────────┘
```

`--round <k>` is the iteration count; it drives the reward curriculum
(full shaping -> 50/50 blend -> pure win/loss, switching every 200 rounds).

## Commands

```bash
# Unit tests (no game needed)
.venv/bin/python -m pytest tests/

# Capture smoke test (game open in iPhone Mirroring)
python -c "from src.capture.mac_capture import capture_frame; f = capture_frame(); print(f.shape)"

# Env smoke test: 20 random steps against the live game
python -m src.main --mode collect --steps 20 --random

# Full collection round
python -m src.main --mode collect --steps 2048 --round 0

# Pipeline test without any game (mock env)
python -m src.main --mode collect --mock --steps 64 --random

# Evaluate a checkpoint over 20 live episodes
python -m src.main --mode eval --model checkpoints/best.zip --episodes 20

# v2: Brawl Stars (separate config and checkpoints)
python -m src.main --mode collect --steps 2048 --config config/brawl_stars.yaml \
    --model checkpoints/brawl_best.zip --output rollouts/brawl_rollouts.npz

# v2: Android emulator instead of iPhone Mirroring
python -m src.main --mode collect --steps 2048 --platform adb --serial emulator-5554
```

## Layout

```
src/
  capture/    mac_capture.py (Quartz + mss), adb_capture.py, preprocess.py
  actions/    mapping.py (action -> coords), mac_actions.py, adb_actions.py
  games/      base.py (Gesture + GameAdapter protocol), registry.py
  games/clash_royale/
              detector.py (screen state, elixir, tower HP, hand)
              reward.py   (PBRS + destroy + activation + waste + curriculum)
              state.py    (GameState / ScreenState / TowerHP)
              adapter.py  (GameAdapter implementation)
  games/brawl_stars/
              detector.py (screen state, own HP, ammo, super)
              reward.py   (HP PBRS + survival + victory/defeat)
              adapter.py  (joystick movement + attack/super actions)
  env/        game_env.py (live), mock_game_env.py (synthetic, same spaces)
  agent/      model.py (PPO factory), collect.py (actor), rollout_io.py
              (.npz contract), train_colab.py (learner), evaluate.py
config/       clash_royale.yaml (UI regions, deck, reward and PPO params)
notebooks/    train.ipynb (Colab learner), mock_train_test.ipynb (sanity)
tests/        offline unit tests + synthetic frame fixtures
```

## Calibration notes

The UI regions in `config/clash_royale.yaml` are normalized estimates.
Before the first live run, capture a real frame (`capture_frame()` at full
resolution via `MacCapture.capture_raw()`), overlay the regions, and adjust
the YAML until elixir/HP readings match the screen. Card icon templates for
hand detection are optional in v1 (the agent plays by slot index).
