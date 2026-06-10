# Refined Plan: Mobile Game RL Agent

## Context

Build a reinforcement learning agent that autonomously plays Clash Royale via **iPhone Mirroring on Mac** (macOS 15 Sequoia, primary) or **Android emulator via ADB** (secondary). The agent captures game frames, detects game state, injects touch actions, and trains with PPO.

**Key architectural constraint**: The gymnasium env requires a live game running on Mac — Google Colab cannot connect to it. Solution: **decoupled actor-learner**. Mac collects rollout experience with the current policy and saves it to disk; Colab loads those rollouts, runs the PPO gradient update, and saves an updated checkpoint. The cycle repeats.

---

## Architecture

```
game-agent/
├── src/
│   ├── capture/
│   │   ├── mac_capture.py         # Quartz window find + mss crop
│   │   └── adb_capture.py         # adbutils device.screenshot()
│   ├── actions/
│   │   ├── mac_actions.py         # pyautogui tap/drag → iPhone Mirroring coords
│   │   └── adb_actions.py         # adb shell input tap/swipe
│   ├── games/
│   │   └── clash_royale/
│   │       ├── detector.py        # CV: elixir, card hand, tower HP, screen state
│   │       └── reward.py          # multi-component reward (PBRS + destroy + elixir waste)
│   ├── env/
│   │   ├── game_env.py            # gymnasium.Env (frame-stacked, 84×84×12) — needs live game
│   │   └── mock_game_env.py       # gymnasium.Env with synthetic obs/rewards — no game needed
│   ├── agent/
│   │   ├── model.py               # CnnPolicy feature extractor override
│   │   ├── collect.py             # local: run env → serialize RolloutBuffer → save .npz
│   │   ├── train_colab.py         # Colab: load .npz → PPO gradient update → save .zip
│   │   └── evaluate.py            # run N episodes, log win rate / crowns / elixir efficiency
│   └── main.py                    # CLI: --platform mac|adb --mode collect|play|eval
├── config/
│   └── clash_royale.yaml          # grid size, UI regions, timing constants, deck list
├── notebooks/
│   ├── train.ipynb                # Colab: load rollouts → train → download model
│   └── mock_train_test.ipynb      # Colab sanity check: train on mock env, no game needed
├── tests/
│   ├── fixtures/                  # static .png screenshots for offline detector tests
│   ├── test_capture.py            # mock screen → correct shape/dtype
│   ├── test_detector.py           # fixture frames → correct state/HP/elixir extraction
│   ├── test_reward.py             # game state dicts → correct reward values
│   ├── test_env_mock.py           # mock env: obs/action/reward shapes, step/reset contract
│   └── test_rollout.py            # collect.py output → correct .npz keys/shapes
├── requirements-mac.txt           # pyobjc, mss, pyautogui, adbutils
├── requirements-train.txt         # torch, stable-baselines3, gymnasium, opencv
└── README.md
```

---

## Actor-Learner Training Loop

```
┌──────────────────────────────────┐        ┌─────────────────────────────┐
│          Mac (Actor)             │        │      Google Colab (Learner)  │
│                                  │        │                              │
│  1. load checkpoint (best.zip)   │        │  4. load rollouts.npz        │
│  2. run game_env for N steps     │ ──────>│  5. PPO gradient update      │
│  3. save rollouts.npz +          │        │  6. save updated best.zip    │
│     (obs, act, rew, done,        │ <──────│                              │
│      logp, value)                │        │                              │
└──────────────────────────────────┘        └─────────────────────────────┘
         cycle repeats each iteration (e.g. 2048 steps per round)
```

**Implementation**: SB3's `PPO.collect_rollouts()` fills a `RolloutBuffer` of numpy arrays. `collect.py` calls it, then serializes the buffer to `.npz`. `train_colab.py` rebuilds a PPO model (same architecture, no env needed), injects the numpy arrays into the rollout buffer, and calls `model.train()` (the pure gradient step). The updated `.zip` checkpoint is downloaded and reloaded for the next collection round.

File transfer: Google Drive (manual upload/download) or `gdown`/`google.colab.files` in the notebook.

---

## Key Design Decisions

### Observation space
- **84 × 84 × 12**: 4-frame stack × 3 RGB channels — captures troop motion between frames
- `gymnasium.spaces.Box(0, 255, shape=(84, 84, 12), dtype=np.uint8)`

### Action space — concretely defined
Clash Royale: pick a card + drop location.
```
MultiDiscrete([5, 70])
  [0]: card choice — 0=no-op, 1–4 = card slots
  [1]: grid cell  — 0–69, flattened 10×7 arena grid (ignored when card=0)
```
No-op is essential — the agent must be able to wait for elixir to regenerate.

### Reward function — research-backed multi-component design

Informed by: [arxiv 2504.04783](https://arxiv.org/abs/2504.04783v1) (Clash Royale, 2025), [arxiv 2010.03956](https://arxiv.org/abs/2010.03956) (Action Guidance, RTS), MOBA reward decomposition from King of Glory research.

```
r_total = r_pbrs + r_destroy + r_king_activate + r_elixir_waste + r_survival
```

**Component breakdown:**

| Component | Signal | Formula | Notes |
|---|---|---|---|
| `r_pbrs` | Potential-based HP shaping | `γ·Φ(s') − Φ(s)` | Dense, policy-invariant. `Φ(s) = enemy_tower_hp_lost − our_tower_hp_lost` (normalized 0–1). Fires every step as tower HPs change. |
| `r_destroy` | Tower destruction | aux: `+1.0`, king: `+3.0` | King tower worth 3× — matches how it ends the game. From 2504.04783. |
| `r_king_activate` | Opponent king tower becomes active | `+0.5` | Destroying an auxiliary forces king to activate (enter range). Tactical milestone reward. |
| `r_elixir_waste` | Elixir overflow penalty | `−0.1 × max(0, elixir − 9)` | Penalizes letting elixir cap at 10 (wasted economy). From 2504.04783's `r_elixir`. |
| `r_survival` | Timestep alive | `+0.01` | Keeps gradients flowing early in training; prevents instant-concede policies. |

**Win/loss applied at episode end only:**
```python
WIN_BONUS    = +5.0
LOSS_PENALTY = -2.0
```
Win/loss is intentionally sparse — the dense PBRS signal does the heavy lifting. Keeping the win bonus moderate prevents the agent from ignoring all intermediate signals.

**Logarithmic crown scaling (alternative to linear):**
```python
def crown_reward(crowns_won, crowns_lost):
    f = lambda c: 4.9 * math.log(4.8 * c + 0.75) + 1.4
    return f(crowns_won) - f(crowns_lost)
```
Produces ≈ [−15, +15] range. The log shape rewards the first crown more than the third. From [Jaso1024's implementation](https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale).

**Action Guidance curriculum (reward annealing):**

From [arxiv 2010.03956](https://arxiv.org/abs/2010.03956): shaped rewards can cause the agent to optimize the shaping signal instead of winning. Fix with a curriculum:

```
Training phase 1 (early):  r = r_total (full shaping)
Training phase 2 (mid):    r = 0.5·r_total + 0.5·r_win_loss_only
Training phase 3 (late):   r = r_win_loss_only  (pure sparse)
```

Triggered by iteration count (e.g., switch phases every 200 rollout collection rounds).

### Deck selection — fixed deck for v1

Clash Royale requires a deck of 8 cards chosen *before* a match starts. During a match, 4 cards are visible at a time and cycle as they are played.

**v1 approach:** fix one deck for all training. Multi-deck generalization is v2+.

**Training deck selection criteria:**
- Low average elixir cost (~2.6–3.0) → more plays per match → denser training signal
- Visually distinct card icons → reliable template matching in `detector.py`
- Simple mechanics → fewer interaction edge cases

**Recommended starter deck:** Knight · Archers · Fireball · Log · Minions · Mega Minion · Ice Golem · Ice Spirit (avg elixir ≈ 2.6)

The deck is declared in `config/clash_royale.yaml`. The detector loads the 8 corresponding icon templates and matches against the hand strip each step.

### Step timing
- Target **4–5 FPS** (250ms/step): matches Clash Royale action resolution tempo
- Frame skip: hold each action for 2 rendered frames (~500ms) before observing

### iPhone Mirroring capture risk
macOS may copy-protect the iPhone Mirroring window.
- Try `mss` first; if frame is black, fall back to `CGWindowListCreateImage`
- If both fail: disable "Prevent screen recording" in iPhone Settings → Privacy

### Game state machine for `reset()`
`detector.py` classifies four states:
```
MAIN_MENU → BATTLE_LOADING → IN_BATTLE → END_SCREEN → MAIN_MENU
```
`reset()`: detect END_SCREEN → tap OK → wait MAIN_MENU → tap Battle → wait IN_BATTLE → return obs

### Split requirements
```
# requirements-mac.txt  (install locally)
pyobjc-framework-Quartz>=10.0, mss>=9.0, pyautogui>=0.9.54, adbutils>=2.8, pillow>=10.0

# requirements-train.txt  (install on Colab)
torch>=2.3, stable-baselines3>=2.3, gymnasium>=0.29, opencv-python>=4.9, numpy>=1.26
```

---

## Implementation Order

```
1. Scaffold (dirs, config yaml, requirements files)
        ↓
2. Capture layer — verify frame shape locally
        ↓
3. Detector (screen state + CV features) — unit-testable with fixture screenshots
        ↓
4. Action layer — manual smoke test (tap fires in game)
        ↓
5. Gym env (game_env.py) — random policy run, confirm obs/reward shapes
        ↓
6. collect.py — run N steps, save rollouts.npz, inspect buffer contents
        ↓
7. train_colab.py + train.ipynb — load buffer, PPO update on Colab T4, download .zip
        ↓
8. Close the loop — load updated .zip back into collect.py, repeat
```

---

## Testing Strategy

### Tests that require NO live game (run anywhere — Mac, Colab, CI)

| Test | Command | What it checks |
|---|---|---|
| Unit: detector | `pytest tests/test_detector.py` | CV functions on fixture `.png` frames → correct HP/elixir/state |
| Unit: reward | `pytest tests/test_reward.py` | Reward math on synthetic game-state dicts |
| Unit: mock env | `pytest tests/test_env_mock.py` | `MockGameEnv` obeys gymnasium contract (obs shape, reset, done) |
| Unit: rollout format | `pytest tests/test_rollout.py` | `collect.py` `.npz` has correct keys/dtypes for SB3 |
| Colab pipeline | `notebooks/mock_train_test.ipynb` | Full end-to-end on Colab without any game. |

### `MockGameEnv` design (`src/env/mock_game_env.py`)
- Identical `observation_space` and `action_space` as `GameEnv`
- `step()`: returns random 84×84×12 uint8 obs, simulated reward, random `done` after 50–200 steps
- Purpose: test the SB3 training loop, `collect.py`, `train_colab.py`, and `evaluate.py` without any game

### Tests that require a live game (run on Mac with game open)

| Test | Command | What it checks |
|---|---|---|
| Capture | `python -c "from src.capture.mac_capture import capture_frame; f=capture_frame(); print(f.shape)"` | Frame shape = `(84,84,3)`, not black |
| Env smoke | `python main.py --mode collect --steps 20 --random` | Full loop: capture→detect→reward→action |
| Full collect | `python main.py --mode collect --steps 2048` | Saves `rollouts.npz` ready for Colab upload |

### Model evaluation (`evaluate.py`)

```bash
python main.py --mode eval --model checkpoints/best.zip --episodes 20
```

Reports per episode: win/loss/draw, crowns scored/conceded, avg elixir spent, episode length.
Summary: win rate %, mean crowns ± std.

---

## Scope

- **v1**: Clash Royale, Mac platform, actor-learner with Colab
- **v2**: ADB platform parity, Brawl Stars env
- **Out of scope**: multi-agent, self-play, distributed training

---

## Future Work: YouTube Pre-training (Imitation Learning from Video)

Training from YouTube gameplay videos is a **viable bootstrapping method** — primarily as a pre-training step before live RL, not a full replacement.

### Why it helps

RL from scratch requires thousands of games before the agent plays a legal card. An agent pre-trained on expert video already knows the basic structure of play, dramatically accelerating live RL.

### Approach: Behavioral Cloning from Observation (BCO)

Video doesn't include action labels. BCO ([Torabi et al., 2018](https://arxiv.org/abs/1805.01954)) handles this:

```
Phase 1 — Train an Inverse Dynamics Model (IDM):
  Collect (state_t, state_{t+1}) pairs from live env interaction
  Train IDM: (frame_t, frame_{t+1}) → predicted action

Phase 2 — Label and clone:
  Extract (frame_t, frame_{t+1}) pairs from YouTube videos
  Use IDM to predict the action that caused the transition
  Run behavioral cloning: supervised loss on (frame_t, predicted_action)
```

**LAPO** ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/27985d21f0b751b933d675930aa25022-Paper-Conference.pdf)) learns latent actions from video directly, requiring only ~200 labeled transitions to decode to real actions.

### Feasibility notes

- Card identification from video is possible: card disappearing from hand + troop appearing at a location → infer card played and approximate grid position via CV
- YouTube high-elo gameplay videos are abundant; sample at 5 FPS
- Inferred actions are approximate (±1 grid cell error) — acceptable for pre-training

### Implementation sketch (future)

```
scripts/
  youtube_scraper.py        # yt-dlp wrapper
  extract_demonstrations.py # CV pipeline: frame pairs → (state, inferred_action) .npz
  train_bc.py               # behavioral cloning on demonstration dataset
  train_idm.py              # inverse dynamics model training
notebooks/
  bc_pretrain.ipynb         # Colab: load demonstrations → BC → export pretrained .zip
```

The pretrained `.zip` becomes the starting checkpoint for live PPO training, replacing random initialization.
