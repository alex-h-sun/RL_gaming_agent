# Training Guide

How to train the Clash Royale agent end to end. Training is split across two
machines because the live game only exists on the Mac and the GPU only exists
on Colab:

- **Mac (actor)** — runs the game via iPhone Mirroring, steps the environment
  with the current policy, and saves the experience to `rollouts/rollouts.npz`.
- **Colab (learner)** — loads that file, runs the PPO gradient update on GPU,
  and saves an updated `checkpoints/best.zip`.

One round trip of (collect on Mac) -> (update on Colab) -> (download new
checkpoint) is a **training round**. You repeat rounds until the win rate
stops improving.

```
        Mac (actor)                          Colab (learner)
  ┌──────────────────────────┐         ┌──────────────────────────┐
  │ load checkpoints/best.zip│         │ load rollouts.npz        │
  │ play 2048 steps          │ ──────> │ PPO gradient update      │
  │ save rollouts.npz        │         │ save updated best.zip    │
  └──────────────────────────┘ <────── └──────────────────────────┘
                       repeat each round
```

## 0. One-time setup

### Mac

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
.venv/bin/pip install -r requirements-train.txt   # torch/SB3, needed to run the policy locally
```

- System Settings -> Privacy & Security: grant your terminal **Screen
  Recording** and **Accessibility**.
- Open **iPhone Mirroring**, launch Clash Royale, sit on the main menu.
- On the iPhone, set the training deck from `config/clash_royale.yaml`
  (Knight, Archers, Mini Pekka, Giant, Minions, Fireball, Arrows,
  Musketeer) as the active deck.

### Verify capture works

```bash
.venv/bin/python -c "from src.capture.mac_capture import capture_frame; f = capture_frame(); print(f.shape, f.mean())"
```

Expected: `(84, 84, 3)` and a mean well above 0. A black frame (mean ~0)
means the window is copy-protected — disable screen-recording protection on
the iPhone and retry.

### Calibrate UI regions (do this once, carefully)

The detector reads elixir and tower HP from normalized regions in
`config/clash_royale.yaml`. They are estimates and must be tuned to your
window before any real training:

```bash
.venv/bin/python -m scripts.calibrate
```

Run it during a battle. It saves `calibration_overlay.png` with every
configured region drawn on the captured frame (green boxes = detection
regions, red circles = tap points) and prints the detected state. If the
boxes don't sit on the real elixir bar / HP bars, adjust the `ui_regions`
fractions in the YAML and re-run until the printed state matches the
screen. Also verify the `battle_button` and `ok_button` tap points from the
menu and end screens. For Brawl Stars, repeat with
`--config config/brawl_stars.yaml`.

Capture grabs the window by ID, so other windows overlapping the game do
not corrupt frames. The window must still be on screen and unminimized —
and during collection it must also be unobstructed and frontmost, because
mouse clicks (unlike capture) always go to whatever is on top.

### Sanity-check the whole pipeline without the game

```bash
# all offline unit tests, no game needed
.venv/bin/python -m pytest tests/

# full collect pipeline against the mock env
.venv/bin/python -m src.main --mode collect --mock --steps 64 --random
```

Or run `notebooks/mock_train_test.ipynb` on Colab — it exercises the full
collect -> train -> reload cycle with the mock env.

## 1. Round 0 — bootstrap with a random policy

There is no checkpoint yet, so collect the first batch with random actions:

```bash
# quick smoke test first: 20 steps, watch the game respond to taps
.venv/bin/python -m src.main --mode collect --steps 20 --random

# real first batch (~10-15 min of play at 4 FPS)
.venv/bin/python -m src.main --mode collect --steps 2048 --random --round 0
```

This produces `rollouts/rollouts.npz`. Keep the game window visible and the
Mac awake for the whole run (`caffeinate -d` helps).

## 2. Learner step on Colab

1. Open `notebooks/train.ipynb` in Colab, set runtime to **GPU** (T4 is fine).
2. Fix the repo URL in the first cell if you haven't already.
3. Run all cells: it installs `requirements-train.txt`, prompts you to upload
   `rollouts.npz` (and `best.zip` from round 1 onward), runs the PPO update,
   and downloads the updated `checkpoints/best.zip`.

Equivalently, anywhere with torch:

```bash
python -m src.agent.train_colab \
    --rollouts rollouts/rollouts.npz \
    --checkpoint checkpoints/best.zip \
    --output checkpoints/best.zip
```

On the first round there is no checkpoint yet — the trainer initializes a
fresh model automatically.

## 3. Rounds 1..N — the loop

Place the downloaded `best.zip` in `checkpoints/`, then:

```bash
.venv/bin/python -m src.main --mode collect --steps 2048 --round <k>
```

then repeat step 2, incrementing `--round` each time. **`--round` matters**:
it drives the reward curriculum (`curriculum.phase_rounds: 200` in the YAML):

| Rounds    | Reward signal                          |
|-----------|----------------------------------------|
| 0–199     | full dense shaping (PBRS, destroys, elixir) |
| 200–399   | 50/50 blend of shaping and win/loss    |
| 400+      | pure sparse win/loss                   |

A suggested shell loop for the Mac side (you still do the Colab step between
iterations, or swap in Google Drive sync to automate the transfer):

```bash
k=1   # current round
.venv/bin/python -m src.main --mode collect --steps 2048 --round $k \
    --output rollouts/rollouts_r$k.npz
# upload rollouts_r$k.npz, run train.ipynb, download best.zip, k=$((k+1))
```

## 4. Evaluate

Every ~10 rounds, measure the policy against the live game:

```bash
.venv/bin/python -m src.main --mode eval --model checkpoints/best.zip --episodes 20
```

Reports per-episode win/loss/draw, crowns, reward, and length, plus a
summary (win rate, mean crowns +/- std). Keep the best-evaluating checkpoint
around — `best.zip` is whatever came out of the last update, not necessarily
the best so far.

To just watch it play:

```bash
.venv/bin/python -m src.main --mode play --model checkpoints/best.zip
```

## What to expect

- Random policy baseline wins essentially never against humans; use Trainer
  battles (vs AI) early if possible for denser wins.
- The first ~50 rounds mostly teach "play cards at all, don't waste elixir"
  via the dense shaping terms; win rate is a lagging indicator.
- 2048 steps/round at 4 FPS is roughly 3-4 matches, ~10-15 minutes of
  collection per round. Hundreds of rounds are expected — this is the cost of
  on-policy PPO with one environment; treat v1 as proving the loop works.

## Hyperparameters

All in `config/clash_royale.yaml` under `ppo:` (learning rate, batch size,
epochs, clip range, entropy). Both the Mac and Colab sides build the model
through `src/agent/model.py::make_ppo`, so editing the YAML keeps the two in
sync. Keep `n_steps` equal to the `--steps` you collect with.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Black captured frames | Disable screen-recording protection on iPhone; check Screen Recording permission |
| `Window 'iPhone Mirroring' not found` | Window must be open, unminimized, on the active Space |
| Taps don't register in game | Accessibility permission missing; window moved (rect is cached — restart the run) |
| `reset()` times out | `battle_button` / `ok_button` coords wrong, or detector misclassifies the menu — recalibrate |
| Elixir/HP readings wrong | Recalibrate `ui_regions` (section 0) |
| Colab: shape mismatch on load | Mac and Colab built different models — make sure both sides are on the same commit and YAML |
