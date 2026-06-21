# Imitation Pre-training — Complete Reproduction Guide

This document explains **exactly** how to train the two models used to
bootstrap the Clash Royale agent before reinforcement learning:

1. an **Inverse Dynamics Model (IDM)**, and
2. a **behavior-cloned PPO policy** produced from it.

It is written so that someone with **no prior knowledge of this repo** can
reproduce the result, and so that the same recipe can be re-applied to a
different game. Everything here is exact: real commands, real file shapes,
real hyperparameters. Where a number is a tunable choice rather than a hard
requirement, it is labeled as such.

If you only want the short personal checklist, see [`IMITATION.md`](IMITATION.md).
For the underlying RL loop this feeds into, see [`TRAINING.md`](TRAINING.md).
For the architecture rationale, see [`PLAN.md`](PLAN.md).

---

## 1. What you are building and why

The agent observes the screen and outputs an action
`MultiDiscrete([5, 70])` = `(card_slot, target_cell)`:

- `card_slot`: `0` = **no-op** (play nothing), `1..4` = play hand slot 1-4.
- `target_cell`: one of 70 arena grid cells (a 7-wide x 10-tall placement grid).

Training this from scratch with RL alone is slow: a random policy almost
never plays a card at a sensible place, so early reward signal is sparse.
**Behavioral Cloning from Observation (BCO)** (Torabi et al., 2018) gives the
policy a warm start by imitating human gameplay from YouTube — *without*
needing the humans' actual button presses, which videos do not contain.

The trick is two-stage:

```
                    live rollouts (frames + REAL action labels)
                                     |
                                     v
                        +------------------------+
                        |  Inverse Dynamics Model |   learns:
                        |  (obs_t, obs_t+1) -> a  |   "what action caused
                        +------------------------+    this frame change?"
                                     |
       YouTube frames                | used to LABEL
   (obs_t, obs_t+1) pairs,  -------->|  action-free pairs
   NO action labels                  v
                        +------------------------+
                        |   Behavioral Cloning    |   supervised training of
                        |   labeled pairs -> PPO  |   the SAME PPO policy used
                        +------------------------+    for live RL
                                     |
                                     v
                        checkpoints/bc_pretrained.zip
                        (drop-in replacement for a fresh policy)
```

**Why an IDM at all?** YouTube videos give you *what the screen looked like*
but never *what the player tapped*. The IDM is the bridge: it is trained on
your own live rollouts (where you DO know the action, because the agent chose
it) to infer the action from a before/after frame pair, then applied to the
unlabeled YouTube pairs.

---

## 2. The data contracts (memorize these shapes)

Everything is 84x84 RGB, stacked 4 deep (12 channels), `uint8`, channels-last.

| Artifact | File | Key arrays | Shape / dtype |
|---|---|---|---|
| Live rollout | `rollouts/*.npz` | `observations` | `(n, 84, 84, 12)` uint8 |
| | | `actions` | `(n, 2)` int64 = `[card, cell]` |
| | | `episode_starts` | `(n,)` float32, `1.0` at episode start |
| | | (+`rewards`,`values`,`log_probs`,`last_value`,`last_done`) | see `src/agent/rollout_io.py` |
| Demonstration pairs | `data/demos/pairs.npz` | `observations` | `(n, 84, 84, 12)` uint8 |
| | | `next_observations` | `(n, 84, 84, 12)` uint8 |
| IDM weights | `checkpoints/idm.pt` | torch `state_dict` | — |
| Cloned policy | `checkpoints/bc_pretrained.zip` | SB3 PPO archive | — |

Two invariants that the code enforces and you must not break:

- **A pair never spans an episode or video boundary.** Within rollouts, the
  IDM excludes any pair where `episode_starts[i+1] == 1` (the action did not
  cause that frame change — a new episode began). Within video extraction,
  each video contributes only its own consecutive pairs.
- **`uint8` end to end.** The models divide by 255 internally. Do not
  pre-cast to float32 — it is 4x the RAM and host-to-device transfer.

---

## 3. Prerequisites

### 3.1 To collect live rollouts (Mac only)
You need a working capture+collection setup — this is the normal agent setup,
covered in [`README.md`](README.md) and [`TRAINING.md`](TRAINING.md). In short:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
.venv/bin/pip install -r requirements-train.txt   # torch + SB3, for local pipeline runs
```

Grant the terminal **Screen Recording** and **Accessibility** permissions
(System Settings -> Privacy & Security). Open Clash Royale in iPhone Mirroring.

### 3.2 To train the models (anywhere with a GPU)
Only `requirements-train.txt` (PyTorch, stable-baselines3, gymnasium,
numpy, opencv) plus `yt-dlp` for video download. No game, no Mac, no capture
permissions needed — the training stage is pure data crunching. This is why
it runs unchanged on Google Colab.

---

## 4. Model A — the Inverse Dynamics Model

### 4.1 Architecture (`src/agent/idm.py`)
A Nature-CNN over the **concatenated** pair `(obs_t, obs_t+1)` = 24 channels,
with two independent classification heads:

```
input  (B, 24, 84, 84) uint8
  -> /255.0
  -> Conv2d(24->32, k8, s4) -> ReLU
  -> Conv2d(32->64, k4, s2) -> ReLU
  -> Conv2d(64->64, k3, s1) -> ReLU -> Flatten
  -> Linear(->512) -> ReLU                 (shared trunk)
       |- card_head: Linear(512 -> 5)      (which slot, incl. no-op)
       \- cell_head: Linear(512 -> 70)     (which arena cell)
```

Loss = `CrossEntropy(card) + CrossEntropy(cell)`, summed with equal weight.

### 4.2 Training data
Live rollouts with real action labels. **Random-policy rollouts are fine** —
the IDM only needs to see varied (frame-change, action) pairs, not good play.
Collect on the Mac:

```bash
.venv/bin/python -m src.main --mode collect --steps 1024 --random \
    --output rollouts/rollouts_idm.npz
```

Recommended total: **2,000-4,000 steps** across one or more files. More and
more varied transitions = a better labeler. You can pass several `.npz` files;
the loader forces a boundary at each file start so pairs never span files.

### 4.3 Train it
Colab notebook cell, or locally:

```bash
.venv/bin/python -m scripts.train_idm rollouts/*.npz \
    --out checkpoints/idm.pt --epochs 10 --batch-size 64
```

Defaults (all in `train_idm`): `epochs=10`, `batch_size=64`, `lr=3e-4`,
`val_fraction=0.1`, optimizer Adam. A 10% held-out split is carved off
automatically (when there are >=20 pairs).

### 4.4 How to know it worked
The final line printed is the **held-out** accuracy:

```
IDM held-out (NN pairs): card acc XX.XX%, cell acc YY.YY%
```

- **Card accuracy** is the number that matters most. Chance is 20% (5 classes),
  but the stream is no-op-dominated, so an "always no-op" baseline can already
  look high — compare against that baseline, not against 20%. Beating it means
  the IDM learned the *play vs no-play* and *which-slot* signal.
- **Cell accuracy** will be much lower (70 classes, fine spatial detail from
  two tiny 84x84 frames). This is expected and acceptable — placement is the
  hardest thing to infer and BC only needs it approximately (+/-1 cell).

If card accuracy is at the always-no-op rate, the IDM did not learn: collect
more and more varied live rollouts and retrain.

---

## 5. Demonstration pairs from video

### 5.1 Choose videos
3-10 clips of **high-level ladder gameplay**, portrait game view, minimal
overlays. Strongly prefer videos that play the **same deck** as your config
(`config/clash_royale.yaml`) — the clone imitates *which slot* is played, so a
different deck teaches wrong slot->situation mappings. Avoid montages,
facecam-heavy edits, and 2v2.

### 5.2 Download
```bash
.venv/bin/python -m scripts.download_videos <URL> [<URL> ...] --out data/youtube
```
Caps at <=1080p mp4 (frames become 84x84 anyway).

### 5.3 Find the crop, then extract
Videos are 16:9 but the game is portrait, so there is pillarboxing / facecam /
chat to trim. `--crop L,T,R,B` are fractions trimmed from each edge, each in
`[0, 0.5)`. **Always preview first:**

```bash
# 1) dump one cropped frame per video as <video>_crop_preview.png
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 \
    --crop 0.30,0.0,0.30,0.0 --preview-frame

# 2) open the PNGs, adjust until ONLY the arena is visible, then extract
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 \
    --crop 0.30,0.0,0.30,0.0 --fps 4 --out data/demos/pairs.npz
```

Sampling is **4 FPS** to match the env step rate; frames are cropped -> RGB ->
84x84 -> 4-frame stacks -> consecutive `(obs_t, obs_t+1)` pairs. One global crop
applies to all videos in a run, so group videos with the same layout, or run
the extractor once per layout and concatenate.

---

## 6. Model B — the behavior-cloned policy

### 6.1 Label the video pairs with the IDM
`scripts/train_bc.py` loads `idm.pt`, runs `label_pairs` (argmax of each head)
over `pairs.npz`, and prints the play/no-op breakdown:

```
Labeled NNNN pairs: PPP card plays, QQQ no-ops
```

**Sanity target:** at 4 FPS a real player plays roughly one card per 3-10
seconds, so **plays should be ~2-8% of pairs**. If almost everything is no-op
(or almost everything is a play), the IDM is too weak — go back to section 4.2.

### 6.2 No-op downsampling
Most frames have no play, so cloning them all collapses the policy to "always
wait." `train_bc` keeps **every** play and only `--keep-noop-fraction` (default
**0.25**) of no-op examples. Tune this if the play ratio is far from healthy.

### 6.3 Clone into the PPO policy
The key design choice: BC trains the **exact same** SB3 PPO policy used for
live RL (built via `make_ppo`, with `MockGameEnv` only as a spaces donor), so
the output `.zip` is a drop-in `checkpoints/best.zip`. Training minimizes
negative log-likelihood of the demonstrated action under the policy:

```python
loss = -policy.get_distribution(obs).log_prob(action).mean()
```

Run it:

```bash
.venv/bin/python -m scripts.train_bc data/demos/pairs.npz \
    --idm checkpoints/idm.pt \
    --out checkpoints/bc_pretrained.zip \
    --epochs 5 --config config/clash_royale.yaml
```

Defaults: `epochs=5`, `batch_size=64`, `lr=3e-4`, Adam. Watch the NLL fall:

```
BC epoch 1/5: nll X.XXXX
...
BC epoch 5/5: nll lower
```

> **Note:** BC only trains the **policy (action) head**. PPO's value head
> stays randomly initialized; the first few live PPO rounds repair it. This is
> expected, not a bug.

---

## 7. Install and verify

```bash
# install the clone as the RL starting checkpoint
cp checkpoints/bc_pretrained.zip checkpoints/best.zip

# evaluate vs random — expect "plays at sensible times", not wins
.venv/bin/python -m src.main --mode eval --model checkpoints/best.zip --episodes 5
```

A good BC policy bootstraps *structure* (it plays cards when it has elixir,
roughly where humans do). PPO does the rest. Then resume the normal loop in
[`TRAINING.md`](TRAINING.md) from round 0 with this checkpoint in place.

---

## 8. End-to-end recipe (copy/paste)

### On Colab (recommended — GPU, nothing local)
Open `notebooks/bc_pretrain.ipynb`, set `BRANCH`, run cells top to bottom:
upload rollout `.npz`, train IDM, set `VIDEO_URLS` + `CROP` (preview!), extract,
clone, download `bc_pretrained.zip`.

### Locally (Mac/Linux with a GPU or patience)
```bash
.venv/bin/pip install -r requirements-train.txt yt-dlp

# 1. (Mac) collect IDM data
.venv/bin/python -m src.main --mode collect --steps 1024 --random \
    --output rollouts/rollouts_idm.npz

# 2. train the IDM
.venv/bin/python -m scripts.train_idm rollouts/*.npz --out checkpoints/idm.pt --epochs 10

# 3. videos -> pairs (preview the crop first!)
.venv/bin/python -m scripts.download_videos <URL> ... --out data/youtube
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 --crop <L,T,R,B> --preview-frame
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 --crop <L,T,R,B> --fps 4 --out data/demos/pairs.npz

# 4. label + clone
.venv/bin/python -m scripts.train_bc data/demos/pairs.npz --idm checkpoints/idm.pt \
    --out checkpoints/bc_pretrained.zip --epochs 5

# 5. install + eval
cp checkpoints/bc_pretrained.zip checkpoints/best.zip
.venv/bin/python -m src.main --mode eval --model checkpoints/best.zip --episodes 5
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| IDM card acc ~ always-no-op rate | too little / unvaried live data | collect 2-4k+ steps, varied; retrain |
| Labeled plays ~ 0% or ~ 100% | weak IDM | more live rollouts (4.2), retrain IDM |
| Crop preview shows facecam/chat/bars | wrong `--crop` | adjust L,T,R,B; re-preview |
| `No pairs extracted` | video too short / wrong fps | longer clips; check `--fps` |
| BC policy does nothing | over-downsampled plays | raise `--keep-noop-fraction` |
| BC policy spams cards | too few no-ops kept / deck mismatch | lower fraction; use deck-matched videos |
| Black frames in extraction | copy-protected source | use a different video |

---

## 10. Adapting this to another game (the general recipe)

The pipeline is game-agnostic; to recreate it for a different game you need
exactly five things to line up:

1. **A discrete action space** with a known no-op (here `card==0`). Update the
   IDM head sizes (`N_CARDS`, `N_CELLS` in `src/agent/idm.py`) and the
   `played = actions[:,0] > 0` rule in `src/agent/bc.py`.
2. **An observation pipeline** that turns both live capture and video frames
   into the *same* tensor (here 84x84x12 via `src/capture/preprocess.py`).
   This shared representation is what lets the IDM transfer from your rollouts
   to YouTube frames.
3. **Live rollouts with real action labels** to train the IDM (any policy,
   even random).
4. **Action-free demonstration video** of competent play, sampled at the env
   step rate, cropped to the same view as your live capture.
5. **One policy network** reused for both BC and RL, so the cloned weights
   load directly. Build BC and RL through the same factory (`make_ppo`).

Get those five aligned and the four scripts
(`train_idm` -> `extract_demonstrations` -> `train_bc`) reproduce this result
for any game.

---

## 11. Known limitations (accepted for v1)

- IDM labels are approximate (especially the 70-way cell head); good enough to
  bootstrap, not for evaluation or as ground truth.
- BCO has a built-in distribution-shift weakness: the IDM is trained on your
  (often random) rollouts but labels expert transitions it never saw. If labels
  look poor, the principled upgrade is *iterative* BCO — collect new rollouts
  with the improving policy and retrain the IDM.
- Video color grading / scale differs from iPhone Mirroring capture; the 84x84
  downscale absorbs most of it, but expect a few live PPO rounds to adapt.
- One global crop per extraction run; different layouts need separate runs.
