# Imitation Learning (BCO) — User Workflow

Pre-train the policy from YouTube gameplay before live RL, using Behavioral
Cloning from Observation (Torabi et al., 2018 — see PLAN.md "Future Work").

How it works:

1. An Inverse Dynamics Model (IDM) learns `(frame_t, frame_t+1) -> action`
   from your live rollouts, which carry real action labels.
2. Frames sampled from YouTube videos become action-free `(obs_t, obs_t+1)`
   pairs; the IDM labels them with inferred actions.
3. Behavioral cloning trains the standard PPO policy on the labeled pairs and
   exports a normal SB3 checkpoint that replaces random initialization.

Everything heavy runs on Colab (`notebooks/bc_pretrain.ipynb`); the scripts
also run locally if you prefer.

> For a complete, anyone-can-follow reproduction guide (architecture, exact
> data shapes, hyperparameters, verification, and how to adapt this to another
> game), see [`IMITATION_GUIDE.md`](IMITATION_GUIDE.md).

## Your TODO checklist

- [ ] **1. Collect IDM training data on the Mac.** The IDM needs live
      rollouts with action labels. Your existing `rollouts/rollouts.npz`
      works; more is better (2-4k steps total recommended). Random-policy
      rollouts are fine:

      .venv/bin/python -m src.main --mode collect --steps 1024 --random \
          --output rollouts/rollouts_idm.npz

- [ ] **2. Pick 3-10 YouTube videos.** High-level ladder gameplay, portrait
      game view, minimal overlays. Strongly prefer videos that use **your
      training deck** (giant push deck from `config/clash_royale.yaml`) —
      the cloned policy imitates which slots get played, so deck mismatch
      teaches wrong card-to-situation mappings. Avoid montages, facecam-heavy
      edits, and 2v2.

- [ ] **3. Push the current branch** so Colab can clone it:

      git push -u origin worktree-imitation-learning

      (Or merge to main and push; the notebook checks out `main` by default —
      set `BRANCH = 'worktree-imitation-learning'` in the first cell if
      unmerged.)

- [ ] **4. Run `notebooks/bc_pretrain.ipynb` on Colab** (GPU runtime):
      upload your rollout `.npz` file(s), paste the video URLs, and —
      important — iterate on the `CROP` value using the preview cell until
      the cropped frame shows only the game arena (no pillarboxing, no
      facecam, no chat overlay). Then run the remaining cells.

- [ ] **5. Sanity-check the IDM and labeling printouts.** The `train_idm`
      step ends with a held-out accuracy line — card accuracy well above
      chance (20%) means the IDM learned something. The `train_bc` step
      prints "N card plays, M no-ops". A healthy ratio is roughly 1 play per
      3-10 seconds of gameplay (at 4 FPS: plays should be ~2-8% of pairs).
      If nearly everything is labeled no-op or everything is a play, the IDM
      is weak — collect more live rollouts (step 1) and retrain it.

- [ ] **6. Download `bc_pretrained.zip`** from the last cell and install it
      as the starting checkpoint on the Mac:

      cp ~/Downloads/bc_pretrained.zip checkpoints/best.zip

- [ ] **7. Evaluate before trusting it.** Compare a few live episodes of the
      BC policy against random:

      .venv/bin/python -m src.main --mode eval --model checkpoints/best.zip --episodes 5

      Expect "plays cards at sensible times" rather than wins — BC bootstraps
      structure, PPO does the rest.

- [ ] **8. Resume the normal RL loop** (`TRAINING.md`) from round 0 with the
      BC checkpoint in place.

## Local (Mac) alternative to the Colab cells

```bash
# install the extra deps
.venv/bin/pip install yt-dlp torch stable-baselines3 opencv-python

# download videos
.venv/bin/python -m scripts.download_videos <URL> [<URL> ...]

# preview the crop, then extract pairs
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 \
    --crop 0.0,0.0,0.0,0.0 --preview-frame
.venv/bin/python -m scripts.extract_demonstrations data/youtube/*.mp4 \
    --crop <L,T,R,B> --fps 4 --out data/demos/pairs.npz

# train the IDM on live rollouts, then clone
.venv/bin/python -m scripts.train_idm rollouts/*.npz --out checkpoints/idm.pt
.venv/bin/python -m scripts.train_bc data/demos/pairs.npz \
    --idm checkpoints/idm.pt --out checkpoints/bc_pretrained.zip
```

## Known limitations (accepted for v1 of BCO)

- IDM labels are approximate (PLAN.md estimates +/-1 grid cell); fine for
  pre-training, not for evaluation.
- Most video frames are no-ops; `train_bc` downsamples no-op labels to 25%
  (`--keep-noop-fraction`) so the cloned policy doesn't collapse to waiting.
- Card-slot semantics in videos differ from your deck's slot order unless the
  video uses the same deck — hence the deck-matching recommendation above.
- Video color grading / scaling differs from iPhone Mirroring capture; the
  84x84 downscale absorbs most of it, but expect the BC policy to need a few
  live PPO rounds to adapt.
