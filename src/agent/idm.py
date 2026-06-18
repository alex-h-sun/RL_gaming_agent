"""Inverse Dynamics Model for Behavioral Cloning from Observation (BCO).

The IDM predicts which action caused the transition (obs_t -> obs_{t+1}).
It is trained on live rollouts (which have real action labels) and then used
to label action-free demonstration pairs extracted from YouTube videos.

Action space mirrors the env: MultiDiscrete([5, 70]) -> two softmax heads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_CARDS = 5
N_CELLS = 70


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InverseDynamicsModel(nn.Module):
    """CNN over the concatenated (obs_t, obs_{t+1}) pair: 24 input channels."""

    def __init__(self, n_channels: int = 24, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_channels, 84, 84)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.card_head = nn.Linear(features_dim, N_CARDS)
        self.cell_head = nn.Linear(features_dim, N_CELLS)

    def forward(self, obs_pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_pair: (batch, 24, 84, 84) uint8 or float in [0, 255]."""
        features = self.trunk(self.cnn(obs_pair / 255.0))
        return self.card_head(features), self.cell_head(features)


def _pairs_to_tensor(obs: np.ndarray, next_obs: np.ndarray) -> torch.Tensor:
    """Stack channels-last uint8 pairs into a (n, 24, 84, 84) uint8 tensor.

    Kept uint8 end to end; the model divides by 255 internally, so this is
    4x cheaper in RAM and host-to-device transfer than float32.
    """
    pair = np.concatenate([obs, next_obs], axis=-1)  # (n, 84, 84, 24)
    return torch.from_numpy(np.ascontiguousarray(pair.transpose(0, 3, 1, 2)))


def valid_pair_indices(
    n_observations: int, episode_starts: np.ndarray | None = None
) -> np.ndarray:
    """Indices i for which the pair (obs[i], obs[i+1]) is a real transition.

    A pair is bogus when obs[i+1] begins a new episode (or a new rollout
    file): the labeled action did not cause that frame change.
    """
    indices = np.arange(n_observations - 1)
    if episode_starts is None:
        return indices
    return indices[episode_starts[1:n_observations] < 0.5]


class _PairDataset(torch.utils.data.Dataset):
    """Builds (obs_t, obs_{t+1}) tensors on demand instead of materializing
    the full 24-channel dataset up front (which is ~680KB/pair in float32)."""

    def __init__(
        self, observations: np.ndarray, actions: np.ndarray, indices: np.ndarray
    ):
        self._observations = observations
        self._actions = actions
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self._indices[item]
        pair = np.concatenate(
            [self._observations[i], self._observations[i + 1]], axis=-1
        )
        inputs = torch.from_numpy(np.ascontiguousarray(pair.transpose(2, 0, 1)))
        return inputs, torch.as_tensor(self._actions[i], dtype=torch.long)


@torch.no_grad()
def _evaluate_idm(
    model: InverseDynamicsModel,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[float, float]:
    """Return (card accuracy, cell accuracy) over a loader."""
    model.eval()
    correct_card, correct_cell, count = 0, 0, 0
    for batch_inputs, batch_labels in loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        card_logits, cell_logits = model(batch_inputs)
        correct_card += (card_logits.argmax(1) == batch_labels[:, 0]).sum().item()
        correct_cell += (cell_logits.argmax(1) == batch_labels[:, 1]).sum().item()
        count += len(batch_labels)
    model.train()
    return correct_card / count, correct_cell / count


def train_idm(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_starts: np.ndarray | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    device: str | None = None,
) -> InverseDynamicsModel:
    """Train an IDM on consecutive rollout observations with action labels.

    observations: (n, 84, 84, 12) uint8; actions: (n, 2) int64.
    Pairs are (obs[i], obs[i+1]) labeled with actions[i]; pairs that span an
    episode boundary (episode_starts[i+1] == 1) are excluded. A held-out
    split reports honest accuracy — the number to watch before trusting the
    IDM to label demonstrations.
    """
    device = device or pick_device()
    model = InverseDynamicsModel().to(device)
    indices = valid_pair_indices(len(observations), episode_starts)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    n_val = int(len(indices) * val_fraction) if len(indices) >= 20 else 0
    train_set = _PairDataset(observations, actions, indices[n_val:])
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = (
        torch.utils.data.DataLoader(
            _PairDataset(observations, actions, indices[:n_val]),
            batch_size=batch_size,
        )
        if n_val
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total, correct_card, count = 0.0, 0, 0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            card_logits, cell_logits = model(batch_inputs)
            loss = loss_fn(card_logits, batch_labels[:, 0]) + loss_fn(
                cell_logits, batch_labels[:, 1]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_labels)
            correct_card += (card_logits.argmax(1) == batch_labels[:, 0]).sum().item()
            count += len(batch_labels)
        print(
            f"IDM epoch {epoch + 1}/{epochs}: "
            f"loss {total / count:.4f}, card acc {correct_card / count:.2%}"
        )
    if val_loader is not None:
        card_acc, cell_acc = _evaluate_idm(model, val_loader, device)
        print(
            f"IDM held-out ({n_val} pairs): "
            f"card acc {card_acc:.2%}, cell acc {cell_acc:.2%}"
        )
    return model


@torch.no_grad()
def label_pairs(
    model: InverseDynamicsModel,
    observations: np.ndarray,
    next_observations: np.ndarray,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Predict actions (n, 2) int64 for action-free demonstration pairs."""
    device = device or next(model.parameters()).device
    model.eval()
    predictions = []
    for start in range(0, len(observations), batch_size):
        batch = _pairs_to_tensor(
            observations[start : start + batch_size],
            next_observations[start : start + batch_size],
        ).to(device)
        card_logits, cell_logits = model(batch)
        predictions.append(
            torch.stack([card_logits.argmax(1), cell_logits.argmax(1)], dim=1).cpu()
        )
    return torch.cat(predictions).numpy().astype(np.int64)


def save_idm(model: InverseDynamicsModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_idm(path: str | Path, device: str | None = None) -> InverseDynamicsModel:
    device = device or pick_device()
    model = InverseDynamicsModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
