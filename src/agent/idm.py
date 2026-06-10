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
        """obs_pair: (batch, 24, 84, 84) float in [0, 255]."""
        features = self.trunk(self.cnn(obs_pair / 255.0))
        return self.card_head(features), self.cell_head(features)


def _pairs_to_tensor(obs: np.ndarray, next_obs: np.ndarray) -> torch.Tensor:
    """Stack channels-last uint8 pairs into a (n, 24, 84, 84) float tensor."""
    pair = np.concatenate([obs, next_obs], axis=-1)  # (n, 84, 84, 24)
    return torch.as_tensor(pair.transpose(0, 3, 1, 2), dtype=torch.float32)


def train_idm(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str | None = None,
) -> InverseDynamicsModel:
    """Train an IDM on consecutive rollout observations with action labels.

    observations: (n, 84, 84, 12) uint8; actions: (n, 2) int64.
    Pairs are (obs[i], obs[i+1]) labeled with actions[i].
    """
    device = device or pick_device()
    model = InverseDynamicsModel().to(device)
    inputs = _pairs_to_tensor(observations[:-1], observations[1:])
    labels = torch.as_tensor(actions[:-1], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
