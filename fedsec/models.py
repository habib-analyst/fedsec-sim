"""Small PyTorch models (CPU) and flat-parameter helpers.

Robust aggregators (Krum, trimmed mean, ...) operate on flattened update
vectors, so every helper here works with a single 1-D parameter tensor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer perceptron for binary classification (single logit output)."""

    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x).squeeze(-1)


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameters into one 1-D float tensor (detached)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat parameter vector back into the model in place."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset : offset + n].reshape(p.shape))
            offset += n


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    """Binary accuracy of the model on (x, y)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x))
        preds = (logits >= 0).long().numpy()
    return float((preds == y).mean())
