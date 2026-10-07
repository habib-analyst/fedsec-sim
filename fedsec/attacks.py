"""Poisoning attacks for federated learning simulation.

All attacks act on the malicious client's side:

- :func:`label_flip_attack` corrupts the local training data (data poisoning).
- :func:`gaussian_poison_update` corrupts the model update the client sends
  (model poisoning: scale + Gaussian noise).
- :func:`backdoor_poison_data` plants a trigger pattern correlated with a
  target label into a fraction of the local training data.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import TRIGGER_FEATURES, TRIGGER_VALUE, apply_trigger


def label_flip_attack(y: np.ndarray) -> np.ndarray:
    """Flip every binary label: 0 -> 1, 1 -> 0."""
    return (1 - y).astype(y.dtype)


def gaussian_poison_update(
    delta: torch.Tensor,
    scale: float = 10.0,
    noise_std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Model-poisoning update: scale the honest update and add Gaussian noise.

    ``delta' = scale * delta + N(0, noise_std^2 I)``. A large positive ``scale``
    over-weights the malicious client's data; a *negative* ``scale`` flips the
    update direction (gradient ascent), actively unlearning the honest model.
    ``noise_std`` makes the poisoned updates harder to filter by clipping.
    """
    noise = torch.randn(
        delta.shape, generator=generator, dtype=delta.dtype
    ) * noise_std
    return scale * delta + noise


def backdoor_poison_data(
    x: np.ndarray,
    y: np.ndarray,
    poison_fraction: float = 0.3,
    target_label: int = 0,
    rng: np.random.Generator | None = None,
    trigger_features: tuple[int, ...] = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    """Plant a trigger pattern on a fraction of samples, relabel to target.

    The trigger (outlier feature values) becomes correlated with
    ``target_label`` in the poisoned client's data, so a global model that
    averages this client's update learns the backdoor.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_poison = int(len(x) * poison_fraction)
    idx = rng.choice(len(x), size=n_poison, replace=False)
    x = x.copy()
    x[idx] = apply_trigger(
        x[idx], trigger_features=trigger_features, trigger_value=trigger_value
    )
    y = y.copy()
    y[idx] = target_label
    return x, y
