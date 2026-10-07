"""Aggregation rules (defenses) for federated learning.

Every rule maps a list of client update vectors (1-D tensors, same shape) to
a single aggregated update vector.

- :func:`fedavg` — plain mean; no defense, used as the vulnerable baseline.
- :func:`trimmed_mean` — coordinate-wise: sort each coordinate, drop the
  ``beta`` fraction from both tails, average the rest.
- :func:`coordinate_median` — coordinate-wise median; breakdown point 50%.
- :func:`krum` — pick the update closest to its ``n - f - 2`` nearest
  neighbours (Blanchard et al., 2017). Returns the average of the best ``m``.
- :func:`dp_fedavg` — clip each update to ``clip_norm``, average, then add
  Gaussian noise with std ``noise_multiplier * clip_norm``. Reports the
  (clip_norm, noise_multiplier) pair used; no formal (eps, delta) accounting
  is attempted here (see README limitations).
"""

from __future__ import annotations

import torch

__all__ = [
    "fedavg",
    "trimmed_mean",
    "coordinate_median",
    "krum",
    "dp_fedavg",
    "aggregate",
]

AGGREGATORS = ("fedavg", "trimmedmean", "median", "krum", "dp_fedavg")


def _stack(updates: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([u.reshape(-1).float() for u in updates], dim=0)


def fedavg(updates: list[torch.Tensor]) -> torch.Tensor:
    """Plain coordinate-wise mean of updates."""
    return _stack(updates).mean(dim=0)


def trimmed_mean(updates: list[torch.Tensor], beta: float = 0.2) -> torch.Tensor:
    """Coordinate-wise trimmed mean.

    Drop the ``beta`` fraction of smallest and largest values per coordinate
    (at least zero, at most n//2 - 1 per side), then average the survivors.
    """
    s = _stack(updates)
    n = s.shape[0]
    k = min(int(beta * n), n // 2 - 1)
    k = max(k, 0)
    if k == 0:
        return s.mean(dim=0)
    s_sorted, _ = torch.sort(s, dim=0)
    return s_sorted[k : n - k].mean(dim=0)


def coordinate_median(updates: list[torch.Tensor]) -> torch.Tensor:
    """Coordinate-wise median of updates."""
    return _stack(updates).median(dim=0).values


def krum(updates: list[torch.Tensor], f: int = 1, m: int = 1) -> torch.Tensor:
    """Krum robust aggregation (Blanchard et al., NeurIPS 2017).

    For each update, score = sum of squared distances to its
    ``n - f - 2`` nearest neighbours. Returns the average of the ``m``
    updates with the lowest scores (``m = 1`` selects a single update, the
    classical Krum rule).
    """
    s = _stack(updates)
    n = s.shape[0]
    if n < 3:
        return s.mean(dim=0)
    keep = max(n - f - 2, 1)
    # Pairwise squared Euclidean distances.
    dists = torch.cdist(s, s, p=2).pow(2)
    # Exclude self-distance by sorting and skipping index 0.
    nearest, _ = torch.sort(dists, dim=1)
    scores = nearest[:, 1 : keep + 1].sum(dim=1)
    order = torch.argsort(scores)
    return s[order[:m]].mean(dim=0)


def dp_fedavg(
    updates: list[torch.Tensor],
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """DP-FedAvg-style aggregation: per-update clipping + Gaussian noise.

    Each update is clipped to L2 norm ``clip_norm``, the clipped updates are
    averaged, and Gaussian noise with std ``noise_multiplier * clip_norm`` is
    added. Returns ``(aggregated, report)`` where report carries the
    ``clip_norm`` and ``noise_multiplier`` for the run's accounting table.

    Note: this provides the *mechanism* only; converting (clip, sigma) into
    an (epsilon, delta) guarantee needs a moments/RDP accountant, which is
    out of scope for this simulator.
    """
    clipped = []
    for u in _stack(updates):
        norm = u.norm()
        factor = min(1.0, clip_norm / (norm.item() + 1e-12))
        clipped.append(u * factor)
    mean = torch.stack(clipped).mean(dim=0)
    noise_std = noise_multiplier * clip_norm
    noise = torch.randn(
        mean.shape, generator=generator, dtype=mean.dtype
    ) * noise_std
    report = {
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "noise_std": noise_std,
        "n_updates": len(updates),
    }
    return mean + noise, report


def aggregate(
    name: str,
    updates: list[torch.Tensor],
    seed: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Dispatch to an aggregation rule by name.

    Returns ``(aggregated_update, report)``; ``report`` is empty for
    non-DP rules.
    """
    name = name.lower()
    if name == "fedavg":
        return fedavg(updates), {}
    if name in ("trimmedmean", "trimmed_mean"):
        return trimmed_mean(updates, beta=kwargs.get("trim_ratio", 0.2)), {}
    if name in ("median", "coordinate_median"):
        return coordinate_median(updates), {}
    if name == "krum":
        f = kwargs.get("krum_f", kwargs.get("malicious_count", 1))
        return krum(updates, f=f, m=kwargs.get("krum_m", 1)), {}
    if name == "dp_fedavg":
        gen = torch.Generator().manual_seed(seed)
        return dp_fedavg(
            updates,
            clip_norm=kwargs.get("dp_clip_norm", 1.0),
            noise_multiplier=kwargs.get("dp_noise_multiplier", 0.5),
            generator=gen,
        )
    raise ValueError(f"Unknown aggregation rule: {name!r} (expected one of {AGGREGATORS})")
