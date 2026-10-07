"""Synthetic non-IID federated data generation.

Binary classification on tabular data with Dirichlet-skewed class
distributions per client (the standard non-IID recipe from FedAvg-era
experiments), plus a planted trigger pattern used by the backdoor attack.
"""

from __future__ import annotations

import numpy as np

#: Feature indices that carry the backdoor trigger pattern.
TRIGGER_FEATURES = (0, 1)
#: Value the trigger writes into the trigger features (far outside the data range).
TRIGGER_VALUE = 4.0


def _make_gaussian_blobs(n: int, n_features: int, rng: np.random.Generator,
                        class_sep: float = 1.0):
    """Two Gaussian blobs with means at ``-class_sep`` and ``+class_sep``."""
    half = n // 2
    x0 = rng.normal(loc=-class_sep, scale=1.0, size=(half, n_features))
    x1 = rng.normal(loc=+class_sep, scale=1.0, size=(n - half, n_features))
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).astype(np.int64)
    perm = rng.permutation(n)
    return x[perm].astype(np.float32), y[perm]


def apply_trigger(
    x: np.ndarray,
    trigger_features: tuple[int, ...] = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
) -> np.ndarray:
    """Stamp the backdoor trigger onto a batch: overwrite trigger features."""
    x = x.copy()
    x[:, list(trigger_features)] = trigger_value
    return x.astype(np.float32)


def make_federated_data(
    n_clients: int,
    samples_per_client: int,
    n_features: int = 10,
    dirichlet_alpha: float = 0.5,
    test_size: int = 2000,
    backdoor_target_label: int = 0,
    class_sep: float = 1.0,
    seed: int = 42,
):
    """Create a non-IID federated dataset.

    Each client draws a class proportion from Dirichlet(alpha, alpha) and
    receives that many samples of each class (class counts are integerized,
    leftover samples are dropped). Small ``alpha`` -> extreme skew.

    Returns:
        clients: list of (X, y) tuples, one per client.
        test: (X_test, y_test) held-out clean test set.
        backdoor_test: (X_triggered, y_target) triggered inputs labelled with
            the attacker's target label, used to measure attack success rate.
    """
    rng = np.random.default_rng(seed)

    # Big shared pool, then split per client by Dirichlet class proportions.
    pool_size = n_clients * samples_per_client * 2
    x_pool, y_pool = _make_gaussian_blobs(pool_size, n_features, rng, class_sep)
    pool_by_class = [x_pool[y_pool == c] for c in (0, 1)]
    idx = [0, 0]

    def take(c: int, k: int) -> np.ndarray:
        start, idx[c] = idx[c], idx[c] + k
        return pool_by_class[c][start:idx[c]]

    clients = []
    for _ in range(n_clients):
        proportions = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
        counts = (proportions * samples_per_client).astype(int)
        # Guarantee at least one sample per class so every client can train.
        counts = np.maximum(counts, 1)
        xs = [take(c, counts[c]) for c in (0, 1)]
        x = np.vstack(xs).astype(np.float32)
        y = np.concatenate(
            [np.full(counts[c], c, dtype=np.int64) for c in (0, 1)]
        )
        perm = rng.permutation(len(x))
        clients.append((x[perm], y[perm]))

    x_test, y_test = _make_gaussian_blobs(test_size, n_features, rng, class_sep)
    x_triggered = apply_trigger(x_test)
    y_target = np.full(len(x_test), backdoor_target_label, dtype=np.int64)
    return clients, (x_test, y_test), (x_triggered, y_target)


def label_distribution(clients) -> np.ndarray:
    """Fraction of class-1 samples per client (diagnostic for skew)."""
    return np.array([y.mean() for _, y in clients])
