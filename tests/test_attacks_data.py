"""Unit tests for attacks and data generation."""

import numpy as np
import torch

from fedsec import attacks, data


def test_label_flip():
    y = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    assert (attacks.label_flip_attack(y) == np.array([1, 0, 0, 1, 0])).all()


def test_gaussian_poison_no_noise_is_exact_scale():
    delta = torch.tensor([1.0, -2.0, 0.5])
    out = attacks.gaussian_poison_update(delta, scale=10.0, noise_std=0.0)
    assert torch.allclose(out, delta * 10.0)


def test_gaussian_poison_noise_perturbs():
    delta = torch.zeros(8)
    gen = torch.Generator().manual_seed(3)
    out = attacks.gaussian_poison_update(delta, scale=1.0, noise_std=1.0, generator=gen)
    assert not torch.allclose(out, delta)
    # Seeded generator reproduces the noise exactly.
    gen2 = torch.Generator().manual_seed(3)
    expected = torch.randn(8, generator=gen2)
    assert torch.allclose(out, expected)


def test_backdoor_poison_plants_trigger_and_relabels():
    rng = np.random.default_rng(0)
    x = np.zeros((100, 6), dtype=np.float32)
    y = np.ones(100, dtype=np.int64)
    xp, yp = attacks.backdoor_poison_data(
        x, y, poison_fraction=0.2, target_label=0, rng=rng
    )
    n_poisoned = (yp == 0).sum()
    assert n_poisoned == 20
    # Trigger features stamped to TRIGGER_VALUE exactly on poisoned rows...
    assert (xp[yp == 0][:, list(data.TRIGGER_FEATURES)] == data.TRIGGER_VALUE).all()
    # ...and untouched rows keep their original values and labels.
    assert (xp[yp == 1] == 0.0).all()
    assert (yp[yp == 1] == 1).all()


def test_apply_trigger_only_touches_trigger_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 6)).astype(np.float32)
    xt = data.apply_trigger(x)
    others = [i for i in range(6) if i not in data.TRIGGER_FEATURES]
    assert (xt[:, others] == x[:, others]).all()
    assert (xt[:, list(data.TRIGGER_FEATURES)] == data.TRIGGER_VALUE).all()


def test_make_federated_data_shapes():
    clients, (x_test, y_test), (x_bd, y_bd) = data.make_federated_data(
        n_clients=5, samples_per_client=100, n_features=8, seed=0
    )
    assert len(clients) == 5
    for x, y in clients:
        # Dirichlet integerization floors counts, so totals sit at/under target.
        assert 95 <= x.shape[0] <= 101
        assert x.shape[1] == 8
        assert y.shape == (x.shape[0],)
        assert set(np.unique(y)) <= {0, 1}
    assert x_test.shape[1] == 8
    assert len(x_test) == len(y_test) == len(x_bd) == len(y_bd)
    # Backdoor test labels are all the target label (default 0).
    assert (y_bd == 0).all()


def test_dirichlet_alpha_controls_skew():
    clients_lo, _, _ = data.make_federated_data(
        n_clients=30, samples_per_client=200, dirichlet_alpha=0.1, seed=11
    )
    clients_hi, _, _ = data.make_federated_data(
        n_clients=30, samples_per_client=200, dirichlet_alpha=10.0, seed=11
    )
    skew_lo = data.label_distribution(clients_lo).std()
    skew_hi = data.label_distribution(clients_hi).std()
    assert skew_lo > 3 * skew_hi  # extreme skew vs near-IID
