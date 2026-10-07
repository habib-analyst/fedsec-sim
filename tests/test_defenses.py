"""Unit tests for aggregation rules — all verified against hand computation."""

import pytest
import torch

from fedsec.defenses import (
    aggregate,
    coordinate_median,
    dp_fedavg,
    fedavg,
    krum,
    trimmed_mean,
)


def t(rows):
    return [torch.tensor(r, dtype=torch.float32) for r in rows]


def test_fedavg_is_plain_mean():
    out = fedavg(t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert torch.allclose(out, torch.tensor([3.0, 4.0]))


def test_trimmed_mean_hand_computed():
    # n=4, beta=0.25 -> k=1: drop min and max per coordinate, mean the rest.
    out = trimmed_mean(t([[0.0], [10.0], [20.0], [30.0]]), beta=0.25)
    assert torch.allclose(out, torch.tensor([15.0]))


def test_trimmed_mean_beta_zero_is_mean():
    updates = t([[1.0, 0.0], [2.0, 10.0], [3.0, -4.0]])
    assert torch.allclose(trimmed_mean(updates, beta=0.0), fedavg(updates))


def test_trimmed_mean_2d_hand_computed():
    # per-coordinate: x -> drop 0,30 -> mean(10,20)=15; y -> drop -5,50 -> mean(0,5)=2.5
    out = trimmed_mean(
        t([[0.0, -5.0], [10.0, 0.0], [20.0, 5.0], [30.0, 50.0]]), beta=0.25
    )
    assert torch.allclose(out, torch.tensor([15.0, 2.5]))


def test_coordinate_median_hand_computed():
    out = coordinate_median(t([[0.0], [10.0], [20.0]]))
    assert torch.allclose(out, torch.tensor([10.0]))
    out2 = coordinate_median(t([[5.0, 100.0], [1.0, 2.0], [3.0, 3.0]]))
    assert torch.allclose(out2, torch.tensor([3.0, 3.0]))


def test_krum_rejects_outlier():
    # Three clustered updates + one far outlier; f=1.
    # n - f - 2 = 1 nearest neighbour.
    # a=[0,0]: nn b, d2=0.01 -> score 0.01
    # b=[0.1,0]: nn a, d2=0.01 -> score 0.01
    # c=[0.1,0.1]: nn b, d2=0.01 -> score 0.01
    # d=[10,10]: nn c, d2=196.02 -> score 196.02
    updates = t([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [10.0, 10.0]])
    out = krum(updates, f=1, m=1)
    assert torch.allclose(out, updates[0])  # argmin picks the first min-score
    # m=2 averages the two best updates (a and b).
    out2 = krum(updates, f=1, m=2)
    assert torch.allclose(out2, torch.tensor([0.05, 0.0]))


def test_krum_degenerate_few_updates_falls_back_to_mean():
    updates = t([[1.0], [3.0]])
    assert torch.allclose(krum(updates, f=1), torch.tensor([2.0]))


def test_dp_fedavg_no_noise_equals_clipped_mean():
    updates = t([[3.0, 0.0], [0.0, 4.0]])  # norms 3 and 4
    out, report = dp_fedavg(updates, clip_norm=1.0, noise_multiplier=0.0)
    # clipped: [1,0] and [0,1] -> mean [0.5, 0.5]
    assert torch.allclose(out, torch.tensor([0.5, 0.5]))
    assert report["clip_norm"] == 1.0
    assert report["noise_multiplier"] == 0.0
    assert report["noise_std"] == 0.0


def test_dp_fedavg_noise_actually_perturbs():
    updates = t([[1.0, 2.0], [3.0, 4.0]])
    gen = torch.Generator().manual_seed(0)
    out, report = dp_fedavg(updates, clip_norm=10.0, noise_multiplier=1.0, generator=gen)
    plain = fedavg(updates)
    assert not torch.allclose(out, plain)  # noise was added
    assert report["noise_std"] == pytest.approx(10.0)
    # Same seed -> same noise: reproducible.
    gen2 = torch.Generator().manual_seed(0)
    out2, _ = dp_fedavg(updates, clip_norm=10.0, noise_multiplier=1.0, generator=gen2)
    assert torch.allclose(out, out2)


def test_dp_fedavg_clipping_scales_large_updates():
    big = torch.ones(4) * 5.0  # norm 10
    small = torch.ones(4) * 0.1  # norm 0.2
    out, _ = dp_fedavg([big, small], clip_norm=2.0, noise_multiplier=0.0)
    # big clipped to norm 2 -> all entries 1.0; small untouched.
    expected = (torch.ones(4) * 1.0 + torch.ones(4) * 0.1) / 2
    assert torch.allclose(out, expected)


def test_aggregate_dispatch_unknown_raises():
    with pytest.raises(ValueError, match="Unknown aggregation rule"):
        aggregate("byzantine_magic", t([[1.0]]))


def test_aggregate_dispatch_all_rules():
    updates = t([[1.0], [2.0], [3.0], [4.0], [5.0]])
    for name, kwargs in [
        ("fedavg", {}),
        ("trimmedmean", {"trim_ratio": 0.2}),
        ("median", {}),
        ("krum", {"malicious_count": 1, "krum_m": 1}),
        ("dp_fedavg", {"dp_clip_norm": 10.0, "dp_noise_multiplier": 0.0}),
    ]:
        out, report = aggregate(name, updates, seed=0, **kwargs)
        assert out.shape == (1,)
        assert isinstance(report, dict)
