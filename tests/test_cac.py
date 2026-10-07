"""Unit tests for CAC (Centered Adaptive Clipping).

Verified against hand computation. Key properties under test:
- exact FedAvg reduction when all updates lie inside the trust ball;
- hand-computed robust-branch output;
- breakdown point floor((n-1)/2): bounded output below it, corrupted at it.
"""

import pytest
import torch

from fedsec.defenses import aggregate, cac, fedavg


def t(rows):
    return [torch.tensor(r, dtype=torch.float32) for r in rows]


def test_cac_identical_updates_is_exactly_fedavg():
    updates = t([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    out, report = cac(updates)
    assert torch.allclose(out, fedavg(updates))
    assert report["cac_branch"] == "fedavg"


def test_cac_homogeneous_cluster_is_exactly_fedavg():
    # Tight cluster, max d well inside the trust radius -> FedAvg branch.
    # (alpha=4 gives comfortable margin; float32 knife-edges avoided.)
    updates = t([[0.0], [0.01], [0.02], [0.03]])
    out, report = cac(updates, alpha=4.0)
    assert report["cac_branch"] == "fedavg"
    assert torch.allclose(out, fedavg(updates))


def test_cac_robust_branch_hand_computed():
    # n=5, 1-D. center (median) = 2. d = [2,1,0,1,98]; med_d = 1; r = 2.
    # max d = 98 > 2 -> robust branch.
    # clipped: [0,1,2,3, 2+98*(2/98)] = [0,1,2,3,4].
    # weights 1/(1+(d/2)^2): [0.5, 0.8, 1.0, 0.8, 1/2402].
    # agg = (0*0.5 + 1*0.8 + 2*1.0 + 3*0.8 + 4/2402) / (3.1 + 1/2402)
    #     = 5.201665 / 3.100416 ~= 1.67773.
    updates = t([[0.0], [1.0], [2.0], [3.0], [100.0]])
    out, report = cac(updates, alpha=2.0, power=2.0)
    assert report["cac_branch"] == "robust"
    assert out.item() == pytest.approx(1.67773, abs=1e-4)
    # The Byzantine update was clipped to center + radius = 4, then downweighted.
    assert report["cac_radius"] == pytest.approx(2.0)
    assert report["cac_max_distance"] == pytest.approx(98.0)


def test_cac_breakdown_point_below_threshold_stays_bounded():
    # n=10, f=4 = floor((10-1)/2): 4 Byzantine sending 1e6, 6 honest near 0.
    honest = [torch.zeros(8) + 0.01 * i for i in range(6)]
    byz = [torch.ones(8) * 1e6 for _ in range(4)]
    out, report = cac(honest + byz)
    assert out.norm().item() < 10.0


def test_cac_breakdown_point_at_threshold_corrupts():
    # n=11 (odd), f=6 > floor((11-1)/2) = 5: the median itself is Byzantine
    # -> center corrupted -> aggregate unbounded.
    honest = [torch.zeros(8) + 0.01 * i for i in range(5)]
    byz = [torch.ones(8) * 1e6 for _ in range(6)]
    out, _ = cac(honest + byz)
    assert out.norm().item() > 1e4


def test_cac_single_update_is_identity():
    updates = t([[3.0, -1.0]])
    out, report = cac(updates)
    assert torch.allclose(out, updates[0])
    assert report["cac_branch"] == "fedavg"


def test_cac_weights_favor_close_updates():
    # Two tight honest updates + one distant Byzantine; Byzantine weight ~ 0.
    updates = t([[0.0], [0.1], [50.0]])
    out, report = cac(updates, alpha=2.0, power=2.0)
    assert report["cac_branch"] == "robust"
    assert report["cac_min_weight"] < 0.01
    # Aggregate stays near the honest cluster, far from 50.
    assert out.item() == pytest.approx(0.05, abs=0.5)


def test_cac_deterministic():
    updates = t([[float(i)] for i in range(7)])
    out1, _ = cac(updates)
    out2, _ = cac(updates)
    assert torch.allclose(out1, out2)


def test_aggregate_dispatch_cac():
    updates = t([[1.0], [2.0], [3.0], [4.0], [5.0]])
    out, report = aggregate("cac", updates, seed=0, cac_alpha=2.0, cac_power=2.0)
    assert out.shape == (1,)
    assert report["cac_branch"] in ("fedavg", "robust")
