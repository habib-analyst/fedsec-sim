"""Integration test: end-to-end simulation with real training.

Asserts the core security story of the simulator:
  1. A sign-flip + noise model-poisoning attack with no defense devastates
     final accuracy.
  2. Trimmed-mean aggregation recovers essentially all of the lost accuracy.

Small model, few clients/rounds, fixed seed: runs in seconds on CPU.
"""

import time

from fedsec.simulate import run_simulation

BASE = {
    "seed": 7,
    "n_clients": 10,
    "samples_per_client": 150,
    "n_features": 8,
    "hidden": 8,
    "dirichlet_alpha": 0.5,
    "test_size": 1000,
    "local_epochs": 1,
    "lr": 0.1,
    "rounds": 8,
    # Sign-flip model poisoning: malicious clients send -5 * update + noise.
    "poison_scale": -5.0,
    "poison_noise_std": 1.0,
}


def test_attack_hurts_and_defense_recovers():
    clean = run_simulation({**BASE, "malicious_fraction": 0.0,
                            "attack": "none", "defense": "fedavg"})
    attacked = run_simulation({**BASE, "malicious_fraction": 0.2,
                               "attack": "gaussian_poison", "defense": "fedavg"})
    defended = run_simulation({**BASE, "malicious_fraction": 0.2,
                               "attack": "gaussian_poison", "defense": "trimmedmean",
                               "trim_ratio": 0.2})

    acc_clean = clean["final_accuracy"]
    acc_attacked = attacked["final_accuracy"]
    acc_defended = defended["final_accuracy"]

    # 1. The attack with no defense must hurt badly (sign-flip ~ chance level).
    assert acc_attacked < acc_clean - 0.2, (
        f"attack did not hurt: clean={acc_clean:.4f} attacked={acc_attacked:.4f}"
    )
    # 2. The defense must recover at least 90% of the lost accuracy...
    gap = acc_clean - acc_attacked
    recovered = acc_defended - acc_attacked
    assert recovered > 0.9 * gap, (
        f"defense recovered too little: clean={acc_clean:.4f} "
        f"attacked={acc_attacked:.4f} defended={acc_defended:.4f}"
    )
    # ...and land within 2 points of the clean baseline.
    assert acc_defended > acc_clean - 0.02, (
        f"defense did not approach clean: clean={acc_clean:.4f} "
        f"defended={acc_defended:.4f}"
    )


def test_label_flip_attack_hurts():
    # Softer data-poisoning story: label flip degrades the global model.
    cfg = {**BASE, "class_sep": 0.6, "local_epochs": 2, "rounds": 10,
           "dirichlet_alpha": 0.5, "malicious_fraction": 0.2}
    clean = run_simulation({**cfg, "attack": "none", "defense": "fedavg"})
    attacked = run_simulation({**cfg, "attack": "labelflip", "defense": "fedavg"})
    assert attacked["final_accuracy"] < clean["final_accuracy"], (
        f"label flip did not hurt: clean={clean['final_accuracy']:.4f} "
        f"attacked={attacked['final_accuracy']:.4f}"
    )


def test_backdoor_attack_success_rate_is_measured():
    hist = run_simulation({**BASE, "class_sep": 0.6, "malicious_fraction": 0.2,
                           "attack": "backdoor", "defense": "fedavg",
                           "poison_fraction": 0.5, "rounds": 10})
    # With no defense, the backdoor should take hold on the triggered test set.
    assert hist["final_asr"] > 0.5, f"backdoor ASR too low: {hist['final_asr']:.4f}"
    assert len(hist["accuracy"]) == 10
    assert len(hist["attack_success_rate"]) == 10


def test_integration_runs_fast():
    t0 = time.time()
    run_simulation({**BASE, "rounds": 2})
    assert time.time() - t0 < 60
