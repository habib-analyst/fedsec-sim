"""Simulation loop: federated rounds with attack injection and defense.

One round: sample clients -> local SGD training -> malicious clients poison
their data (label flip / backdoor) or their update (Gaussian model poisoning)
-> server aggregates with the configured defense -> evaluate the global model.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from . import attacks, defenses
from .data import make_federated_data
from .models import MLP, accuracy, flatten_params, load_flat_params

DEFAULTS = {
    "seed": 42,
    "n_clients": 20,
    "malicious_fraction": 0.2,
    "samples_per_client": 200,
    "n_features": 10,
    "hidden": 16,
    "dirichlet_alpha": 0.5,
    "class_sep": 1.0,
    "test_size": 2000,
    "client_fraction": 1.0,
    "local_epochs": 2,
    "local_batch_size": 32,
    "lr": 0.05,
    "rounds": 20,
    "attack": "none",  # none | labelflip | gaussian_poison | backdoor
    "defense": "fedavg",  # fedavg | trimmedmean | median | krum | dp_fedavg
    "poison_fraction": 0.3,  # backdoor: fraction of malicious client's data poisoned
    "backdoor_target_label": 0,
    "poison_scale": -5.0,  # gaussian model poisoning: update scale (negative = sign-flip)
    "poison_noise_std": 1.0,  # gaussian model poisoning: noise std
    "trim_ratio": 0.2,
    "krum_m": 1,
    "dp_clip_norm": 1.0,
    "dp_noise_multiplier": 0.5,
}

ATTACKS = ("none", "labelflip", "gaussian_poison", "backdoor")


def load_config(path: str, overrides: dict | None = None) -> dict:
    """Load a YAML config file and fill in defaults."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged = {**DEFAULTS, **cfg}
    if overrides:
        merged.update(overrides)
    if merged["attack"] not in ATTACKS:
        raise ValueError(f"unknown attack {merged['attack']!r}")
    return merged


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _local_train(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y).float()
    n = len(x)
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()


def run_simulation(cfg: dict, verbose: bool = False) -> dict:
    """Run the full federated simulation described by ``cfg``.

    Returns a dict with per-round ``accuracy`` and ``attack_success_rate``
    histories plus run metadata.
    """
    cfg = {**DEFAULTS, **cfg}
    _seed_all(cfg["seed"])
    data_rng = np.random.default_rng(cfg["seed"] + 1)

    clients, (x_test, y_test), (x_bd, y_bd) = make_federated_data(
        n_clients=cfg["n_clients"],
        samples_per_client=cfg["samples_per_client"],
        n_features=cfg["n_features"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        class_sep=cfg["class_sep"],
        test_size=cfg["test_size"],
        backdoor_target_label=cfg["backdoor_target_label"],
        seed=cfg["seed"],
    )
    n_malicious = int(round(cfg["n_clients"] * cfg["malicious_fraction"]))
    malicious_ids = set(
        data_rng.choice(cfg["n_clients"], size=n_malicious, replace=False).tolist()
    )
    if verbose:
        print(f"malicious clients: {sorted(malicious_ids)}")

    model = MLP(cfg["n_features"], hidden=cfg["hidden"])
    global_flat = flatten_params(model)
    dp_report: dict = {}

    history = {"accuracy": [], "attack_success_rate": [], "config": cfg}

    for rnd in range(cfg["rounds"]):
        n_sample = max(1, int(cfg["n_clients"] * cfg["client_fraction"]))
        sampled = data_rng.choice(cfg["n_clients"], size=n_sample, replace=False)
        updates = []
        for cid in sampled:
            x_c, y_c = clients[int(cid)]
            local = MLP(cfg["n_features"], hidden=cfg["hidden"])
            load_flat_params(local, global_flat)
            # Data poisoning happens before local training.
            if int(cid) in malicious_ids:
                if cfg["attack"] == "labelflip":
                    y_c = attacks.label_flip_attack(y_c)
                elif cfg["attack"] == "backdoor":
                    x_c, y_c = attacks.backdoor_poison_data(
                        x_c,
                        y_c,
                        poison_fraction=cfg["poison_fraction"],
                        target_label=cfg["backdoor_target_label"],
                        rng=data_rng,
                    )
            _local_train(
                local,
                x_c,
                y_c,
                epochs=cfg["local_epochs"],
                batch_size=cfg["local_batch_size"],
                lr=cfg["lr"],
                rng=data_rng,
            )
            delta = flatten_params(local) - global_flat
            # Model poisoning happens after local training.
            if int(cid) in malicious_ids and cfg["attack"] == "gaussian_poison":
                gen = torch.Generator().manual_seed(cfg["seed"] + rnd * 1000 + int(cid))
                delta = attacks.gaussian_poison_update(
                    delta,
                    scale=cfg["poison_scale"],
                    noise_std=cfg["poison_noise_std"],
                    generator=gen,
                )
            updates.append(delta)

        agg, report = defenses.aggregate(
            cfg["defense"],
            updates,
            seed=cfg["seed"] + rnd,
            trim_ratio=cfg["trim_ratio"],
            krum_m=cfg["krum_m"],
            krum_f=n_malicious,
            malicious_count=n_malicious,
            dp_clip_norm=cfg["dp_clip_norm"],
            dp_noise_multiplier=cfg["dp_noise_multiplier"],
        )
        if report:
            dp_report = report
        global_flat = global_flat + agg
        load_flat_params(model, global_flat)

        acc = accuracy(model, x_test, y_test)
        asr = accuracy(model, x_bd, y_bd)  # triggered -> target label?
        history["accuracy"].append(acc)
        history["attack_success_rate"].append(asr)
        if verbose:
            print(f"round {rnd + 1:>3}/{cfg['rounds']}  acc={acc:.4f}  asr={asr:.4f}")

    history["final_accuracy"] = history["accuracy"][-1]
    history["final_asr"] = history["attack_success_rate"][-1]
    history["n_malicious"] = n_malicious
    history["dp_report"] = dp_report
    return history
