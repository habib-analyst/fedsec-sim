"""Command-line interface: ``fedsec run | compare | demo``."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from .plots import plot_accuracy, plot_asr_bars
from .simulate import load_config, run_simulation

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def _resolve_config(path: str) -> str:
    if os.path.exists(path):
        return path
    cand = os.path.join(CONFIG_DIR, path)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"config not found: {path}")


def _run_one(config_path: str, rounds: int | None, out_dir: str | None, label: str | None):
    cfg_path = _resolve_config(config_path)
    overrides = {"rounds": rounds} if rounds else None
    cfg = load_config(cfg_path, overrides)
    print(f"[fedsec] running '{label or cfg_path}' "
          f"(attack={cfg['attack']}, defense={cfg['defense']}, rounds={cfg['rounds']})")
    t0 = time.time()
    hist = run_simulation(cfg, verbose=True)
    dt = time.time() - t0
    print(f"[fedsec] done in {dt:.1f}s: final acc={hist['final_accuracy']:.4f} "
          f"asr={hist['final_asr']:.4f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = (label or os.path.splitext(os.path.basename(cfg_path))[0]).replace(" ", "_")
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(
                {k: v for k, v in hist.items() if k != "config"}, f, indent=2
            )
        with open(os.path.join(out_dir, f"{name}.config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        plot_accuracy({name: hist}, os.path.join(out_dir, f"{name}.accuracy.png"))
    return hist, cfg


def _summary_table(results: list[tuple[str, dict, dict]]) -> str:
    header = ["config", "attack", "defense", "rounds", "final acc", "final ASR"]
    rows = [
        [
            name,
            cfg["attack"],
            cfg["defense"],
            str(cfg["rounds"]),
            f"{hist['final_accuracy']:.4f}",
            f"{hist['final_asr']:.4f}",
        ]
        for name, hist, cfg in results
    ]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    def fmt(row):
        return " | ".join(v.ljust(w) for v, w in zip(row, widths))
    lines = [fmt(header), "-+-".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def cmd_run(args) -> int:
    out = args.out or "results"
    _run_one(args.config, args.rounds, out, args.label)
    return 0


def cmd_compare(args) -> int:
    out = args.out or "results"
    os.makedirs(out, exist_ok=True)
    results = []
    histories = {}
    for cfg_path in args.configs:
        label = os.path.splitext(os.path.basename(cfg_path))[0]
        hist, cfg = _run_one(cfg_path, args.rounds, out, label)
        results.append((label, hist, cfg))
        histories[label] = hist
    table = _summary_table(results)
    print("\n" + table)
    with open(os.path.join(out, "compare_summary.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out, "compare_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "attack", "defense", "rounds", "final_acc", "final_asr"])
        for name, hist, cfg in results:
            w.writerow([name, cfg["attack"], cfg["defense"], cfg["rounds"],
                        hist["final_accuracy"], hist["final_asr"]])
    plot_accuracy(histories, os.path.join(out, "compare.accuracy.png"))
    plot_asr_bars(histories, os.path.join(out, "compare.asr.png"))
    print(f"\n[fedsec] wrote {out}/compare_summary.csv and plots")
    return 0


def cmd_demo(args) -> int:
    """Fast end-to-end demo: clean vs label-flip (no defense) vs trimmed mean."""
    out = args.out or "results/demo"
    os.makedirs(out, exist_ok=True)
    demo_cfgs = {
        "clean (no attack, FedAvg)": {
            "seed": 7, "n_clients": 10, "malicious_fraction": 0.0,
            "samples_per_client": 150, "n_features": 8, "hidden": 8,
            "dirichlet_alpha": 0.5, "test_size": 1000, "local_epochs": 1,
            "lr": 0.1, "rounds": 8, "attack": "none", "defense": "fedavg",
        },
        "sign-flip poisoning 20% (FedAvg, no defense)": {
            "seed": 7, "n_clients": 10, "malicious_fraction": 0.2,
            "samples_per_client": 150, "n_features": 8, "hidden": 8,
            "dirichlet_alpha": 0.5, "test_size": 1000, "local_epochs": 1,
            "lr": 0.1, "rounds": 8, "attack": "gaussian_poison",
            "poison_scale": -5.0, "poison_noise_std": 1.0, "defense": "fedavg",
        },
        "sign-flip poisoning 20% (trimmed mean)": {
            "seed": 7, "n_clients": 10, "malicious_fraction": 0.2,
            "samples_per_client": 150, "n_features": 8, "hidden": 8,
            "dirichlet_alpha": 0.5, "test_size": 1000, "local_epochs": 1,
            "lr": 0.1, "rounds": 8, "attack": "gaussian_poison",
            "poison_scale": -5.0, "poison_noise_std": 1.0,
            "defense": "trimmedmean", "trim_ratio": 0.2,
        },
    }
    results, histories = [], {}
    for name, cfg in demo_cfgs.items():
        print(f"[fedsec] demo: {name}")
        t0 = time.time()
        hist = run_simulation(cfg, verbose=False)
        print(f"         rounds={cfg['rounds']} final acc={hist['final_accuracy']:.4f} "
              f"({time.time() - t0:.1f}s)")
        results.append((name, hist, cfg))
        histories[name] = hist
    table = _summary_table(results)
    print("\n" + table)
    with open(os.path.join(out, "demo_summary.txt"), "w") as f:
        f.write(table + "\n")
    plot_accuracy(histories, os.path.join(out, "demo.accuracy.png"))
    print(f"\n[fedsec] demo complete. Plot: {out}/demo.accuracy.png")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fedsec",
                                description="Federated learning security simulator")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one config")
    r.add_argument("--config", required=True, help="YAML config path (or name in configs/)")
    r.add_argument("--rounds", type=int, default=None, help="override rounds")
    r.add_argument("--out", default="results", help="output directory")
    r.add_argument("--label", default=None, help="run label for outputs")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="run several configs and compare")
    c.add_argument("--configs", nargs="+", required=True, help="YAML config paths")
    c.add_argument("--rounds", type=int, default=None, help="override rounds")
    c.add_argument("--out", default="results", help="output directory")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo", help="fast 3-way demo (<60s CPU)")
    d.add_argument("--out", default="results/demo", help="output directory")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
