"""CAC experiments E1 (breakdown sweep) and E2 (non-IID stress).

E1: f/n in {0, 0.1, 0.2, 0.3, 0.4} x attacks {labelflip, gaussian_poison,
    backdoor} x defenses {fedavg, trimmedmean, median, krum, dp_fedavg, cac}.
    Metrics: final clean accuracy, final backdoor ASR.
E2: attack=none, defense in {fedavg, cac}, Dirichlet alpha in
    {0.1, 0.5, 1.0, 10}. Metrics: final accuracy gap CAC-vs-FedAvg and
    CAC's FedAvg-branch fallback rate.

Results -> hidden_files/papers/cac/results/cac_e1_e2.json (plus CSV tables).
Run: python3 experiments/run_cac_experiments.py [--workers 2]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedsec.simulate import DEFAULTS, run_simulation  # noqa: E402

RESULTS_DIR = os.path.expanduser(
    "~/workspace/goals/github-portfolio-linkedin-presence-buildout"
    "/hidden_files/papers/cac/results"
)

E1_F_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4]
E1_ATTACKS = ["labelflip", "gaussian_poison", "backdoor"]
E1_DEFENSES = ["fedavg", "trimmedmean", "median", "krum", "dp_fedavg", "cac"]
E1_SEEDS = [0, 1]

E2_ALPHAS = [0.1, 0.5, 1.0, 10.0]
E2_DEFENSES = ["fedavg", "cac"]
E2_SEEDS = [0, 1, 2]


def _run_cell(job: dict) -> dict:
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, **job["cfg"]}
    h = run_simulation(cfg, verbose=False)
    out = {
        "job": {k: v for k, v in job["cfg"].items() if k != "seed"},
        "seed": job["cfg"]["seed"],
        "final_accuracy": h["final_accuracy"],
        "final_asr": h["final_asr"],
        "n_malicious": h["n_malicious"],
    }
    if job["cfg"].get("defense") == "cac":
        branches = [r.get("cac_branch") for r in h["reports"]]
        out["cac_fedavg_branch_rate"] = sum(
            b == "fedavg" for b in branches
        ) / max(len(branches), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    jobs: list[dict] = []
    # E1: breakdown sweep.
    for f_frac, attack, defense, seed in itertools.product(
        E1_F_FRACS, E1_ATTACKS, E1_DEFENSES, E1_SEEDS
    ):
        jobs.append(
            {
                "exp": "E1",
                "cfg": {
                    "experiment": "E1",
                    "malicious_fraction": f_frac,
                    "attack": attack,
                    "defense": defense,
                    "seed": 1000 + seed,
                    "rounds": 20,
                },
            }
        )
    # E2: non-IID stress, honest.
    for alpha, defense, seed in itertools.product(
        E2_ALPHAS, E2_DEFENSES, E2_SEEDS
    ):
        jobs.append(
            {
                "exp": "E2",
                "cfg": {
                    "experiment": "E2",
                    "malicious_fraction": 0.0,
                    "attack": "none",
                    "defense": defense,
                    "dirichlet_alpha": alpha,
                    "seed": 2000 + seed,
                    "rounds": 20,
                },
            }
        )

    print(f"running {len(jobs)} cells on {args.workers} workers ...", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(_run_cell, jobs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "cac_e1_e2.json"), "w") as f:
        json.dump(results, f, indent=1)

    # CSV summary: mean over seeds per cell.
    cells: dict[tuple, list[dict]] = {}
    for r in results:
        key = (
            r["job"]["experiment"],
            r["job"].get("malicious_fraction"),
            r["job"].get("attack"),
            r["job"].get("defense"),
            r["job"].get("dirichlet_alpha"),
        )
        cells.setdefault(key, []).append(r)
    csv_path = os.path.join(RESULTS_DIR, "cac_e1_e2_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "malicious_fraction",
                "attack",
                "defense",
                "dirichlet_alpha",
                "n_seeds",
                "mean_accuracy",
                "std_accuracy",
                "mean_asr",
                "mean_cac_fedavg_branch_rate",
            ]
        )
        for key in sorted(cells, key=str):
            rs = cells[key]
            accs = [r["final_accuracy"] for r in rs]
            asrs = [r["final_asr"] for r in rs]
            fb = [r.get("cac_fedavg_branch_rate") for r in rs]
            fb = [x for x in fb if x is not None]
            mean_acc = sum(accs) / len(accs)
            var = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
            w.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    len(rs),
                    round(mean_acc, 4),
                    round(var**0.5, 4),
                    round(sum(asrs) / len(asrs), 4),
                    round(sum(fb) / len(fb), 4) if fb else "",
                ]
            )
    print(f"wrote {RESULTS_DIR}/cac_e1_e2.json and cac_e1_e2_summary.csv", flush=True)


if __name__ == "__main__":
    main()
