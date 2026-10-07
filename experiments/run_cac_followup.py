"""CAC follow-up: (a) krum without oracle f, (b) CAC alpha sensitivity.

Cells (all f/n=0.4, 20 rounds, 2 seeds):
- labelflip x {krum_f=1, krum_f=4, cac alpha=1.5, cac alpha=3.0}
- backdoor  x {krum_f=1, cac alpha=2.0}
Appends to hidden_files/papers/cac/results/cac_followup.json.
"""

from __future__ import annotations

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


def _run(job: dict) -> dict:
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, **job}
    h = run_simulation(cfg, verbose=False)
    out = {
        "attack": job["attack"],
        "defense": job["defense"],
        "krum_f_override": job.get("krum_f_override"),
        "cac_alpha": job.get("cac_alpha"),
        "seed": job["seed"],
        "final_accuracy": h["final_accuracy"],
        "final_asr": h["final_asr"],
    }
    if job["defense"] == "cac":
        branches = [r.get("cac_branch") for r in h["reports"]]
        out["cac_fedavg_branch_rate"] = sum(
            b == "fedavg" for b in branches
        ) / max(len(branches), 1)
    return out


def main() -> None:
    jobs: list[dict] = []
    for seed in (1000, 1001):
        for krum_f in (1, 4):
            jobs.append(
                {
                    "attack": "labelflip",
                    "defense": "krum",
                    "krum_f_override": krum_f,
                    "malicious_fraction": 0.4,
                    "seed": seed,
                    "rounds": 20,
                }
            )
        for alpha in (1.5, 3.0):
            jobs.append(
                {
                    "attack": "labelflip",
                    "defense": "cac",
                    "cac_alpha": alpha,
                    "malicious_fraction": 0.4,
                    "seed": seed,
                    "rounds": 20,
                }
            )
        jobs.append(
            {
                "attack": "backdoor",
                "defense": "krum",
                "krum_f_override": 1,
                "malicious_fraction": 0.4,
                "seed": seed,
                "rounds": 20,
            }
        )
        jobs.append(
            {
                "attack": "backdoor",
                "defense": "cac",
                "cac_alpha": 2.0,
                "malicious_fraction": 0.4,
                "seed": seed,
                "rounds": 20,
            }
        )
    print(f"running {len(jobs)} follow-up cells ...", flush=True)
    with Pool(2) as pool:
        results = pool.map(_run, jobs)
    path = os.path.join(RESULTS_DIR, "cac_followup.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(
            "%(attack)s %(defense)s krum_f=%(krum_f_override)s alpha=%(cac_alpha)s "
            "seed=%(seed)s acc=%(final_accuracy).4f asr=%(final_asr).4f fb=%(cac_fedavg_branch_rate)s" % {
                **r, "cac_fedavg_branch_rate": r.get("cac_fedavg_branch_rate", "-"),
            },
            flush=True,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
