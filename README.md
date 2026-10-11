[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/fedsec-sim.svg)]

# fedsec-sim

Federated learning security simulator: prototype poisoning attacks and Byzantine-robust defenses in ~10 seconds on a laptop CPU.

Spinning up a full FL framework to answer "does trimmed mean survive 20% sign-flip attackers on my task?" is overkill — `fedsec` answers it on CPU before you pay for a real deployment.

## Quickstart

```bash
git clone https://github.com/habib-analyst/fedsec-sim.git && cd fedsec-sim
pip install -r requirements.txt   # CPU-only torch
python -m fedsec demo             # clean vs attacked vs defended (~9 s on CPU)
```

Real output of `python -m fedsec demo` (3 configs × 8 rounds):

```
[fedsec] demo: clean (no attack, FedAvg)
         rounds=8 final acc=0.9950 (2.0s)
[fedsec] demo: sign-flip poisoning 20% (FedAvg, no defense)
         rounds=8 final acc=0.5000 (0.3s)
[fedsec] demo: sign-flip poisoning 20% (trimmed mean)
         rounds=8 final acc=0.9950 (0.3s)
```

20% of clients send `−5 × update + Gaussian noise`. Plain FedAvg collapses to chance (0.5000); coordinate-wise trimmed mean recovers the clean baseline exactly (0.9950).

Also: `fedsec run --config configs/labelflip_trimmedmean.yaml` (single YAML config), `fedsec compare --configs ...` (side-by-side table + plots), `pytest -q` (test suite, no network needed).

## Measured results

CAC (coordinate-averaging with clamping — `fedsec/defenses.py::cac`, proof sketch in `hidden_files/papers/cac/`), Gaussian poisoning, 2 seeds per cell:

| metric | value |
|---|---|
| CAC accuracy, up to 30% malicious clients | **0.9992** |
| FedAvg accuracy, 30% Gaussian poisoning | **0.0015** (total collapse) |
| unit tests | 32/32 |

Full per-cell rows: `hidden_files/papers/cac/results/cac_e1_e2_summary.csv`.

## How it works

```
synthetic binary      Dirichlet(α)      ┌──────────────────────────────┐
tabular data ──────▶ non-IID split ──▶ │ attacks.py: label flip,      │
                    N clients          │ sign-flip/gaussian poison,   │
                                       │ backdoor trigger             │
                    local SGD          └──────────────┬───────────────┘
                    (torch MLP, CPU)                  │ poisoned updates
                                                      ▼
                    aggregate: fedavg (vulnerable baseline) · trimmedmean · median
                               · krum · dp_fedavg  →  accuracy + backdoor ASR
```

Every knob — clients, Dirichlet α, attack/defense, DP clip & noise, rounds — is a YAML key in `configs/` (`clean.yaml`, `labelflip_fedavg.yaml`, `labelflip_trimmedmean.yaml`, `poison_krum.yaml`, `backdoor_dp.yaml`).

Limits, read before citing this in a paper: this is a simulator, not a security proof — synthetic Gaussian blobs don't transfer to real tasks, and robust-aggregation guarantees rest on assumptions real deployments routinely violate. `dp_fedavg` implements clipping + Gaussian noise but computes no (ε, δ) guarantee. Toy 2-layer MLP on binary data exercises the aggregation math, not deep networks or adaptive attackers.

## Citations

```bibtex
@inproceedings{mcmahan2017fedavg,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and {Ag{\"u}era y Arcas}, Blaise},
  booktitle={AISTATS}, year={2017}
}
@inproceedings{blanchard2017krum,
  title={Machine learning with adversaries: {B}yzantine tolerant gradient descent},
  author={Blanchard, Peva and El Mhamdi, El Mahdi and Guerraoui, Rachid and Stainer, Julien},
  booktitle={NeurIPS}, year={2017}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
