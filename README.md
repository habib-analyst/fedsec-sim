# fedsec — Federated Learning Security Simulator

[![CI](https://github.com/habib-analyst/fedsec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/fedsec-sim/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Problem

Federated learning researchers need a **fast, CPU-only testbed** to prototype
poisoning attacks and Byzantine-robust defenses *before* paying for a real
deployment. Spinning up a full FL framework to answer "does trimmed mean
survive 20% sign-flip attackers on my task?" is overkill — `fedsec` answers
it in ~10 seconds on a laptop.

## Architecture

```
┌─────────────┐   Dirichlet(α)    ┌──────────────────┐
│  data.py    │  non-IID split   │   attacks.py       │
│ synthetic   ├─────────────────►│  • label flip    │
│ binary      │  N clients       │  • sign-flip +     │
│ tabular     │                  │    Gaussian poison │
└─────────────┘                  │  • backdoor trigger│
        │                        └────────┬─────────┘
        │  local SGD (torch MLP, CPU)     │ poisoned updates
        ▼                                 ▼
┌──────────────────────────────────────────────────┐
│                 simulate.py                      │
│   sample clients → local train → attack inject   │
│   → aggregate → evaluate (acc + backdoor ASR)    │
└──────────────────────┬───────────────────────────┘
                       │ updates
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐
  │ FedAvg   │  │ trimmed    │  │ median   │  │ Krum /   │
  │ (no def) │  │ mean       │  │          │  │ DP-FedAvg│
  └──────────┘  └────────────┘  └──────────┘  └──────────┘
                   defenses.py — all operate on flat update vectors
```

## Quickstart (< 5 min)

```bash
git clone https://github.com/habib-analyst/fedsec-sim.git
cd fedsec-sim
pip install -r requirements.txt   # CPU-only torch via the PyTorch CPU index

# 1. Run the fast demo: clean vs attacked vs defended (~10 s on CPU)
python -m fedsec demo

# 2. Run a single YAML config
python -m fedsec run --config configs/labelflip_trimmedmean.yaml --rounds 20 --out results/

# 3. Compare several configs side by side (table + plots)
python -m fedsec compare --configs configs/clean.yaml configs/labelflip_fedavg.yaml \
    configs/poison_krum.yaml --rounds 20 --out results/

# 4. Run the test suite (no network needed)
pytest -q
```

## Example output

Actual output of `python -m fedsec demo` (3 configs × 8 rounds, ~9 s on CPU):

```
[fedsec] demo: clean (no attack, FedAvg)
         rounds=8 final acc=0.9950 (2.0s)
[fedsec] demo: sign-flip poisoning 20% (FedAvg, no defense)
         rounds=8 final acc=0.5000 (0.3s)
[fedsec] demo: sign-flip poisoning 20% (trimmed mean)
         rounds=8 final acc=0.9950 (0.3s)

config                                       | attack          | defense     | rounds | final acc | final ASR
---------------------------------------------+-----------------+-------------+--------+-----------+----------
clean (no attack, FedAvg)                    | none            | fedavg      | 8      | 0.9950    | 0.2610
sign-flip poisoning 20% (FedAvg, no defense) | gaussian_poison | fedavg      | 8      | 0.5000    | 1.0000
sign-flip poisoning 20% (trimmed mean)       | gaussian_poison | trimmedmean | 8      | 0.9950    | 0.2560
```

20% of clients send `-5 × update + Gaussian noise` (a sign-flip/gradient-ascent
attack). Plain FedAvg collapses to chance (0.5000); coordinate-wise trimmed mean
recovers the clean baseline exactly (0.9950). The `final ASR` column is the
backdoor attack-success rate — only meaningful for backdoor runs; for other
attacks it is reported for completeness.

`fedsec compare` additionally writes `compare_summary.csv` and matplotlib plots
(`compare.accuracy.png`, `compare.asr.png`).

## Attacks & defenses

| Attack | Mechanism |
|---|---|
| `labelflip` | malicious clients flip all local labels before training (data poisoning) |
| `gaussian_poison` | malicious clients send `scale × update + N(0, noise_std²)`; negative scale = sign-flip (model poisoning) |
| `backdoor` | malicious clients plant a trigger pattern (outlier feature values) correlated with a target label on a fraction of their data |

| Defense | Mechanism |
|---|---|
| `fedavg` | plain mean — the vulnerable baseline |
| `trimmedmean` | coordinate-wise: drop `trim_ratio` from each tail, average the rest |
| `median` | coordinate-wise median |
| `krum` | select the update closest to its `n − f − 2` nearest neighbours (Blanchard et al., 2017) |
| `dp_fedavg` | clip each update to `dp_clip_norm`, average, add Gaussian noise (`dp_noise_multiplier × clip_norm`) |

Configs live in `configs/` (`clean.yaml`, `labelflip_fedavg.yaml`,
`labelflip_trimmedmean.yaml`, `poison_krum.yaml`, `backdoor_dp.yaml`) — every
knob (clients, Dirichlet α, attack/defense, DP clip & noise, rounds…) is a YAML
key.

## Limitations (read before citing this in a paper)

- **This is a simulator, not a security proof.** Results on synthetic Gaussian
  blobs do not transfer to real tasks; robust-aggregation guarantees depend on
  assumptions (bounded fraction of Byzantines, i.i.d.-ish honest updates) that
  real deployments routinely violate.
- **No formal DP accounting.** `dp_fedavg` implements the clipping + Gaussian
  mechanism and reports `(clip_norm, noise_multiplier)`; it does *not* compute
  an (ε, δ) guarantee — that needs a moments/RDP accountant (e.g. Opacus).
- **Toy model and task.** A 2-layer MLP on binary blobs exercises the
  aggregation math, not deep networks, non-IID text/vision data, or adaptive
  attackers that react to the defense.
- **Krum selects one update.** With `krum_m=1` the global model follows a single
  client's update per round — statistically inefficient; it is a robustness
  demo, not a training recipe.

## Roadmap

- [ ] Adaptive attacks (attacker observes the defense, e.g. "a little is enough" scaling)
- [ ] Bulyan / multi-Krum and norm-clipping baselines
- [ ] RDP accountant for honest (ε, δ) reporting in DP-FedAvg
- [ ] Real dataset loaders (FEMNIST-style partitions) behind the same API
- [ ] Async / straggler-tolerant round simulation

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

## License

MIT © 2026 Habib Ur Rehman — see [LICENSE](LICENSE).
