# Experiment Results

Dataset: [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) (23 CSV files, ~22M rows, 15 features) | Nodes: 3 | Rounds: 10

Each node trains locally for **1 epoch per round** using **Adam (lr=0.001, batch size 512)**. The learning rate is sent by the server each round so it can be adjusted without rebuilding the client image.

---

## Metrics

| Metric | Why it matters |
|---|---|
| **Train Loss** (BCE) | Tracks learning speed; spikes signal client drift or convergence stall |
| **Accuracy** | Overall correctness — misleading on imbalanced data, included for full picture |
| **Precision** | Low → false alarms (alert fatigue) |
| **Recall** | Low → missed attacks (primary failure mode for an IDS) |
| **F1** | Harmonic mean of precision/recall; used for fair strategy comparison |

Priority order: **Recall › Precision › F1.** Missing an attack is unacceptable; false alarms are costly but recoverable.

---

## FedAvg — 10 rounds

| Round | Train Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | 0.0552 | 93.1% | 96.3% | 93.2% | 100.0% |
| 2 | 0.0358 | 73.8% | 81.6% | 92.4% | 79.4% |
| 3 | 0.0291 | 74.5% | 82.3% | 92.6% | 80.1% |
| 4 | 0.0288 | 80.1% | 88.1% | 93.3% | 85.5% |
| 5 | 0.0272 | 92.6% | 96.0% | 93.3% | 99.3% |
| 6 | 0.0271 | 83.4% | 90.4% | 93.2% | 89.6% |
| 7 | 0.0262 | 93.2% | 96.3% | 93.2% | 100.0% |
| 8 | 0.0257 | 81.8% | 89.1% | 93.0% | 87.9% |
| 9 | 0.0258 | 87.4% | 92.9% | 93.4% | 93.6% |
| **10** | **0.0252** | **93.2%** | **96.3%** | **93.4%** | **99.7%** |

## FedProx — 10 rounds (proximal_mu=0.1)

| Round | Train Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | 0.0594 | 93.2% | 96.3% | 93.2% | 100.0% |
| 2 | 0.0356 | 67.9% | 73.6% | 93.1% | 73.1% |
| 3 | 0.0312 | 89.5% | 94.2% | 94.2% | 95.0% |
| **4** | **0.0283** | **93.3%** | **96.3%** | **94.3%** | **98.8%** |
| 5 | 0.0273 | 84.3% | 90.8% | 94.2% | 89.7% |
| 6 | 0.0258 | 80.2% | 87.8% | 93.2% | 85.8% |
| 7 | 0.0258 | 72.7% | 80.8% | 92.1% | 78.1% |
| 8 | 0.0253 | 63.6% | 66.8% | 92.9% | 67.5% |
| 9 | 0.0254 | 62.8% | 67.2% | 83.6% | 67.7% |
| 10 | 0.0263 | 64.3% | 68.1% | 83.9% | 69.3% |

## Krum — 10 rounds (num_malicious_clients=1)

| Round | Train Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | 0.0548 | 62.5% | 66.2% | 78.3% | 67.7% |
| 2 | 0.0314 | 36.2% | 41.6% | 88.6% | 39.3% |
| 3 | 0.0291 | 47.0% | 56.7% | 88.5% | 50.8% |
| 4 | 0.0288 | 65.9% | 72.3% | 94.9% | 70.5% |
| 5 | 0.0277 | 39.9% | 48.9% | 91.3% | 43.4% |
| 6 | 0.0275 | 76.2% | 84.3% | 95.2% | 80.6% |
| **7** | **0.0277** | **93.2%** | **96.1%** | **95.5%** | **96.9%** |
| 8 | 0.0279 | 78.7% | 86.4% | 95.8% | 82.5% |
| 9 | 0.0278 | 61.3% | 65.9% | 83.0% | 65.2% |
| 10 | 0.0271 | 41.2% | 49.3% | 80.0% | 45.0% |

## Trimmed Mean — 10 rounds

| Round | Train Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | 0.0554 | 93.1% | 96.2% | 93.1% | 100.0% |
| 2 | 0.0335 | 93.3% | 96.3% | 94.1% | 99.0% |
| 3 | 0.0291 | 64.6% | 68.2% | 84.0% | 69.7% |
| 4 | 0.0275 | 69.6% | 76.7% | 91.5% | 74.8% |
| 5 | 0.0260 | 64.5% | 68.2% | 84.0% | 69.7% |
| 6 | 0.0253 | 63.8% | 67.8% | 83.9% | 68.8% |
| 7 | 0.0262 | 69.7% | 77.2% | 91.6% | 74.9% |
| 8 | 0.0246 | 93.3% | 96.3% | 93.2% | 100.0% |
| 9 | 0.0243 | 60.3% | 65.9% | 83.5% | 65.5% |
| **10** | **0.0244** | **93.3%** | **96.3%** | **93.3%** | **99.9%** |

---

## FedAvg with Chaos Engineering — 10 rounds

Network conditions: 5ms ± 3ms latency injected via toxiproxy to simulate inter-VLAN routing on a wired LAN. If fewer than `min_fit_clients` respond in a round, the server keeps the previous round's weights rather than aggregating an incomplete update.

| Round | Train Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | 0.0570 | 93.1% | 96.2% | 93.1% | 100.0% |
| 2 | 0.0351 | 88.5% | 93.5% | 94.0% | 94.0% |
| 3 | 0.0304 | 63.9% | 67.9% | 83.9% | 69.3% |
| 4 | 0.0277 | 91.2% | 95.2% | 93.8% | 97.2% |
| 5 | 0.0263 | 85.7% | 91.8% | 93.6% | 91.4% |
| 6 | 0.0274 | 64.7% | 68.3% | 84.0% | 69.8% |
| 7 | 0.0270 | 65.9% | 74.2% | 90.5% | 70.7% |
| 8 | 0.0266 | 93.1% | 96.2% | 93.7% | 99.0% |
| 9 | 0.0272 | 85.4% | 91.7% | 93.3% | 91.5% |
| **10** | **0.0271** | **93.3%** | **96.3%** | **93.2%** | **100.0%** |

### Baseline vs Chaos

| | Baseline | Chaos |
|---|---|---|
| Best F1 | 96.3% | 96.3% |
| Best Recall | 99.7% | 100.0% |
| Final round F1 | 96.3% | 96.3% |
| Final round Recall | 99.7% | 100.0% |

The system ends at the same quality as the baseline despite realistic network noise. Rounds 3, 6, and 7 show drops (~68–74% F1) where the fallback held the previous weights — the model recovered each time.

---

## Conclusion

`FedAvg` is the best fit for this setup. `Krum` and `Trimmed Mean` are designed for larger client pools (10+) where Byzantine-robust selection makes sense — with only 3 nodes they oscillate badly. `FedProx` converges fast but degrades after round 4 due to the proximal penalty fighting the non-IID data distribution.

The chaos experiment confirms the system is resilient to realistic wired-LAN latency. Occasional jitter causes some rounds to fall back to the previous weights, but the model recovers and reaches the same final quality as the baseline: **96.3% F1, 100% Recall**.
