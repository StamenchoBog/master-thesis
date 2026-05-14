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
| **F1** | Harmonic mean of precision and recall |

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
