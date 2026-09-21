# Experiment Results

Dataset: [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets), 23 CSV files, ~22M rows, 15 features. 3 nodes, 10 rounds.

Each node trains locally for 1 epoch per round with Adam (lr=0.001, batch size 512). The learning rate comes from the server each round, so it can be changed without rebuilding the client image.

## Metrics

Train loss (BCE) shows how fast training is converging — spikes usually mean a client drifted or something stalled. Accuracy is reported for completeness but isn't very informative here since the classes are imbalanced. Precision and recall matter more: low precision means more false alarms, low recall means missed attacks, which for an IDS is the worse failure. F1 is just the harmonic mean of the two.

Recall is the metric we care about most, then precision, then F1 — a missed attack is worse than an extra alert.

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

## FedAvg with chaos engineering — 10 rounds

Same setup, but with 5ms ± 3ms latency injected via toxiproxy (simulating inter-VLAN routing on a wired LAN). If fewer than `min_fit_clients` respond in a round, the server just keeps the previous round's weights instead of aggregating an incomplete update.

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

### Baseline vs chaos

| | Baseline | Chaos |
|---|---|---|
| Best F1 | 96.3% | 96.3% |
| Best Recall | 99.7% | 100.0% |
| Final round F1 | 96.3% | 96.3% |
| Final round Recall | 99.7% | 100.0% |

Ends up at basically the same quality despite the added network noise. Rounds 3, 6 and 7 dip (~68-74% F1) where the fallback kicked in and held the previous weights, but the model recovers every time.
