# MedSure Vision — Evaluation Benchmark Results

> **Note:** Tampered images are synthetic and self-made for internal testing only. Registry results are simulated. These results do not constitute clinical validation.

## Key Performance Metrics

| Metric | Value | Description |
|---|---|---|
| **Total Evaluation Samples** | 100 | Total processed images (20 untampered, 80 tampered) |
| **FALSE-SAFE RATE** | **7.5%** | Fraction of tampered packs mistakenly classified as `low_risk` (6/80) |
| **Abstention Rate** | **80.0%** | Inconclusive cases requiring physical inspection (`needs_verification` + `insufficient_quality`) |
| **p50 Latency (Median)** | **10.3 ms** | Median end-to-end pipeline execution time |
| **p95 Latency** | **4914.68 ms** | 95th percentile latency |
| **Mean Latency** | **1231.53 ms** | Average pipeline execution time |

---

## Confusion Matrix

| True Class | low_risk | needs_verification | high_suspicion | insufficient_quality | Total |
|---|---|---|---|---|---|
| **Untampered** | 2 | 1 | 3 | 14 | 20 |
| **Tampered** | 6 | 3 | 9 | 62 | 80 |
| **Total Predicted** | 8 | 4 | 12 | 76 | 100 |

---

## Per-Class Precision and Recall

| Evaluation Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| **Untampered** (Safe Verification) | 25.0% | 10.0% | 14.29% | 20 |
| **Tampered** (Anomaly Detection) | 80.43% | 92.5% | 86.04% | 80 |

---

## Predicted Label Distribution

| Pipeline Label | Count | Share |
|---|---|---|
| `low_risk` | 8 | 8.0% |
| `needs_verification` | 4 | 4.0% |
| `high_suspicion` | 12 | 12.0% |
| `insufficient_quality` | 76 | 76.0% |

---

## Latency Profile

- **Minimum:** 8.01 ms
- **p50 (Median):** 10.3 ms
- **p95:** 4914.68 ms
- **Maximum:** 13974.61 ms
- **Mean:** 1231.53 ms
