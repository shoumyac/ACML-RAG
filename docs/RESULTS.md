# Results

All numbers below are from the paper. Per-model JSON reports are in `data/reports/`.

## Selective generation under noise shift (Table 3)

Target risk 0.10, clean-heavy calibration, noise-heavy deployment, 200 resplits, flan-t5-base. Violation is the fraction of resplits whose realized risk exceeds the target.

| Method | Realized risk | Violation | Answer rate |
|---|---:|---:|---:|
| Marginal CRC | 0.215 | 100% | 70.5% |
| Heuristic-group CRC | 0.189 | 100% | 60.9% |
| Learned-group CRC (k-means) | 0.213 | 100% | 67.0% |
| Density-ratio weighted CRC | 0.176 | 100% | 61.5% |
| Worst-group CRC | 0.035 | 22% | 12.7% |
| **Risk-binned Mondrian CRC (ours)** | **0.063** | **0%** | **29.2%** |

A fresh run of `examples/reproduce_headline.py` with a newer scikit-learn lands within a few hundredths of these figures (gradient boosting drifts slightly between versions). The qualitative result is identical: marginal-style rules violate in every resplit, risk-binned certification in about 0 to 1%.

## Risk estimator (AUROC, flan-t5-base)

| Signal set | AUROC |
|---|---:|
| Combined (noise-conditional) | 0.915 |
| Retrieval-side descriptors | 0.897 |
| Model + contrast | 0.811 |
| Model-internal | 0.788 |
| Closed-book contrast | 0.700 |

Combined minus model-internal: +0.127 (95% CI 0.115 to 0.140). Combined beats sampling-based semantic entropy by 0.31 to 0.40 AUROC across seven instruction-tuned generators.

## Breadth and transfer

Nine generators (80M to 14B parameters, five model families, two architectures) and two content domains. AUROC ranges from 0.61 to 0.92 across generators, while calibration does not transfer: realized risk is 1.4 to 3.2 times the target, so the risk score should be recalibrated locally.

## Further write-ups

- [`MULTIFAMILY_RESULTS.md`](MULTIFAMILY_RESULTS.md)
- [`NATURALISTIC_RESULTS.md`](NATURALISTIC_RESULTS.md)
- [`NATURAL_AND_DECOMP_FINDINGS.md`](NATURAL_AND_DECOMP_FINDINGS.md)
