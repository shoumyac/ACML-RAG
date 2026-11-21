# Method

## Setting

For each query a RAG system retrieves passages and generates an answer. Each answer is labelled correct, hallucination or abstain. We want a rule that answers a query only when the hallucination rate among answered queries stays at most `alpha`.

## 1. Noise-conditional risk estimator

Gradient-boosted trees (150 trees, depth 3) estimate `r(x) = P(hallucination | x)` from three observable signal families:

| Family | Features |
|---|---|
| Retrieval descriptors | mean, top-1, margin, min and std of retrieval scores; inter-document conflict flag; number of claims; whether any context was retrieved |
| Model-internal | mean and min log-probability; mean, max and first-token entropy; answer length |
| Closed-book contrast | change in log-probability and entropy when the context is removed; whether the answer changed |

Evaluation uses question-grouped five-fold cross-validation, so no question is in both a training and a test fold (`ncsrag.out_of_fold_risk`).

## 2. Why standard conformal risk control breaks

Marginal CRC picks a score threshold on calibration data so the finite-sample-corrected risk is at most `alpha`. That is valid only if calibration and deployment are exchangeable. Retrieval-noise shift changes the mixture of noise regimes, so it is not.

## 3. Risk-binned Mondrian CRC

1. Split the calibration score range into `B` quantile bins.
2. For each bin compute the Clopper-Pearson upper bound at level `1 - delta` on its hallucination rate.
3. Certify a bin when that bound is at most `alpha`.
4. At deployment, answer iff the score falls in a certified bin.

The rule never extrapolates a threshold into score regions it has not measured.

**Assumption (bin-measurable shift).** Deployment `Q` and calibration `P` satisfy `Q(y | x) = P(y | x)` and the likelihood ratio `dQ/dP(x)` is constant on each bin.

**Proposition.** Under the assumption, with probability at least `1 - B * delta` over the calibration draw, the hallucination rate among answered deployment queries is at most `alpha`, simultaneously for every such `Q`. Deployment may reweight bins arbitrarily, including adversarially toward high-risk bins, but may not move mass within a bin.

## 4. Deployment monitor

The one failure mode is a within-bin shift that raises a certified bin's risk without moving its scores. A label-free monitor decertifies a bin when a held-out observable covariate (the conflict flag) occurs more often at deployment than at calibration (`ncsrag.decertify_on_drift`). In the paper it flags 4.37 bins on average under an adversarial within-bin shift against 0.24 under benign reweighting.

## Implementation map

| Paper | Code |
|---|---|
| Risk estimator | `src/ncsrag/estimator.py` |
| Risk-binned Mondrian CRC | `src/ncsrag/conformal.py` (`RiskBinnedCRC`) |
| Marginal CRC baseline | `src/ncsrag/conformal.py` (`marginal_threshold`) |
| Noise-shift protocol | `src/ncsrag/protocol.py` |
| Other baselines (density-ratio, learned-group, worst-group), ablations | `code/analyze_v2.py`, `code/analyze_advanced.py`, `code/analyze_baselines.py` |
