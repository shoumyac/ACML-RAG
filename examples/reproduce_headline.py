"""Reproduce the paper's headline result (Table 3) from the shipped data in ~1 minute.

    python examples/reproduce_headline.py

Paper (Table 3; flan-t5-base, 200 resplits, target risk 0.10):
    marginal CRC          risk 0.215   violation 100%   answer rate 70.5%
    risk-binned Mondrian  risk 0.063   violation   0%   answer rate 29.2%
Gradient boosting drifts slightly across scikit-learn versions, so a fresh run lands
within a few hundredths of these (marginal violates in 100% of resplits, risk-binned
in about 0 to 1%, answering roughly 30% of queries).
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from ncsrag import (ALL_FEATURES, CONTRAST_FEATURES, MODEL_FEATURES, RETRIEVAL_FEATURES, answered_rows,
                    hallucination_labels, out_of_fold_risk, run_noise_shift)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "controlled" / "results_flan_t5_base.json"
if not DATA.exists():
    sys.exit(f"missing {DATA}")

rows = answered_rows(json.load(open(DATA)))
y = hallucination_labels(rows)
groups = [r["qid"] for r in rows]
conds = [r["cond"] for r in rows]
print(f"{len(rows)} answered queries, hallucination rate {y.mean():.3f}")

print("\nStep 1. Hallucination-risk estimator (question-grouped 5-fold CV)")
for name, feats in [("retrieval descriptors", RETRIEVAL_FEATURES), ("model-internal", MODEL_FEATURES),
                    ("closed-book contrast", CONTRAST_FEATURES), ("combined (ours)", ALL_FEATURES)]:
    p = out_of_fold_risk(rows, feats)
    print(f"  AUROC {name:<22s} {roc_auc_score(y, p):.3f}")
p_all = out_of_fold_risk(rows, ALL_FEATURES)

print("\nStep 2. Selective generation under noise shift (alpha = 0.10, 200 resplits)")
res = run_noise_shift(p_all, y, groups, conds, alpha=0.10, n_resplits=200)
print(f"  {'method':<24s}{'risk':>8s}{'violation':>12s}{'answer rate':>14s}")
for m, label in [("marginal", "Marginal CRC"), ("riskbin", "Risk-binned Mondrian CRC")]:
    r = res[m]
    print(f"  {label:<24s}{r['risk']:>8.3f}{r['violation']:>11.0%}{r['answer_rate']:>14.1%}")
