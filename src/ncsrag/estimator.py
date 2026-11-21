"""Noise-conditional hallucination-risk estimator.

Combines observable retrieval descriptors, model-internal signals and (optionally)
semantic entropy in a gradient-boosted classifier, evaluated with
question-grouped cross-validation so no question is in both train and test folds.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

RETRIEVAL_FEATURES = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "conflict", "n_claims", "has_ctx"]
MODEL_FEATURES = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
# closed-book contrast: how the answer changes when the retrieved context is removed
CONTRAST_FEATURES = ["d_lp", "d_ent", "ans_changed"]
ALL_FEATURES = RETRIEVAL_FEATURES + MODEL_FEATURES + CONTRAST_FEATURES


def answered_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Drop abstentions: the risk estimator is trained on answered queries only."""
    return [r for r in rows if r["label"] != "abstain"]


def hallucination_labels(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([1 if r["label"] == "hallucination" else 0 for r in rows])


def out_of_fold_risk(rows: Sequence[Dict], features: Sequence[str] = ALL_FEATURES,
                     n_splits: int = 5, n_estimators: int = 150, max_depth: int = 3,
                     seed: int = 0) -> np.ndarray:
    """Out-of-fold P(hallucination) for each answered row (question-grouped CV)."""
    X = np.array([[r[k] for k in features] for r in rows], float)
    y = hallucination_labels(rows)
    groups = np.array([r["qid"] for r in rows])
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=seed, n_estimators=n_estimators, max_depth=max_depth)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p
