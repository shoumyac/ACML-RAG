"""The paper's noise-shift evaluation protocol.

Calibration is clean-heavy (noisy items are kept with probability 0.30), deployment
is noise-heavy (clean items are kept with probability 0.50). Questions are split
into disjoint calibration and deployment halves, repeated over many resplits.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from .conformal import RiskBinnedCRC, marginal_threshold

NOISY_CONDITIONS = {"irr75", "irr100", "contra_r1", "contra_r4", "contra_only", "mixed"}


def _marginal(pc, yc, pt, yt, alpha):
    t = marginal_threshold(pc, yc, alpha)
    m = pt <= t
    return (float(yt[m].mean()), float(m.mean())) if m.sum() else (0.0, 0.0)


def _riskbin(pc, yc, pt, yt, alpha, n_bins=10):
    return RiskBinnedCRC(alpha=alpha, n_bins=n_bins).fit(pc, yc).evaluate(pt, yt)


METHODS: Dict[str, Callable] = {"marginal": _marginal, "riskbin": _riskbin}


def run_noise_shift(scores, labels, groups, conds, alpha: float = 0.10, n_resplits: int = 200,
                    cal_noisy_keep: float = 0.30, dep_clean_keep: float = 0.50,
                    methods: Iterable[str] = ("marginal", "riskbin")) -> Dict[str, Dict[str, float]]:
    """Run the protocol; returns {method: {risk, violation, answer_rate, n}}.

    ``violation`` is the fraction of resplits whose realized risk exceeds ``alpha``.
    """
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    is_noisy = np.array([c in NOISY_CONDITIONS for c in conds])
    qids = sorted(set(groups.tolist()))
    out = {m: [] for m in methods}
    for rep in range(n_resplits):
        rr = np.random.default_rng(5000 + rep)
        sh = rr.permutation(qids)
        half = set(sh[: len(qids) // 2])
        other = set(sh[len(qids) // 2:])
        cal = np.array([q in half for q in groups])
        dep = np.array([q in other for q in groups])
        ck = cal & (~is_noisy | (rr.random(len(cal)) < cal_noisy_keep))
        tk = dep & (is_noisy | (rr.random(len(dep)) < dep_clean_keep))
        if ck.sum() < 50 or tk.sum() < 50:
            continue
        for m in methods:
            out[m].append(METHODS[m](scores[ck], labels[ck], scores[tk], labels[tk], alpha))
    res = {}
    for m, v in out.items():
        a = np.array(v)
        res[m] = dict(risk=float(a[:, 0].mean()), violation=float((a[:, 0] > alpha).mean()),
                      answer_rate=float(a[:, 1].mean()), n=len(a))
    return res
