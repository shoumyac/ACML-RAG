"""Conformal risk control for selective generation.

Two selection rules are provided:

* :func:`marginal_threshold`: the standard split-conformal-style risk-control
  threshold. It assumes calibration and deployment are exchangeable.
* :class:`RiskBinnedCRC`: the risk-binned Mondrian procedure from the paper. It
  certifies score bins one by one using a Clopper-Pearson upper bound, so it
  stays valid when deployment only reweights bins (bin-measurable shift).

Conventions: ``scores`` are estimated hallucination risks in [0, 1] (lower is
safer), ``labels`` are 1 for a hallucination and 0 for a correct answer, and a
query is *answered* when it is accepted by the rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.stats import beta as _beta


def clopper_pearson_upper(k: int, n: int, delta: float = 0.10) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate at level 1 - delta.

    ``k`` hallucinations among ``n`` answers. Returns 1.0 when ``n == 0`` or
    ``k == n`` (nothing can be certified).
    """
    if n == 0 or k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


def marginal_threshold(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """Largest score threshold whose finite-sample-corrected risk is at most ``alpha``.

    Answer a query iff ``score <= threshold``. Returns ``-inf`` (answer nothing)
    when no threshold qualifies.
    """
    order = np.argsort(scores)
    s, l = np.asarray(scores)[order], np.asarray(labels)[order]
    cum_bad = np.cumsum(l)
    cum_n = np.arange(1, len(s) + 1)
    ok = (cum_bad + 1) / (cum_n + 1) <= alpha
    return float(s[np.where(ok)[0].max()]) if ok.any() else -np.inf


@dataclass
class BinCertificate:
    """Calibration summary for one score bin."""
    index: int
    low: float
    high: float
    n: int
    n_hallucinations: int
    upper_bound: float
    certified: bool


@dataclass
class RiskBinnedCRC:
    """Risk-binned Mondrian conformal risk control.

    Calibration scores are split into ``n_bins`` quantile bins. A bin is
    certified when the Clopper-Pearson upper bound (level ``1 - delta``) on its
    hallucination rate is at most ``alpha``. At deployment a query is answered
    iff its score falls in a certified bin. With probability at least
    ``1 - n_bins * delta`` over the calibration draw, the hallucination rate
    among answered queries is at most ``alpha`` for every deployment
    distribution that only reweights bins (the paper's Proposition 1).
    """
    alpha: float = 0.10
    n_bins: int = 10
    delta: float = 0.10
    edges_: np.ndarray = field(default=None, init=False, repr=False)
    certificates_: List[BinCertificate] = field(default_factory=list, init=False, repr=False)

    def fit(self, scores, labels) -> "RiskBinnedCRC":
        pc, yc = np.asarray(scores, float), np.asarray(labels, int)
        qs = np.quantile(pc, np.linspace(0, 1, self.n_bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        self.edges_ = qs
        self.certificates_ = []
        for b in range(self.n_bins):
            m = (pc > qs[b]) & (pc <= qs[b + 1])
            n = int(m.sum())
            k = int(yc[m].sum()) if n else 0
            ub = clopper_pearson_upper(k, n, self.delta) if n else 1.0
            self.certificates_.append(
                BinCertificate(b, float(qs[b]), float(qs[b + 1]), n, k, ub, bool(n > 0 and ub <= self.alpha)))
        return self

    @property
    def certified_bins_(self) -> List[int]:
        return [c.index for c in self.certificates_ if c.certified]

    def bin_of(self, scores) -> np.ndarray:
        """Bin index for each score (-1 or ``n_bins`` when outside the calibration range)."""
        return np.digitize(np.asarray(scores, float), self.edges_) - 1

    def answer_mask(self, scores) -> np.ndarray:
        """Boolean mask: True where the query is answered, False where it abstains."""
        return np.isin(self.bin_of(scores), self.certified_bins_)

    def evaluate(self, scores, labels) -> Tuple[float, float]:
        """Return (realized risk among answered, answer rate) on deployment data."""
        acc = self.answer_mask(scores)
        y = np.asarray(labels, int)
        if acc.sum() == 0:
            return 0.0, 0.0
        return float(y[acc].mean()), float(acc.mean())

    def summary(self) -> str:
        lines = ["bin   range                 n   halluc   CP upper   certified"]
        for c in self.certificates_:
            lines.append(f"{c.index:>3}   [{c.low:6.3f}, {c.high:6.3f}]  {c.n:>5}  {c.n_hallucinations:>6}   "
                         f"{c.upper_bound:8.3f}   {'yes' if c.certified else 'no'}")
        return "\n".join(lines)


def decertify_on_drift(crc: RiskBinnedCRC, cal_scores, cal_flag, dep_scores, dep_flag) -> List[int]:
    """Label-free deployment monitor.

    ``flag`` is a binary covariate observable without labels (the paper uses the
    inter-document conflict indicator). A certified bin is flagged when its
    deployment flag rate exceeds its calibration flag rate, which signals a
    within-bin shift that the bin-measurable assumption does not cover.
    Returns the list of flagged (to be decertified) bin indices.
    """
    cb, db = crc.bin_of(cal_scores), crc.bin_of(dep_scores)
    cal_flag, dep_flag = np.asarray(cal_flag, float), np.asarray(dep_flag, float)
    flagged = []
    for b in crc.certified_bins_:
        cm, dm = cb == b, db == b
        if cm.sum() == 0 or dm.sum() == 0:
            continue
        if dep_flag[dm].mean() > cal_flag[cm].mean():
            flagged.append(b)
    return flagged
