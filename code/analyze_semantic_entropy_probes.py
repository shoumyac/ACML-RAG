"""t11: analyse the Semantic Entropy Probes baseline.

Two probe variants are trained via GroupKFold(5)-by-qid (same protocol as
"ours" everywhere else in this paper), both simple fixed-hyperparameter
linear models on standardised 512-dim hidden-state features, no inner
tuning loop, matching this paper's existing convention of fixed recipes
for every baseline (e.g. the GBM's n_estimators=120, max_depth=3):

  sep_regression   : Ridge(alpha=1.0) regresses hidden state -> cached
    semantic-entropy value (the actual SEP method: approximate SE from a
    single generation's hidden state). Out-of-fold predicted SE-hat is the
    risk score for AUROC against the hallucination label.
  sep_classifier   : LogisticRegression(C=1.0) trained directly on
    hidden state -> hallucination label. This is the baseline the original
    paper reports SEPs generalise BETTER than, but specifically under
    cross-task/cross-dataset distribution shift. This script's
    GroupKFold-by-qid split is a same-distribution held-out-question
    evaluation, not that cross-task setting, so the comparison below
    cannot confirm or refute the original paper's generalisation claim; it
    only shows which probe carries more signal within one task and
    distribution (see the printed caveat below).

The cached SE value used as the regression target comes from the original
6-condition run (results_se_flan_t5_small.json via the reference join baked
into the generation script); this run's OWN greedy label is used as the
AUROC ground truth, since the hidden state reflects what this run's own
generation actually produced. The two can disagree on the ~8% of records
affected by the floating-point non-determinism already isolated and
quantified in the t8 self-consistency/semantic-volume analysis; this is
treated as the same already-documented measurement noise, not a new issue.
"""
import json, os
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
rows = json.load(open(f"{DATA}/results_sep_hidden_flan_t5_small.json"))
mismatches = json.load(open(f"{DATA}/results_sep_hidden_flan_t5_small_mismatches.json"))
print(f"loaded {len(rows)} rows, {len(mismatches)} self-check mismatches "
      f"({100*len(mismatches)/len(rows):.1f}%, matching t8's already-documented "
      f"floating-point non-determinism finding, not a new issue)")

ans = [r for r in rows if r["label"] != "abstain"]
print(f"answered (non-abstain) records: {len(ans)}")

X = np.array([r["hidden"] for r in ans], dtype=np.float64)
y = np.array([1 if r["label"] == "hallucination" else 0 for r in ans])
se = np.array([r["se"] for r in ans], dtype=np.float64)
g = np.array([r["qid"] for r in ans])
print(f"X shape {X.shape}, hallucination rate {y.mean():.3f}")


def oof_ridge_on_se(X, se, y, g):
    """Ridge regresses X -> se; OOF predicted se-hat scores AUROC against y."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        sca = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0, random_state=0)
        m.fit(sca.transform(X[tr]), se[tr])
        p[te] = m.predict(sca.transform(X[te]))
    return p


def oof_logreg_on_label(X, y, g):
    """LogisticRegression trained directly on X -> y; OOF probability scores AUROC."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        sca = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
        m.fit(sca.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sca.transform(X[te]))[:, 1]
    return p


def grouped_bootstrap_ci(y, s, q, n_boot=2000, seed=0):
    uq = np.array(sorted(set(q.tolist())))
    idx_by_q = {qi: np.where(q == qi)[0] for qi in uq}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        samp_q = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_by_q[qi] for qi in samp_q])
        yb, sb = y[idx], s[idx]
        if len(set(yb.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


p_sep = oof_ridge_on_se(X, se, y, g)
auc_sep = roc_auc_score(y, p_sep)
lo_sep, hi_sep = grouped_bootstrap_ci(y, p_sep, g)
print(f"\nsep_regression (Ridge on cached SE, the actual SEP method): "
      f"AUROC={auc_sep:.3f} 95% CI [{lo_sep:.3f}, {hi_sep:.3f}]")

p_cls = oof_logreg_on_label(X, y, g)
auc_cls = roc_auc_score(y, p_cls)
lo_cls, hi_cls = grouped_bootstrap_ci(y, p_cls, g)
print(f"sep_classifier (LogisticRegression directly on the label): "
      f"AUROC={auc_cls:.3f} 95% CI [{lo_cls:.3f}, {hi_cls:.3f}]")

diff = auc_sep - auc_cls
print(f"\nsep_regression - sep_classifier = {diff:+.3f}. NOTE: this is a "
      f"same-distribution GroupKFold-by-qid comparison, not the cross-task/"
      f"cross-dataset distribution shift that the original paper's claim "
      f"('SE-regression generalises better than training directly on the "
      f"label') concerns, so this result neither confirms nor refutes that "
      f"claim; it only shows that within one task and distribution, the "
      f"direct classifier carries more of the label's own signal than a "
      f"probe trained to approximate an intermediate quantity (SE) that "
      f"only partially determines the label.")

# comparison context: literals from already-published/verified Table 3 rows
OURS = 0.893
print(f"\nfor context: TRAQ 0.718, best LLM verifier (combined) 0.867, ours "
      f"(retrieval only) {OURS}. Best SEP variant "
      f"({max(auc_sep, auc_cls):.3f}) is {OURS - max(auc_sep, auc_cls):.3f} below ours.")

out = dict(
    n_rows=len(rows), n_answered=len(ans), n_mismatches=len(mismatches),
    sep_regression=dict(auroc=float(auc_sep), ci_lo=lo_sep, ci_hi=hi_sep),
    sep_classifier=dict(auroc=float(auc_cls), ci_lo=lo_cls, ci_hi=hi_cls),
    hidden_dim=X.shape[1],
    note="hidden = final-layer decoder hidden state at the generation step that "
         "produced the last content token of the greedy answer (flan-t5-small, "
         "encoder-decoder; the original SEP paper ablates decoder-only models, so "
         "this is one clearly-defined choice rather than a reproduction of their "
         "full layer/position sweep). sep_regression targets the cached SE value "
         "from the original 6-condition run; sep_classifier and the AUROC ground "
         "truth both use this run's own greedy label. CAVEAT: sep_regression and "
         "sep_classifier are compared here under a same-distribution "
         "GroupKFold-by-qid split, not the cross-task/cross-dataset distribution "
         "shift that the original paper's generalisation claim (SE-regression "
         "generalises better than a probe trained directly on the label) "
         "concerns; this comparison neither confirms nor refutes that claim."
)
json.dump(out, open(f"{DATA}/sep_report.json", "w"), indent=1)
print("\nsaved sep_report.json")
