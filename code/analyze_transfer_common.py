"""Cross-model risk-estimator transfer, restricted to the 12 features common
to every generator in the testbed (6 retrieval descriptors + 6 model-internal
descriptors). Excludes conflict/n_claims/d_lp/d_ent/ans_changed, which only
the flan-t5 pair's original harness computed, so every pair reported here
uses an identical feature set -- including a recomputation of the original
flan-t5-small/flan-t5-base pair under the common schema, for consistency
with the newly added pairs.

Methodology matches analyze_v2.py's cross-model-transfer step exactly:
- fit a full (non-cross-validated) GradientBoostingClassifier on ALL of the
  source model's answered rows (label != abstain) using the 12-feature vector
- predict_proba on ALL of the target model's answered rows; AUROC against
  the target's true labels
- the deployed decision threshold t = crc_threshold, computed IN-SAMPLE on
  the source model's own predictions at alpha=0.10 (a Laplace-corrected
  marginal conformal-style threshold, not the more conservative risk-binned
  certification used elsewhere in the paper)
- realized risk and answer rate/coverage = eval_rule(target_probs, target_
  labels, t)

Usage: python3 analyze_transfer_common.py
Loads release/data/controlled/results_{TAG}.json for each tag in PAIRS.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "controlled"))

RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "has_ctx"]
MOD = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
ALL = RET + MOD
ALPHA = 0.10

PAIRS = [
    ("flan_t5_small", "flan_t5_base"),
    ("flan_t5_base", "qwen2_5_7b_instruct"),
    ("qwen2_5_1_5b_instruct", "qwen2_5_14b_instruct"),
    ("mistral_7b_instruct_v0_3", "qwen2_5_7b_instruct"),
    ("gemma_2_9b_it", "qwen2_5_14b_instruct"),
]

def load(tag):
    return json.load(open(f"{DATA}/results_{tag}.json"))

def answered(rows):
    return [r for r in rows if r["label"] != "abstain"]

def arr(rs, keys):
    return np.array([[r[k] for k in keys] for r in rs])

def lab(rs):
    return np.array([1 if r["label"] == "hallucination" else 0 for r in rs])

def crc_threshold(scores, labels, alpha, weights=None):
    if weights is None:
        weights = np.ones(len(scores))
    order = np.argsort(scores)
    s, l, w = scores[order], labels[order], weights[order]
    cw = np.cumsum(w); cl = np.cumsum(w * l)
    ok = (cl + 1) / (cw + 1) <= alpha
    if not ok.any():
        return -np.inf
    return s[np.where(ok)[0].max()]

def eval_rule(scores, labels, t):
    m = scores <= t
    if m.sum() == 0:
        return 0.0, 0.0
    return float(labels[m].mean()), float(m.mean())

def transfer(tagA, tagB):
    rowsA = answered(load(tagA)); yA = lab(rowsA)
    rowsB = answered(load(tagB)); yB = lab(rowsB)
    out = {}
    for (src_rows, src_y, tgt_rows, tgt_y, nm) in [
        (rowsA, yA, rowsB, yB, f"{tagA}->{tagB}"),
        (rowsB, yB, rowsA, yA, f"{tagB}->{tagA}"),
    ]:
        m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
        m.fit(arr(src_rows, ALL), src_y)
        p_tgt = m.predict_proba(arr(tgt_rows, ALL))[:, 1]
        auc = roc_auc_score(tgt_y, p_tgt)
        p_src = m.predict_proba(arr(src_rows, ALL))[:, 1]
        t = crc_threshold(p_src, src_y, ALPHA)
        risk, cov = eval_rule(p_tgt, tgt_y, t)
        out[nm] = dict(auc=float(auc), risk=float(risk), coverage=float(cov),
                        threshold=float(t), n_src_answered=len(src_y), n_tgt_answered=len(tgt_y))
        print(f"{nm:45s} AUROC={auc:.3f}  risk={risk:.3f}  coverage={cov:.3f}  "
              f"(n_src={len(src_y)}, n_tgt={len(tgt_y)})")
    return out

if __name__ == "__main__":
    report = {}
    for a, b in PAIRS:
        print(f"\n=== {a} <-> {b} ===")
        report.update(transfer(a, b))
    json.dump(report, open(f"{DATA}/transfer_common_report.json", "w"), indent=2)
    print("\nsaved transfer_common_report.json")
