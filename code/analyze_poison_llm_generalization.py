"""t10: does risk-binned certification, calibrated only on the six original
conditions, still control risk when deployment includes the new poison_llm
condition it never saw during calibration or estimator training? This is the
same question the existing within-bin-shift stress test (S5.6) and the
PoisonedRAG-inclusive naturalistic validation (S5.9) ask, extended to a
genuinely different poisoning STYLE rather than a different mixture of
already-seen conditions.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
orig = json.load(open(f"{DATA}/results_se_flan_t5_small.json"))
new = json.load(open(f"{DATA}/results_poison_llm_flan_t5_small.json"))
print(f"original: {len(orig)} rows, poison_llm: {len(new)} rows")

RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "has_ctx"]
ans_o = [r for r in orig if r["label"] != "abstain"]
y_o = np.array([1 if r["label"] == "hallucination" else 0 for r in ans_o])
g_o = np.array([r["qid"] for r in ans_o])
X_o = np.array([[r[k] for k in RET] for r in ans_o])

ans_n = [r for r in new if r["label"] != "abstain"]
y_n = np.array([1 if r["label"] == "hallucination" else 0 for r in ans_n])
g_n = np.array([r["qid"] for r in ans_n])
X_n = np.array([[r[k] for k in RET] for r in ans_n])
print(f"poison_llm answered: {len(y_n)}, hallucination rate: {y_n.mean():.3f}")

# OOF scores on the original 6 conditions (identical recipe to analyze_baselines.py)
def oof(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
p_o = oof(X_o, y_o, g_o)
auc_o = roc_auc_score(y_o, p_o)
print(f"ours_retrieval OOF AUROC on the original 6 conditions: {auc_o:.3f} (sanity check vs Table 3's 0.893)")

# a single model fit on ALL original-condition data scores poison_llm, which
# was never in any training fold to begin with -- no leakage either way.
m_full = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
m_full.fit(X_o, y_o)
p_n = m_full.predict_proba(X_n)[:, 1]
if len(set(y_n.tolist())) > 1:
    auc_n = roc_auc_score(y_n, p_n)
    print(f"AUROC on poison_llm (never trained on): {auc_n:.3f} (n={len(y_n)}, only "
          f"{int((1-y_n).sum())} non-hallucination examples, so treat this number as "
          f"low-powered; the risk-score LEVEL check below is the more informative one)")
print(f"risk score p_ours on poison_llm: mean={p_n.mean():.3f} min={p_n.min():.3f} max={p_n.max():.3f}")
print(f"for comparison, mean p_ours by original condition:")
for c in sorted(set(r["cond"] for r in ans_o)):
    mask = np.array([r["cond"] for r in ans_o]) == c
    print(f"  {c:14s} mean_score={p_o[mask].mean():.3f} halluc_rate={y_o[mask].mean():.3f}")

# certification: calibrate on a random half of ORIGINAL qids (clean-heavy,
# matching the paper's own resplit convention), deploy on the other half of
# original qids PLUS ALL of poison_llm (entirely unseen at calibration time).
ALPHA = 0.10
def cp_upper(k, n, d=0.10):
    if n == 0: return 1.0
    return float(beta_dist.ppf(1 - d, k + 1, n - k)) if k < n else 1.0
def riskbin_thresholds(pc, yc, a, nb=8):
    qs = np.quantile(pc, np.linspace(0, 1, nb + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cert = [b for b in range(nb) if ((pc > qs[b]) & (pc <= qs[b + 1])).sum() > 0
            and cp_upper(int(yc[(pc > qs[b]) & (pc <= qs[b + 1])].sum()),
                         int(((pc > qs[b]) & (pc <= qs[b + 1])).sum())) <= a]
    return qs, cert
def apply_bins(pt, qs, cert):
    tb = np.digitize(pt, qs) - 1
    return np.isin(tb, cert)

orig_qids = np.array(sorted(set(g_o)))
rng_master = np.random.default_rng(1010)
results = []
for rep in range(150):
    rr = np.random.default_rng(3000 + rep)
    sh = rr.permutation(orig_qids)
    calq = set(sh[:len(orig_qids) // 2]); teq = set(sh[len(orig_qids) // 2:])
    cal = np.array([q in calq for q in g_o]); te_orig = np.array([q in teq for q in g_o])
    if cal.sum() < 30 or te_orig.sum() < 30: continue
    qs, cert = riskbin_thresholds(p_o[cal], y_o[cal], ALPHA)
    # deployment mix: held-out original test half + ALL of poison_llm
    pt_mix = np.concatenate([p_o[te_orig], p_n])
    yt_mix = np.concatenate([y_o[te_orig], y_n])
    acc = apply_bins(pt_mix, qs, cert)
    if acc.sum() == 0:
        results.append((0.0, 0.0)); continue
    results.append((float(yt_mix[acc].mean()), float(acc.mean())))
    # also isolate poison_llm-only accepted-rate/risk within this same calibration
    acc_n = apply_bins(p_n, qs, cert)

res = np.array(results)
print(f"\n=== Certification under deployment mix: held-out original half + ALL poison_llm "
      f"(alpha={ALPHA}, 150 resplits) ===")
print(f"realized risk: {res[:,0].mean():.3f}   violation: {(res[:,0]>ALPHA).mean()*100:.0f}%   "
      f"answer rate: {res[:,1].mean()*100:.1f}%")

# poison_llm-only breakdown using ONE representative calibration (first valid resplit)
qs, cert = riskbin_thresholds(p_o[np.array([q in set(orig_qids[:len(orig_qids)//2]) for q in g_o])],
                               y_o[np.array([q in set(orig_qids[:len(orig_qids)//2]) for q in g_o])], ALPHA)
acc_n_only = apply_bins(p_n, qs, cert)
print(f"\npoison_llm alone, evaluated against bins calibrated only on original conditions:")
print(f"  accepted (answered): {acc_n_only.sum()}/{len(y_n)} ({100*acc_n_only.mean():.1f}%)")
if acc_n_only.sum() > 0:
    print(f"  realized risk among accepted: {y_n[acc_n_only].mean():.3f}")

out = dict(
    original_ours_retrieval_auroc=float(auc_o),
    poison_llm_halluc_rate=float(y_n.mean()),
    poison_llm_mean_risk_score=float(p_n.mean()),
    certification_mixed_deployment=dict(realized_risk=float(res[:,0].mean()),
                                          violation_pct=float((res[:,0]>ALPHA).mean()*100),
                                          answer_rate_pct=float(res[:,1].mean()*100)),
    poison_llm_alone_accept_rate_pct=float(100*acc_n_only.mean()),
    note="poison_llm condition and its risk scores were NEVER used to fit the GBM or "
         "the certification bins; this is genuine out-of-distribution generalisation, "
         "not cross-validation on data the estimator was trained on."
)
json.dump(out, open(f"{DATA}/poison_llm_generalization_report.json", "w"), indent=1)
print("\nsaved poison_llm_generalization_report.json")
