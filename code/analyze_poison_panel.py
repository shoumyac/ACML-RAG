"""t12: does risk-binned certification, calibrated only on the six original
conditions, still control risk when deployment includes ALL FOUR LLM-authored
poisoning styles now tested (t10's poison_llm plus t12's authority, negation,
and corrob), none of which the estimator or the bins ever saw during training
or calibration? This extends analyze_poison_llm_generalization.py (t10, one
style) to the full multi-attack panel, in the spirit of PoisonArena's point
that robustness to one fixed recipe need not generalise to a stylistically
different one (Chen et al., arXiv:2505.12574).

poison_llm's cap1 record (qid=cap1, capital of Japan) is EXCLUDED throughout:
its generator declined to author a false passage and instead wrote a
corrective statement naming the true answer ("I cannot write this passage
... The capital of Japan is Tokyo, not New Delhi."), which t10's original
error check missed since it only tested for non-empty content. Diagnosed
during t12 prep (2026-09-06); see main.tex for the corrected single-style
numbers (117/119 = 98.3%, not 117/120 = 97.5%). This script reproduces that
correction (as the "poison_llm" row below) and folds it into the combined
panel so every artefact this repo ships is internally consistent.

Outputs two files:
  poison_llm_generalization_report.json  -- overwritten with the corrected
    (cap1-excluded) single-style numbers, so this artefact matches the
    corrected paper text instead of the stale 97.5% figure.
  poison_panel_generalization_report.json -- the new full-panel report: per
    style hallucination rate and mean risk score, plus the combined
    mixed-deployment certification test across all four styles at once.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "has_ctx"]
ALPHA = 0.10

orig = json.load(open(f"{DATA}/results_se_flan_t5_small.json"))
print(f"original: {len(orig)} rows")

ans_o = [r for r in orig if r["label"] != "abstain"]
y_o = np.array([1 if r["label"] == "hallucination" else 0 for r in ans_o])
g_o = np.array([r["qid"] for r in ans_o])
X_o = np.array([[r[k] for k in RET] for r in ans_o])

# OOF sanity check against the known-published figure before trusting anything downstream
def oof(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
p_o = oof(X_o, y_o, g_o)
auc_o = roc_auc_score(y_o, p_o)
print(f"ours_retrieval OOF AUROC on the original 6 conditions: {auc_o:.3f} "
      f"(sanity check vs Table 3's 0.893 -- MUST match before trusting anything below)")
assert abs(auc_o - 0.893) < 0.002, f"sanity check FAILED: {auc_o:.4f} vs expected ~0.893"

m_full = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
m_full.fit(X_o, y_o)

def load_style(name, path, exclude_qids=()):
    rows = json.load(open(path))
    rows = [r for r in rows if r["qid"] not in exclude_qids]
    ans = [r for r in rows if r["label"] != "abstain"]
    y = np.array([1 if r["label"] == "hallucination" else 0 for r in ans])
    g = np.array([r["qid"] for r in ans])
    X = np.array([[r[k] for k in RET] for r in ans])
    p = m_full.predict_proba(X)[:, 1] if len(ans) else np.array([])
    return dict(name=name, n=len(ans), y=y, g=g, X=X, p=p)

styles = [
    load_style("poison_llm", f"{DATA}/results_poison_llm_flan_t5_small.json", exclude_qids={"cap1"}),
    load_style("authority", f"{DATA}/results_poison_panel_authority_flan_t5_small.json"),
    load_style("negation", f"{DATA}/results_poison_panel_negation_flan_t5_small.json"),
    load_style("corrob", f"{DATA}/results_poison_panel_corrob_flan_t5_small.json"),
]

print("\n=== per-style hallucination rate and risk score (estimator never trained on any of these) ===")
per_style_out = {}
for s in styles:
    y, p, n = s["y"], s["p"], s["n"]
    halluc_rate = float(y.mean())
    line = (f"{s['name']:12s} n={n:3d}  halluc_rate={halluc_rate:.3f}  "
            f"mean_risk_score={p.mean():.3f}  min={p.min():.3f}  max={p.max():.3f}")
    auc = None
    if len(set(y.tolist())) > 1:
        auc = float(roc_auc_score(y, p))
        line += f"  AUROC={auc:.3f}"
        n_minor = int(min(y.sum(), (1 - y).sum()))
        if n_minor < 15:
            line += f"  (LOW POWER: only {n_minor} in minority class, treat as indicative only)"
    else:
        line += "  AUROC=undefined (single class)"
    print(line)
    per_style_out[s["name"]] = dict(n=n, halluc_rate=halluc_rate, mean_risk_score=float(p.mean()),
                                     min_risk_score=float(p.min()), max_risk_score=float(p.max()),
                                     auroc=auc)

print("\nfor comparison, mean p_ours by original condition:")
for c in sorted(set(r["cond"] for r in ans_o)):
    mask = np.array([r["cond"] for r in ans_o]) == c
    print(f"  {c:14s} mean_score={p_o[mask].mean():.3f} halluc_rate={y_o[mask].mean():.3f}")

# --- certification machinery, identical recipe to analyze_poison_llm_generalization.py ---
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
y_all_new = np.concatenate([s["y"] for s in styles])
p_all_new = np.concatenate([s["p"] for s in styles])
n_total_new = len(y_all_new)
_sizes_str = ", ".join(f"{s['name']}={s['n']}" for s in styles)
print(f"\ncombined panel size (all 4 styles pooled): {n_total_new} answered queries "
      f"({_sizes_str})")

def run_certification(p_new, y_new, label):
    results = []
    acc_new_list = []
    for rep in range(150):
        rr = np.random.default_rng(3000 + rep)
        sh = rr.permutation(orig_qids)
        calq = set(sh[:len(orig_qids) // 2]); teq = set(sh[len(orig_qids) // 2:])
        cal = np.array([q in calq for q in g_o]); te_orig = np.array([q in teq for q in g_o])
        if cal.sum() < 30 or te_orig.sum() < 30: continue
        qs, cert = riskbin_thresholds(p_o[cal], y_o[cal], ALPHA)
        pt_mix = np.concatenate([p_o[te_orig], p_new])
        yt_mix = np.concatenate([y_o[te_orig], y_new])
        acc = apply_bins(pt_mix, qs, cert)
        if acc.sum() == 0:
            results.append((0.0, 0.0))
        else:
            results.append((float(yt_mix[acc].mean()), float(acc.mean())))
        acc_new_list.append(apply_bins(p_new, qs, cert))
    res = np.array(results)
    print(f"\n=== Certification under deployment mix: held-out original half + {label} "
          f"(alpha={ALPHA}, 150 resplits) ===")
    print(f"realized risk: {res[:,0].mean():.3f}   violation: {(res[:,0]>ALPHA).mean()*100:.0f}%   "
          f"answer rate: {res[:,1].mean()*100:.1f}%")
    # new-condition-alone accept rate, averaged across the same 150 resplits' bin choices
    accept_rates = [float(a.mean()) for a in acc_new_list]
    realized_alone = []
    for a in acc_new_list:
        if a.sum() > 0:
            realized_alone.append(float(y_new[a].mean()))
    print(f"{label} alone: mean accept rate {np.mean(accept_rates)*100:.1f}%  "
          f"(realized risk among accepted, when any accepted: "
          f"{np.mean(realized_alone) if realized_alone else float('nan'):.3f}, "
          f"n_resplits_with_any_accept={len(realized_alone)}/{len(acc_new_list)})")
    return dict(realized_risk=float(res[:,0].mean()), violation_pct=float((res[:,0]>ALPHA).mean()*100),
                answer_rate_pct=float(res[:,1].mean()*100),
                new_alone_accept_rate_pct=float(np.mean(accept_rates)*100),
                new_alone_realized_risk_when_any_accepted=(float(np.mean(realized_alone)) if realized_alone else None))

combined = run_certification(p_all_new, y_all_new, "ALL FOUR panel styles combined")
per_style_cert = {}
for s in styles:
    per_style_cert[s["name"]] = run_certification(s["p"], s["y"], s["name"])

# --- write outputs ---
poison_llm_stats = per_style_out["poison_llm"]
poison_llm_cert = per_style_cert["poison_llm"]
out_single = dict(
    original_ours_retrieval_auroc=float(auc_o),
    poison_llm_halluc_rate=poison_llm_stats["halluc_rate"],
    poison_llm_mean_risk_score=poison_llm_stats["mean_risk_score"],
    poison_llm_n_answered=poison_llm_stats["n"],
    certification_mixed_deployment=dict(realized_risk=poison_llm_cert["realized_risk"],
                                          violation_pct=poison_llm_cert["violation_pct"],
                                          answer_rate_pct=poison_llm_cert["answer_rate_pct"]),
    poison_llm_alone_accept_rate_pct=poison_llm_cert["new_alone_accept_rate_pct"],
    note="CORRECTED 2026-09-06: excludes qid=cap1 (generator declined and wrote the true "
         "answer instead of a false one; see main.tex and generate_poison_panel_docs.py for "
         "the full account). 117/119 = 98.3% hallucination, not the originally reported "
         "117/120 = 97.5%. poison_llm condition and its risk scores were never used to fit "
         "the GBM or the certification bins; this is genuine out-of-distribution "
         "generalisation, not cross-validation on data the estimator was trained on."
)
json.dump(out_single, open(f"{DATA}/poison_llm_generalization_report.json", "w"), indent=1)
print(f"\nsaved CORRECTED poison_llm_generalization_report.json")

out_panel = dict(
    original_ours_retrieval_auroc=float(auc_o),
    per_style=per_style_out,
    per_style_certification=per_style_cert,
    combined_all_four_styles=dict(n_total=n_total_new, **combined),
    note="Four independently-authored LLM poisoning styles (poison_llm: t10's single "
         "naturally-phrased false-fact passage; authority: attributes the fake value to a "
         "fictional authoritative source; negation: states the true value then denies it, "
         "asserting the fake value as a correction; corrob: two independently-generated "
         "passages asserting the same fake value), none ever seen by the estimator or bins "
         "during training/calibration, pooled into one deployment-mix certification test. "
         "poison_llm excludes qid=cap1 (see poison_llm_generalization_report.json note)."
)
json.dump(out_panel, open(f"{DATA}/poison_panel_generalization_report.json", "w"), indent=1)
print("saved poison_panel_generalization_report.json")
