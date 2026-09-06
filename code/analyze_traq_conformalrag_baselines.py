"""t7: TRAQ-style and Conformal-RAG-style selection-rule baselines.

Extends analyze_baselines.py's comparison (does not modify it) with two more
selection rules built entirely from signals already cached in
results_se_flan_t5_small.json: no rerun, no new generation.

TRAQ (Li et al., NAACL 2024) ensembles a retrieval-stage conformal set
(nonconformity = negative similarity between the question and the retrieved
passage) with a generation-stage conformal set (nonconformity = a
sample-clustering confidence, i.e. the same family of quantity as the
semantic-entropy baseline already in the table). We do not have the raw
multi-sample generations needed to recompute the generation-stage set, so we
faithfully reproduce only TRAQ's retrieval-stage nonconformity score, applied
with the same standard (marginal, non-Mondrian) split-conformal calibration
TRAQ uses. This isolates the paper's own contribution (risk-binned Mondrian
calibration) from the choice of marginal calibration alone.

Conformal-RAG (Feng et al., SIGIR 2025) achieves group-conditional coverage
by calibrating a separate split-conformal threshold per group, rather than
one global threshold. Its groups are auto-discovered semantic sub-domains;
the closest available proxy in our cached data is the observable retrieval
condition ("cond": clean / contra_only / contra_r1 / irr50 / irr100 /
closedbook), which is a fair, honestly-labelled substitute: it is an
intrinsic, non-model-derived covariate, exactly like their sub-domain label,
just coarser. We apply their exact split-conformal quantile formula
per-group to 1-P(True) (the strongest single existing signal).

Both new rows share cp_upper/thresh with analyze_baselines.py's own
selective-generation-under-shift evaluation and reuse its exact ALPHA,
noisy-condition set, and calibration/test split procedure (same rng seed
sequence) so every method in the final table is evaluated on identical
cal/test partitions each repetition.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
rows = json.load(open(f"{DATA}/results_se_flan_t5_small.json"))
ans = [r for r in rows if r["label"] != "abstain"]
y = np.array([1 if r["label"] == "hallucination" else 0 for r in ans])
groups = np.array([r["qid"] for r in ans])
conds = np.array([r["cond"] for r in ans])
qids = sorted(set(groups))
print(f"records={len(rows)} answered={len(y)} halluc_rate={y.mean():.3f}")

def col(k): return np.array([r[k] for r in ans], float)
se = col("se"); ptrue = col("p_true")
ret_top1 = col("ret_top1"); ret_min = col("ret_min"); ret_margin = col("ret_margin")
RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "has_ctx"]
Xret = np.array([[r[k] for k in RET] for r in ans])

# TRAQ retrieval-stage nonconformity: higher similarity = more conforming,
# so risk (higher = worse) is the negated similarity of the passage actually
# shown to the generator.
traq_top1 = -ret_top1
traq_min = -ret_min
traq_margin = -ret_margin

# --- AUROC table (extends analyze_baselines.py's) ---
def oof(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
p_ours = oof(Xret)

auc = {
    "semantic_entropy": roc_auc_score(y, se),
    "one_minus_ptrue": roc_auc_score(y, 1 - ptrue),
    "traq_style_ret_top1": roc_auc_score(y, traq_top1),
    "traq_style_ret_min": roc_auc_score(y, traq_min),
    "traq_style_ret_margin": roc_auc_score(y, traq_margin),
    "ours_retrieval": roc_auc_score(y, p_ours),
}
print("\n=== Hallucination-prediction AUROC (extends analyze_baselines.py) ===")
for k, v in auc.items():
    print(f"{k:24s} {v:.3f}")
print("note: Conformal-RAG-style has no separate AUROC row - it recalibrates")
print("acceptance thresholds per observable group, it does not change the")
print("underlying ranking of 1-P(True), so its ranking quality is identical")
print("to the one_minus_ptrue row above. Its contribution only shows up in")
print("the risk-coverage table below.")

# bootstrap: does the TRAQ-style retrieval score add anything ours_retrieval
# (the paper's own retrieval-feature classifier) doesn't already capture?
rng = np.random.default_rng(0)
gaps = []
for _ in range(2000):
    bq = set(rng.choice(qids, len(qids), replace=True))
    m = np.array([g in bq for g in groups])
    if len(set(y[m])) < 2: continue
    gaps.append(roc_auc_score(y[m], p_ours[m]) - roc_auc_score(y[m], traq_top1[m]))
print(f"\nours_retrieval - traq_style_ret_top1: {np.mean(gaps):+.3f} "
      f"[95% CI {np.percentile(gaps,2.5):+.3f}, {np.percentile(gaps,97.5):+.3f}]")

# --- selective generation under shift: add TRAQ-threshold and Conformal-RAG group-conditional ---
ALPHA = 0.10
noisy = {"irr100", "contra_r1", "contra_only"}
is_noisy = np.array([r["cond"] in noisy for r in ans])

def cp_upper(k, n, d=0.10):
    if n == 0: return 1.0
    return float(beta_dist.ppf(1 - d, k + 1, n - k)) if k < n else 1.0

def thresh(s, l, a):
    o = np.argsort(s); ss, ll = s[o], l[o]
    cl = np.cumsum(ll); cn = np.arange(1, len(ss) + 1)
    ok = (cl + 1) / (cn + 1) <= a
    return ss[np.where(ok)[0].max()] if ok.any() else -np.inf

def riskbin(pc, yc, pt, yt, a, nb=8):
    qs = np.quantile(pc, np.linspace(0, 1, nb + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cert = [b for b in range(nb) if ((pc > qs[b]) & (pc <= qs[b + 1])).sum() > 0
            and cp_upper(int(yc[(pc > qs[b]) & (pc <= qs[b + 1])].sum()),
                          int(((pc > qs[b]) & (pc <= qs[b + 1])).sum())) <= a]
    tb = np.digitize(pt, qs) - 1; acc = np.isin(tb, cert)
    return (float(yt[acc].mean()), float(acc.mean())) if acc.sum() else (0.0, 0.0)

def evalmarg(s, l, t):
    m = s <= t; return (float(l[m].mean()), float(m.mean())) if m.sum() else (0.0, 0.0)

def groupcond(sc, yc, gc, st, yt, gt, a):
    """Conformal-RAG-style: one split-conformal threshold per observable group."""
    fallback = thresh(sc, yc, a)
    th = {}
    for g in set(gc):
        mask = gc == g
        th[g] = thresh(sc[mask], yc[mask], a) if mask.sum() >= 10 else fallback
    tvec = np.array([th.get(g, fallback) for g in gt])
    acc = st <= tvec
    return (float(yt[acc].mean()), float(acc.mean())) if acc.sum() else (0.0, 0.0)

def groupcond_acc_mask(sc, yc, gc, st, gt, a):
    fallback = thresh(sc, yc, a)
    th = {}
    for g in set(gc):
        mask = gc == g
        th[g] = thresh(sc[mask], yc[mask], a) if mask.sum() >= 10 else fallback
    tvec = np.array([th.get(g, fallback) for g in gt])
    return st <= tvec

res = {"SE-threshold (marginal)": [], "P(True)-threshold (marginal)": [],
       "ours risk-binned": [], "TRAQ-threshold (marginal)": [],
       "Conformal-RAG group-conditional (by cond)": []}
# diagnostic: does Conformal-RAG-style's risk-guarantee failure concentrate in
# the shifted (noisy) test subset, or is it spread evenly? Same question for
# our own risk-binned method as contrast.
breakdown = {"confrag_noisy": [], "confrag_clean": [], "ours_noisy": [], "ours_clean": []}
for rep in range(150):
    rr = np.random.default_rng(900 + rep); sh = rr.permutation(qids)
    calq = set(sh[:len(qids) // 2]); teq = set(sh[len(qids) // 2:])
    cal = np.array([g in calq for g in groups]); te = np.array([g in teq for g in groups])
    ck = cal & (~is_noisy | (rr.random(len(cal)) < 0.30)); tk = te & (is_noisy | (rr.random(len(te)) < 0.50))
    if ck.sum() < 30 or tk.sum() < 30: continue
    res["SE-threshold (marginal)"].append(evalmarg(se[tk], y[tk], thresh(se[ck], y[ck], ALPHA)))
    res["P(True)-threshold (marginal)"].append(evalmarg((1 - ptrue)[tk], y[tk], thresh((1 - ptrue)[ck], y[ck], ALPHA)))
    res["ours risk-binned"].append(riskbin(p_ours[ck], y[ck], p_ours[tk], y[tk], ALPHA))
    res["TRAQ-threshold (marginal)"].append(evalmarg(traq_top1[tk], y[tk], thresh(traq_top1[ck], y[ck], ALPHA)))
    res["Conformal-RAG group-conditional (by cond)"].append(
        groupcond((1 - ptrue)[ck], y[ck], conds[ck], (1 - ptrue)[tk], y[tk], conds[tk], ALPHA))

    cr_acc = groupcond_acc_mask((1 - ptrue)[ck], y[ck], conds[ck], (1 - ptrue)[tk], conds[tk], ALPHA)
    qs = np.quantile(p_ours[ck], np.linspace(0, 1, 9)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cert = [b for b in range(8) if ((p_ours[ck] > qs[b]) & (p_ours[ck] <= qs[b + 1])).sum() > 0
            and cp_upper(int(y[ck][(p_ours[ck] > qs[b]) & (p_ours[ck] <= qs[b + 1])].sum()),
                          int(((p_ours[ck] > qs[b]) & (p_ours[ck] <= qs[b + 1])).sum())) <= ALPHA]
    ours_acc = np.isin(np.digitize(p_ours[tk], qs) - 1, cert)
    tk_noisy = is_noisy[tk]
    for label, acc_mask in [("confrag", cr_acc), ("ours", ours_acc)]:
        for seg, seg_mask in [("noisy", tk_noisy), ("clean", ~tk_noisy)]:
            sel = acc_mask & seg_mask
            if sel.sum() > 0:
                breakdown[f"{label}_{seg}"].append((float(y[tk][sel].mean()), float(sel.sum() / max(seg_mask.sum(), 1))))

print(f"\n=== Selective generation under noise shift (alpha={ALPHA}) ===")
print(f"{'method':42s} {'risk':>6s} {'viol%':>6s} {'answer%':>8s}")
out = {"auc": auc}
for k, v in res.items():
    a = np.array(v)
    if len(a) == 0: continue
    out[k] = dict(risk=float(a[:, 0].mean()), viol=float((a[:, 0] > ALPHA).mean()), cov=float(a[:, 1].mean()))
    print(f"{k:42s} {a[:,0].mean():6.3f} {(a[:,0]>ALPHA).mean()*100:6.0f} {a[:,1].mean()*100:8.1f}")

print(f"\n=== Where does each method's risk-guarantee failure concentrate? (alpha={ALPHA}) ===")
print(f"{'method x segment':28s} {'risk':>6s} {'viol%':>6s} {'coverage-within-segment%':>10s}")
for k, v in breakdown.items():
    a = np.array(v)
    if len(a) == 0:
        print(f"{k:28s}   (no qualifying repetitions)")
        continue
    out[f"breakdown_{k}"] = dict(risk=float(a[:, 0].mean()), viol=float((a[:, 0] > ALPHA).mean()), cov=float(a[:, 1].mean()))
    print(f"{k:28s} {a[:,0].mean():6.3f} {(a[:,0]>ALPHA).mean()*100:6.0f} {a[:,1].mean()*100:10.1f}")

json.dump(out, open(f"{DATA}/traq_conformalrag_report.json", "w"), indent=1)
print("\nsaved traq_conformalrag_report.json")
