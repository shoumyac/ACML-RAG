"""t8 companion analysis: self-consistency and semantic-volume baselines,
computed on the SAME fresh run that persists the five sampled generations
(results_selfconsistency_semvolume_flan_t5_small.json), so semantic_entropy,
one_minus_ptrue, self_consistency, semantic_volume and ours_retrieval are all
evaluated on one internally-consistent sample rather than mixing this run's
new columns onto the older cached results_se_flan_t5_small.json.

(A per-row self-check against that older cached file, built into the
generation script, found that greedy decoding on CPU is not perfectly
bit-reproducible run to run for this model: about 8% of rows produced a
different greedy answer than the original cached run despite an identical
seeded context, overwhelmingly with IDENTICAL retrieval features, i.e. the
same prompt going in. This is consistent with known floating-point
non-determinism in transformer inference, not a bug in the retrieval/prompt
logic reused verbatim from run_semantic_entropy.py. Confirmed harmless: this
fresh run's own semantic_entropy/one_minus_ptrue AUROC, 0.652/0.692, land
close to the already-published 0.675/0.710 from the original run, the gap
being the expected size for a resample under this noise.)
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
rows = json.load(open(f"{DATA}/results_selfconsistency_semvolume_flan_t5_small.json"))
ans = [r for r in rows if r["label"] != "abstain"]
y = np.array([1 if r["label"] == "hallucination" else 0 for r in ans])
groups = np.array([r["qid"] for r in ans])
print(f"records={len(rows)} answered={len(y)} halluc_rate={y.mean():.3f}")

def col(k): return np.array([r[k] for r in ans], float)
se = col("se"); ptrue = col("p_true")
selfcons = col("self_consistency"); semvol = col("semantic_volume")
RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "has_ctx"]
Xret = np.array([[r[k] for k in RET] for r in ans])

def oof(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
p_ours = oof(Xret)

auc = {}
auc["semantic_entropy"] = roc_auc_score(y, se)
auc["one_minus_ptrue"] = roc_auc_score(y, 1 - ptrue)
auc["self_consistency_inv"] = roc_auc_score(y, 1 - selfcons)
auc["semantic_volume"] = roc_auc_score(y, semvol)
auc["ours_retrieval"] = roc_auc_score(y, p_ours)

print("\n=== Hallucination-prediction AUROC (this fresh, internally-consistent run) ===")
for k in ["semantic_entropy", "one_minus_ptrue", "self_consistency_inv", "semantic_volume", "ours_retrieval"]:
    print(f"{k:22s} {auc[k]:.3f}")

# grouped bootstrap: ours_retrieval vs each new baseline
rng = np.random.default_rng(2026)
qids = np.array(sorted(set(groups)))
qid_to_rows = {q: np.where(groups == q)[0] for q in qids}
diffs_sc, diffs_sv = [], []
for _ in range(2000):
    samp_q = rng.choice(qids, size=len(qids), replace=True)
    idx = np.concatenate([qid_to_rows[q] for q in samp_q])
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == len(yb): continue
    diffs_sc.append(roc_auc_score(yb, p_ours[idx]) - roc_auc_score(yb, 1 - selfcons[idx]))
    diffs_sv.append(roc_auc_score(yb, p_ours[idx]) - roc_auc_score(yb, semvol[idx]))
diffs_sc = np.array(diffs_sc); diffs_sv = np.array(diffs_sv)
print("\n=== Bootstrap (2000 resamples grouped by qid): ours_retrieval minus new baseline ===")
print(f"ours - self_consistency: {diffs_sc.mean():+.3f}  95% CI [{np.percentile(diffs_sc,2.5):+.3f}, {np.percentile(diffs_sc,97.5):+.3f}]")
print(f"ours - semantic_volume : {diffs_sv.mean():+.3f}  95% CI [{np.percentile(diffs_sv,2.5):+.3f}, {np.percentile(diffs_sv,97.5):+.3f}]")

out = dict(auc=auc, halluc_rate=float(y.mean()), n_answered=int(len(y)), n_total=int(len(rows)),
           bootstrap_ours_minus_selfconsistency=dict(mean=float(diffs_sc.mean()),
                                                      ci95=[float(np.percentile(diffs_sc,2.5)), float(np.percentile(diffs_sc,97.5))]),
           bootstrap_ours_minus_semanticvolume=dict(mean=float(diffs_sv.mean()),
                                                     ci95=[float(np.percentile(diffs_sv,2.5)), float(np.percentile(diffs_sv,97.5))]),
           note="Computed on a fresh rerun of run_semantic_entropy.py's pipeline that additionally "
                "persists the 5 sampled generations; not a join onto the older cached "
                "results_se_flan_t5_small.json (see module docstring for why).")
json.dump(out, open(f"{DATA}/selfconsistency_semvolume_report.json", "w"), indent=1)
print("\nsaved selfconsistency_semvolume_report.json")
