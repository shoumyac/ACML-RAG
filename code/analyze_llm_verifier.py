"""t9: LLM-as-verifier baseline, combining two independent judges.

Two models independently rate confidence (0-100) that flan-t5-small's answer
is correct, WITHOUT ever being shown the gold answer (same VERIFIER_PROMPT in
both run_llm_verifier.py-style scripts, verified identical by inspection):
  - DeepSeek V4 Flash via OpenRouter (711 records, the generation script
    pre-filtered to originally-answered records only)
  - Qwen2.5-14B, locally hosted via Ollama (720 records, unfiltered; the 9
    records with the pilot's own original label=="abstain" are dropped here
    to match DeepSeek's scope exactly)

Risk score = 100 - verifier_conf (higher = more likely hallucination).
verifier_conf is None when the verifier itself declines to rate ("ABSTAIN");
those records are excluded from that judge's AUROC (legitimate unbiased
abstentions, not missing data caused by an error).

Combined score = mean(deepseek risk, qwen risk) on the intersection of
(qid, cond) keys where BOTH judges returned a numeric confidence.

AUROC bootstrap CIs are grouped by qid (2000 resamples), matching the
convention used for every other baseline in this paper (t7, t8).
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))

ds = json.load(open(f"{DATA}/verifier_deepseek_se_flan_t5_small.json"))
qw_all = json.load(open(f"{DATA}/verifier_qwen_se_flan_t5_small.json"))
qw = [r for r in qw_all if r["label"] != "abstain"]

print(f"deepseek: {len(ds)} records (pre-filtered to non-abstain by the generation script)")
print(f"qwen: {len(qw_all)} total, {len(qw)} after dropping the 9 original label=='abstain' rows")

ds_idx = {(r["qid"], r["cond"]): r for r in ds}
qw_idx = {(r["qid"], r["cond"]): r for r in qw}
shared = set(ds_idx) & set(qw_idx)
print(f"shared (qid,cond) keys: {len(shared)} (deepseek={len(ds_idx)}, qwen={len(qw_idx)})")

mismatches = [k for k in shared if ds_idx[k]["label"] != qw_idx[k]["label"]
              or ds_idx[k]["gold"] != qw_idx[k]["gold"]
              or ds_idx[k]["answer"] != qw_idx[k]["answer"]]
print(f"label/gold/answer mismatches on shared keys: {len(mismatches)} "
      f"(0 expected: both scored the exact same underlying generations)")


def usable_xy(records, conf_key="verifier_conf"):
    u = [r for r in records if r.get(conf_key) is not None]
    y = np.array([1 if r["label"] == "hallucination" else 0 for r in u])
    s = np.array([100.0 - r[conf_key] for r in u])
    q = np.array([r["qid"] for r in u])
    return y, s, q, len(u), len(records)


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
    return float(lo), float(hi), len(aucs)


results = {}
for name, records in [("deepseek", ds), ("qwen", qw)]:
    y, s, q, n_usable, n_total = usable_xy(records)
    auc = roc_auc_score(y, s)
    lo, hi = grouped_bootstrap_ci(y, s, q)[:2]
    n_abstain = n_total - n_usable
    print(f"\n{name}: AUROC={auc:.3f} 95% CI [{lo:.3f}, {hi:.3f}] "
          f"usable={n_usable}/{n_total} (verifier abstained on {n_abstain})")
    results[name] = dict(auroc=float(auc), ci_lo=lo, ci_hi=hi,
                          n_usable=n_usable, n_total=n_total, n_verifier_abstain=n_abstain)

# combined: mean risk score on the intersection where BOTH gave a numeric conf
combo_rows = []
for k in shared:
    rd, rq = ds_idx[k], qw_idx[k]
    if rd.get("verifier_conf") is not None and rq.get("verifier_conf") is not None:
        combo_rows.append(dict(qid=k[0], cond=k[1], label=rd["label"],
                                risk=((100.0 - rd["verifier_conf"]) + (100.0 - rq["verifier_conf"])) / 2))

y_c = np.array([1 if r["label"] == "hallucination" else 0 for r in combo_rows])
s_c = np.array([r["risk"] for r in combo_rows])
q_c = np.array([r["qid"] for r in combo_rows])
auc_c = roc_auc_score(y_c, s_c)
lo_c, hi_c, n_boot_ok = grouped_bootstrap_ci(y_c, s_c, q_c)
print(f"\ncombined (mean of both judges' risk scores, n={len(combo_rows)} where both answered): "
      f"AUROC={auc_c:.3f} 95% CI [{lo_c:.3f}, {hi_c:.3f}]")
results["combined"] = dict(auroc=float(auc_c), ci_lo=lo_c, ci_hi=hi_c, n=len(combo_rows))

# comparison context: our retrieval-only estimator's published AUROC is 0.893
# (Table 3); semantic entropy 0.675; P(True) 0.710 -- reused as literals here
# since those numbers are already published and this script only concerns
# the verifier baseline.
OURS = 0.893
print(f"\nfor context: semantic entropy 0.675, P(True) 0.710, our retrieval-only "
      f"estimator {OURS}. Best verifier ({max(results['deepseek']['auroc'], results['qwen']['auroc'], results['combined']['auroc']):.3f}) "
      f"is {OURS - max(results['deepseek']['auroc'], results['qwen']['auroc'], results['combined']['auroc']):.3f} below ours.")

json.dump(results, open(f"{DATA}/llm_verifier_report.json", "w"), indent=1)
print("\nsaved llm_verifier_report.json")
