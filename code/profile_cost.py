"""Computational-cost profiling for the risk estimator (Reviewer 1, revision).

Measures, on the real cached per-query features in release/data/controlled/:
  (1) feature-vector assembly cost (already-known scalars -> array; no new
      model call: the six retrieval descriptors come from similarity scores
      the retriever already produces, and the six model-internal descriptors
      come from token logits the generator already emits while decoding),
  (2) training cost of the risk estimator (one-off, offline), and
  (3) inference cost of the trained risk estimator, both single-query
      (includes Python call overhead) and batched (amortized).

Usage: python3 profile_cost.py [data_dir]  (default: ./data/controlled)
"""
import json
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "has_ctx"]
MOD = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
FEATS = RET + MOD


def load(path):
    recs = json.load(open(path))
    X = np.array([[r[f] for f in FEATS] for r in recs], dtype=float)
    y = np.array([0 if r["label"] == "correct" else 1 for r in recs], dtype=int)
    return X, y, recs


def profile_one(name, path):
    X, y, recs = load(path)
    n = len(recs)
    n_train = int(0.8 * n)
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr = y[:n_train]

    t0 = time.perf_counter()
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
    clf.fit(Xtr, ytr)
    train_ms = (time.perf_counter() - t0) * 1000

    row = Xte[0:1]
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        clf.predict_proba(row)
    single_us = (time.perf_counter() - t0) / reps * 1e6

    t0 = time.perf_counter()
    clf.predict_proba(Xte)
    batch_us = (time.perf_counter() - t0) / len(Xte) * 1e6

    reps2 = 100_000
    sample_rec = recs[0]
    t0 = time.perf_counter()
    for _ in range(reps2):
        _ = np.array([[sample_rec[f] for f in FEATS]], dtype=float)
    assemble_us = (time.perf_counter() - t0) / reps2 * 1e6

    print(f"{name}: n={n}, n_train={n_train}")
    print(f"  training (one-off, offline):      {train_ms:8.1f} ms")
    print(f"  feature-vector assembly:           {assemble_us:8.2f} us/query")
    print(f"  inference, single-query call:      {single_us:8.1f} us/query")
    print(f"  inference, batched (amortized):    {batch_us:8.2f} us/query")
    print()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/controlled"
    for name, fname in [
        ("flan-t5-small (80M)", "results_flan_t5_small.json"),
        ("Qwen2.5-7B-Instruct", "results_qwen2_5_7b_instruct.json"),
    ]:
        profile_one(name, f"{data_dir}/{fname}")
