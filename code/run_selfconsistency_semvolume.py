"""t8: self-consistency and semantic-volume baselines.

Reruns the SAME pilot generation pipeline as run_semantic_entropy.py (identical
retrieval, identical context construction seeded by random.Random(4000+qi),
identical prompts, identical greedy decoding) on the same already-tested open
model (google/flan-t5-small), but additionally PERSISTS the five sampled
answers that the original script discards after reducing them to a single
(se, se_nclusters) pair. From those persisted samples this script computes
two additional real UQ baselines:

  self_consistency : plurality-agreement fraction among the 5 samples under
    the SAME normalised exact-match equivalence relation norm() already used
    for discrete semantic entropy (Wang et al., ICLR 2023, "Self-Consistency
    Improves Chain of Thought Reasoning in Language Models").
  semantic_volume   : log det(V^T V) of the L2-normalised sentence-embedding
    vectors of the 5 samples (columns of V), embedded with
    sentence-transformers/all-MiniLM-L6-v2. Formula and framing follow
    Li et al., AAAI 2026, "Semantic Volume: Quantifying and Detecting both
    External and Internal Uncertainty in LLMs". A small ridge (eps * I) is
    added before the determinant purely for numerical stability when samples
    are identical (a rank-deficient Gram matrix would otherwise give
    log(0) = -inf); this is standard practice for Gram-determinant volume
    estimators, not a change to the method.

Self-check built in: for every record this script also looks up the ALREADY
CACHED greedy answer/label for the same (qid, cond) in
data/results_se_flan_t5_small.json and confirms they match exactly. Greedy
decoding (do_sample=False) and the context construction (seeded RNG) are both
deterministic given the same model weights, so a faithful reimplementation
must reproduce the cached greedy answers/labels byte for byte; any mismatch
means something in this script's retrieval/prompt logic has drifted from the
original and the new samples should NOT be trusted until it is fixed. Only
the 5 stochastic samples (and everything derived from them: se, se_nclusters,
self_consistency, semantic_volume) are expected to differ from the original
cached run, since torch's generation sampling RNG is not separately seeded
in either script.

Usage: python3 run_selfconsistency_semvolume.py <budget_s>
"""
import json, os, re, sys, time, random, math
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
MODEL = os.environ.get("PILOT_MODEL", "google/flan-t5-small")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TAG = "se_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1])
OUT_TAG = "selfconsistency_semvolume_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1])
SUBSET = int(os.environ.get("SUBSET", "4"))
NSAMP = int(os.environ.get("NSAMP", "5"))
RIDGE_EPS = 1e-6
CKPT = os.environ.get("PILOT_CKPT", "/tmp/ncsrag_t8_ckpt"); os.makedirs(CKPT, exist_ok=True)

docs = json.load(open(f"{DATA}/docs.json"))
questions_all = json.load(open(f"{DATA}/questions.json"))
questions = [q for i, q in enumerate(questions_all)
             if q["regime"] == "real" or (i % SUBSET == 0)]
bm25 = BM25Okapi([re.findall(r"\w+", d["text"].lower()) for d in docs])
by_kind = {}
for i, d in enumerate(docs):
    by_kind.setdefault(d["kind"], []).append(i)
gold_idx, contra_idx, distract_idx = {}, {}, {}
for i, d in enumerate(docs):
    k = (d.get("entity"), d.get("attr"))
    if d["kind"] == "gold": gold_idx[k] = i
    elif d["kind"] == "contra": contra_idx[k] = i
    elif d["kind"] == "distractor": distract_idx[(d.get("target"), d.get("attr"))] = i

# reference cache for the self-check (greedy/label must match this exactly)
ref_path = f"{DATA}/results_{TAG}.json"
ref_by_key = {}
if os.path.exists(ref_path):
    for r in json.load(open(ref_path)):
        ref_by_key[(r["qid"], r["cond"])] = r
print(f"loaded {len(ref_by_key)} cached reference rows from {ref_path}", flush=True)

tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval(); torch.set_num_threads(2)
embedder = SentenceTransformer(EMBED_MODEL)
print("loaded", MODEL, "and", EMBED_MODEL, "questions", len(questions), flush=True)

def retrieve_clean(q, k=4):
    sc = bm25.get_scores(re.findall(r"\w+", q.lower()))
    order = np.argsort(-sc)
    return [int(i) for i in order if docs[i]["kind"] not in ("contra", "distractor")][:k], sc

def build_condition(q, cond, rng):
    if cond == "closedbook": return [], None
    base, sc = retrieve_clean(q["question"])
    gi = gold_idx[(q["entity"], q["attr"])]
    if gi not in base: base = [gi] + base[:3]
    ci = contra_idx.get((q["entity"], q["attr"]))
    if cond == "clean": ctx = list(base); rng.shuffle(ctx)
    elif cond == "irr50": ctx = base[:2] + rng.sample(by_kind["irrelevant"], 2); rng.shuffle(ctx)
    elif cond == "irr100": ctx = rng.sample(by_kind["irrelevant"], 4)
    elif cond == "contra_r1": ctx = [ci] + [d for d in base if d != ci][:3]
    elif cond == "contra_only": ctx = [ci] + rng.sample(by_kind["irrelevant"], 3); rng.shuffle(ctx)
    else: ctx = list(base)
    return ctx, sc

def prompt_for(q, ctx_ids):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        return (f"Answer using the context. If the context does not contain the "
                f"answer, answer unknown.\nContext:\n{ctx}\nQ: {q} A:")
    return f"Q: {q} A:"

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
ABST = re.compile(r"(don.?t know|unknown|cannot|not (mentioned|stated|provided|given)|no answer)")
def label(ans, gold):
    a, g = norm(ans), norm(gold)
    if not a or ABST.search(a): return "abstain"
    if g in a or (a in g and len(a) > 2): return "correct"
    if re.fullmatch(r"\d{4}", g) and g in re.findall(r"\d{4}", a): return "correct"
    return "hallucination"

@torch.no_grad()
def greedy(prompt):
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    o = model.generate(**enc, max_new_tokens=12, do_sample=False)
    return tk.decode(o[0], skip_special_tokens=True).strip()

@torch.no_grad()
def sample(prompt, n):
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    outs = model.generate(**enc, max_new_tokens=12, do_sample=True, temperature=0.7,
                          top_p=0.9, num_return_sequences=n)
    return [tk.decode(o, skip_special_tokens=True).strip() for o in outs]

def discrete_semantic_entropy(samples):
    from collections import Counter
    c = Counter(norm(s) for s in samples)
    tot = sum(c.values())
    return float(-sum((v / tot) * math.log(v / tot) for v in c.values())), len(c)

def self_consistency_score(samples):
    """Plurality-agreement fraction (Wang et al. 2023), same equivalence
    relation (norm()) as discrete semantic entropy above."""
    from collections import Counter
    c = Counter(norm(s) for s in samples)
    return max(c.values()) / len(samples)

def semantic_volume_score(samples):
    """log det(V^T V) with V's columns the L2-normalised embeddings of the
    samples (Li et al., AAAI 2026). Ridge-regularised for numerical stability
    when samples coincide (rank-deficient Gram matrix)."""
    emb = embedder.encode(list(samples), show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = (emb / norms).T  # (d, n)
    gram = V.T @ V + RIDGE_EPS * np.eye(V.shape[1])
    sign, logdet = np.linalg.slogdet(gram)
    return float(logdet)

@torch.no_grad()
def p_true(q, ctx_ids, ans):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        pr = (f"Context:\n{ctx}\nQuestion: {q}\nProposed answer: {ans}\n"
              f"Is the proposed answer correct? Answer True or False. Answer:")
    else:
        pr = (f"Question: {q}\nProposed answer: {ans}\n"
              f"Is the proposed answer correct? Answer True or False. Answer:")
    enc = tk(pr, return_tensors="pt", truncation=True, max_length=512)
    dec = tk("<pad>", return_tensors="pt").input_ids[:, :1]
    out = model(**enc, decoder_input_ids=dec)
    logits = out.logits[0, -1]
    tid = tk("True", add_special_tokens=False).input_ids[0]
    fid = tk("False", add_special_tokens=False).input_ids[0]
    lp = torch.log_softmax(logits, -1)
    pt = float(torch.exp(lp[tid])); pf = float(torch.exp(lp[fid]))
    z = pt + pf + 1e-9
    return pt / z

CONDS = os.environ.get("CONDS", "closedbook,clean,irr50,irr100,contra_r1,contra_only").split(",")

def atomic(obj, path):
    t = path + ".tmp"; json.dump(obj, open(t, "w")); os.replace(t, path)

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    part = f"{CKPT}/results_{OUT_TAG}_partial.json"
    rows = json.load(open(part)) if os.path.exists(part) else []
    done = {(r["qid"], r["cond"]) for r in rows}
    t0 = time.time()
    n_checked, n_mismatch = 0, 0
    mismatches = []
    for qi, q in enumerate(questions):
        rng = random.Random(4000 + qi)
        for cond in CONDS:
            if (q["qid"], cond) in done: continue
            if time.time() - t0 > budget:
                atomic(rows, part)
                print(f"CHECKPOINT rows={len(rows)}", flush=True)
                print(f"self-check so far: {n_checked} checked, {n_mismatch} mismatches", flush=True)
                return
            ctx, sc = build_condition(q, cond, rng)
            pr = prompt_for(q["question"], ctx)
            g = greedy(pr); lab = label(g, q["gold"])
            sm = sample(pr, NSAMP)
            se, nclu = discrete_semantic_entropy(sm)
            sc_score = self_consistency_score(sm)
            vol = semantic_volume_score(sm)
            pt = p_true(q["question"], ctx, g)
            if ctx:
                cs = sorted([float(sc[i]) for i in ctx], reverse=True)
                feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                            ret_margin=cs[0]-cs[1], ret_min=cs[-1], has_ctx=1.0)
            else:
                feat = dict(ret_mean=0, ret_top1=0, ret_margin=0, ret_min=0, has_ctx=0.0)
            ref = ref_by_key.get((q["qid"], cond))
            if ref is not None:
                n_checked += 1
                if ref["answer"] != g or ref["label"] != lab:
                    n_mismatch += 1
                    mismatches.append(dict(qid=q["qid"], cond=cond,
                                            ref_answer=ref["answer"], new_answer=g,
                                            ref_label=ref["label"], new_label=lab))
            rows.append(dict(qid=q["qid"], regime=q["regime"], cond=cond, gold=q["gold"],
                             answer=g, label=lab, se=se, se_nclusters=nclu, p_true=pt,
                             samples=sm, self_consistency=sc_score, semantic_volume=vol,
                             **feat))
            done.add((q["qid"], cond))
        if qi % 5 == 0:
            print(f"q {qi}/{len(questions)} rows={len(rows)} t={time.time()-t0:.0f}s "
                  f"checked={n_checked} mismatch={n_mismatch}", flush=True)
            atomic(rows, part)
    atomic(rows, part)
    atomic(rows, f"{DATA}/results_{OUT_TAG}.json")
    atomic(mismatches, f"{DATA}/results_{OUT_TAG}_mismatches.json")
    print(f"DONE rows={len(rows)} time={time.time()-t0:.0f}s", flush=True)
    print(f"self-check: {n_checked} checked against cached original, {n_mismatch} mismatches", flush=True)

if __name__ == "__main__":
    main()
