"""t11: Semantic Entropy Probes baseline.

Kossen, Han, Razzak, Schut, Malik, Gal, "Semantic Entropy Probes: Robust and
Cheap Hallucination Detection in LLMs" (arXiv:2406.15927). SEPs are linear
probes trained on an LLM's hidden states from a SINGLE generation to
approximate semantic entropy, avoiding the 5-10x resampling cost the direct
method (and this pilot's own semantic-entropy baseline) requires. The
original paper's ablations use decoder-only models (e.g. Llama-2-7B); this
pilot's generator, flan-t5-small, is encoder-decoder, so there is no
"the" analogous position to reuse from their ablation grid. We use one
clearly-defined choice rather than reproducing the full layer/position
sweep: the FINAL-LAYER decoder hidden state at the generation step that
produced the last CONTENT token of the greedy answer (the step before EOS,
when EOS was reached before the 12-token budget; otherwise the final step
outright). This is captured directly from the same greedy decode already
run for every other baseline in this pilot (do_sample=False,
output_hidden_states=True), not a separate reconstruction, so it reflects
the hidden state actually produced during generation.

Reuses run_selfconsistency_semvolume.py's exact retrieval/context/prompt
reconstruction (same seeded random.Random(4000+qi)) and the same
self-check discipline: the reconstructed greedy answer/label is compared
against the cached original in data/results_se_flan_t5_small.json.

Only a single greedy generation per record is needed (no sampling), so
this script does NOT require the five persisted samples from t8 -- it
regresses onto the SE value t8's reference file already computed, which is
exactly the point of the method (cheap at inference time; the expensive
resampling only ever has to happen once, to generate training labels).

Usage: python3 run_semantic_entropy_probes.py <budget_s>
"""
import json, os, re, sys, time, random
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
MODEL = os.environ.get("PILOT_MODEL", "google/flan-t5-small")
TAG = "se_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1])
OUT_TAG = "sep_hidden_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1])
SUBSET = int(os.environ.get("SUBSET", "4"))
CKPT = os.environ.get("PILOT_CKPT", "/tmp/ncsrag_t11_ckpt")
os.makedirs(CKPT, exist_ok=True)

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

ref_path = f"{DATA}/results_{TAG}.json"
ref_by_key = {}
if os.path.exists(ref_path):
    for r in json.load(open(ref_path)):
        ref_by_key[(r["qid"], r["cond"])] = r
print(f"loaded {len(ref_by_key)} cached reference rows from {ref_path}", flush=True)

tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval(); torch.set_num_threads(2)
print("loaded", MODEL, "questions", len(questions), "d_model", model.config.d_model, flush=True)


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
def greedy_with_hidden(prompt):
    """Greedy-decode and capture the final-layer decoder hidden state at the
    step that produced the last CONTENT token (the step before EOS, if EOS
    was reached; otherwise the final step)."""
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                          output_hidden_states=True, return_dict_in_generate=True)
    seq = out.sequences[0]
    text = tk.decode(seq, skip_special_tokens=True).strip()
    gen_tokens = seq[1:]  # exclude the decoder-start token
    n_steps = len(gen_tokens)
    if n_steps > 1 and int(gen_tokens[-1]) == tk.eos_token_id:
        step_idx = n_steps - 2  # step that produced the last content token
    else:
        step_idx = n_steps - 1
    hidden = out.decoder_hidden_states[step_idx][-1][0, -1, :].numpy()
    return text, hidden.astype(np.float32)


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
    CONDS = os.environ.get("CONDS", "closedbook,clean,irr50,irr100,contra_r1,contra_only").split(",")
    for qi, q in enumerate(questions):
        rng = random.Random(4000 + qi)
        for cond in CONDS:
            if (q["qid"], cond) in done: continue
            if time.time() - t0 > budget:
                atomic(rows, part)
                print(f"CHECKPOINT rows={len(rows)} checked={n_checked} mismatch={n_mismatch}", flush=True)
                return
            ctx, sc = build_condition(q, cond, rng)
            pr = prompt_for(q["question"], ctx)
            g, hidden = greedy_with_hidden(pr)
            lab = label(g, q["gold"])
            ref = ref_by_key.get((q["qid"], cond))
            se_ref = ref["se"] if ref is not None else None
            if ref is not None:
                n_checked += 1
                if ref["answer"] != g or ref["label"] != lab:
                    n_mismatch += 1
                    mismatches.append(dict(qid=q["qid"], cond=cond,
                                            ref_answer=ref["answer"], new_answer=g,
                                            ref_label=ref["label"], new_label=lab))
            rows.append(dict(qid=q["qid"], regime=q["regime"], cond=cond, gold=q["gold"],
                              answer=g, label=lab, se=se_ref,
                              hidden=[round(float(x), 6) for x in hidden]))
            done.add((q["qid"], cond))
        if qi % 10 == 0:
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
