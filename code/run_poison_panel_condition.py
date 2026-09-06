"""t12: run each of the three new poisoning-panel styles (authority, negation,
corrob; see generate_poison_panel_docs.py) through the SAME flan-t5-small
pipeline as run_poison_llm_condition.py (t10), producing one held-out
condition per style, none of them ever seen by the estimator during training
or bin calibration.

Context construction mirrors t10's poison_llm condition exactly, so style is
the only thing that varies relative to that already-tested baseline:
  - authority, negation: [1 poison doc] + 3 random irrelevant fillers (4 total
    context docs), identical filler count and identical per-question RNG seed
    (9000 + qi) to t10's own poison_llm run.
  - corrob: [2 poison docs, independently authored, same fake value] + 2 of
    the SAME 3 fillers that seed would draw for the other conditions (the
    third is simply not used), keeping total context size fixed at 4 across
    every condition in the panel rather than letting corrob's context run
    longer, which would confound "two sources" with "more context". Because
    corrob draws its fillers as a strict prefix of the identical 3-filler
    draw the other styles use for the same qid, the filler documents
    themselves are bit-for-bit identical to what t10 and the other two panel
    styles use for that question; only the poison content differs.

Each style's own poison document(s) are appended to a COPY of docs_base, so
BM25 scores them exactly like any other document (no synthetic relevance
boost), exactly as t10 did.
"""
import json, os, re, math, random
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
MODEL = "google/flan-t5-small"
NSAMP = 5
STYLES = ["authority", "negation", "corrob"]

docs_base = json.load(open(f"{DATA}/docs.json"))
questions_all = json.load(open(f"{DATA}/questions.json"))
panel = json.load(open(f"{DATA}/poison_panel_docs.json"))
SUBSET = 4
questions = [q for i, q in enumerate(questions_all)
             if q["regime"] == "real" or (i % SUBSET == 0)]

for style in STYLES:
    rows_by_qid = {r["qid"]: r for r in panel[style]}
    assert all(q["qid"] in rows_by_qid for q in questions), f"{style} missing a selected qid"
    assert all(r.get("error") is None for r in panel[style]), f"{style} has un-regenerated error rows"

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
ABST = re.compile(r"(don.?t know|unknown|cannot|not (mentioned|stated|provided|given)|no answer)")
def label(ans, gold):
    a, g = norm(ans), norm(gold)
    if not a or ABST.search(a): return "abstain"
    if g in a or (a in g and len(a) > 2): return "correct"
    if re.fullmatch(r"\d{4}", g) and g in re.findall(r"\d{4}", a): return "correct"
    return "hallucination"

def prompt_for(q, ctx_ids, docs):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        return (f"Answer using the context. If the context does not contain the "
                f"answer, answer unknown.\nContext:\n{ctx}\nQ: {q} A:")
    return f"Q: {q} A:"

tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval(); torch.set_num_threads(2)
print("loaded", MODEL, flush=True)

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

@torch.no_grad()
def p_true(q, ctx_ids, ans, docs):
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

def atomic(obj, path):
    t = path + ".tmp"; json.dump(obj, open(t, "w")); os.replace(t, path)

def run_style(style):
    rows_by_qid = {r["qid"]: r for r in panel[style]}
    docs = list(docs_base)
    by_kind = {}
    for i, d in enumerate(docs):
        by_kind.setdefault(d["kind"], []).append(i)
    poison_idx = {}
    for q in questions:
        pr = rows_by_qid[q["qid"]]
        if style == "corrob":
            idx_a = len(docs)
            docs.append({"id": f"panel_{style}_{q['qid']}_a", "text": pr["texts"][0],
                         "entity": q["entity"], "attr": q["attr"],
                         "kind": f"panel_{style}", "value": pr["fake"]})
            idx_b = len(docs)
            docs.append({"id": f"panel_{style}_{q['qid']}_b", "text": pr["texts"][1],
                         "entity": q["entity"], "attr": q["attr"],
                         "kind": f"panel_{style}", "value": pr["fake"]})
            poison_idx[q["qid"]] = (idx_a, idx_b)
        else:
            idx = len(docs)
            docs.append({"id": f"panel_{style}_{q['qid']}", "text": pr["text"],
                         "entity": q["entity"], "attr": q["attr"],
                         "kind": f"panel_{style}", "value": pr["fake"]})
            poison_idx[q["qid"]] = (idx,)

    bm25 = BM25Okapi([re.findall(r"\w+", d["text"].lower()) for d in docs])
    out_path = f"{DATA}/results_poison_panel_{style}_flan_t5_small.json"
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {r["qid"] for r in rows}
    for qi, q in enumerate(questions):
        if q["qid"] in done: continue
        rng = random.Random(9000 + qi)  # identical seed/draw to t10's poison_llm run
        fillers3 = rng.sample([i for i in by_kind["irrelevant"]], 3)
        pidx = poison_idx[q["qid"]]
        if style == "corrob":
            ctx = list(pidx) + fillers3[:2]  # 2 poison docs + first 2 of the same 3 fillers
        else:
            ctx = list(pidx) + fillers3      # 1 poison doc + all 3 fillers
        rng.shuffle(ctx)  # continue the SAME rng stream used for .sample() above, exactly as t10 does
        sc_full = bm25.get_scores(re.findall(r"\w+", q["question"].lower()))
        cs = sorted([float(sc_full[i]) for i in ctx], reverse=True)
        feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                    ret_margin=cs[0] - cs[1], ret_min=cs[-1], has_ctx=1.0)
        pr_text = prompt_for(q["question"], ctx, docs)
        g = greedy(pr_text); lab = label(g, q["gold"])
        sm = sample(pr_text, NSAMP)
        se, nclu = discrete_semantic_entropy(sm)
        pt = p_true(q["question"], ctx, g, docs)
        rec = rows_by_qid[q["qid"]]
        row = dict(qid=q["qid"], regime=q["regime"], cond=f"panel_{style}", gold=q["gold"],
                   answer=g, label=lab, se=se, se_nclusters=nclu, p_true=pt,
                   poison_fake_value=rec["fake"], **feat)
        if style == "corrob":
            row["poison_texts"] = rec["texts"]
        else:
            row["poison_text"] = rec["text"]
        rows.append(row)
        done.add(q["qid"])
        if qi % 20 == 0:
            print(f"[{style}] q {qi}/{len(questions)} rows={len(rows)}", flush=True)
            atomic(rows, out_path)
    atomic(rows, out_path)
    n_halluc = sum(1 for r in rows if r["label"] == "hallucination")
    n_correct = sum(1 for r in rows if r["label"] == "correct")
    n_abstain = sum(1 for r in rows if r["label"] == "abstain")
    print(f"DONE style={style} rows={len(rows)} hallucination={n_halluc} "
          f"correct={n_correct} abstain={n_abstain}", flush=True)

def main():
    for style in STYLES:
        print(f"\n=== running style={style} ===", flush=True)
        run_style(style)

if __name__ == "__main__":
    main()
