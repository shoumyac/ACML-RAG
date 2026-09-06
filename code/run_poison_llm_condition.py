"""t10: run the LLM-authored poison passages (generate_poison_llm_docs.py)
through the SAME flan-t5-small pipeline as run_semantic_entropy.py, as a new
"poison_llm" condition, held out entirely from the estimator's training data
so its generalisation to a genuinely different poisoning style can be tested
honestly (not cross-validated on data it was fit on).

Each question's poison_llm document is added to a COPY of docs (not the
shared docs.json) so BM25 retrieval scores it exactly like every other
document rather than assigning it a synthetic relevance score. Context is
[poison_llm doc] + 3 random irrelevant docs, shuffled, mirroring the
existing contra_only/poison condition shape.
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

docs_base = json.load(open(f"{DATA}/docs.json"))
questions_all = json.load(open(f"{DATA}/questions.json"))
poison_llm = {d["qid"]: d for d in json.load(open(f"{DATA}/poison_llm_docs.json"))}
SUBSET = 4
questions = [q for i, q in enumerate(questions_all)
             if q["regime"] == "real" or (i % SUBSET == 0)]
assert all(q["qid"] in poison_llm for q in questions)

# augmented doc list: base docs + one poison_llm doc per selected question
docs = list(docs_base)
poison_llm_idx = {}
for q in questions:
    pd = poison_llm[q["qid"]]
    poison_llm_idx[q["qid"]] = len(docs)
    docs.append({"id": f"pllm_{q['qid']}", "text": pd["text"], "entity": q["entity"],
                 "attr": q["attr"], "kind": "poison_llm", "value": pd["fake"]})

bm25 = BM25Okapi([re.findall(r"\w+", d["text"].lower()) for d in docs])
by_kind = {}
for i, d in enumerate(docs):
    by_kind.setdefault(d["kind"], []).append(i)

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
ABST = re.compile(r"(don.?t know|unknown|cannot|not (mentioned|stated|provided|given)|no answer)")
def label(ans, gold):
    a, g = norm(ans), norm(gold)
    if not a or ABST.search(a): return "abstain"
    if g in a or (a in g and len(a) > 2): return "correct"
    if re.fullmatch(r"\d{4}", g) and g in re.findall(r"\d{4}", a): return "correct"
    return "hallucination"

def prompt_for(q, ctx_ids):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        return (f"Answer using the context. If the context does not contain the "
                f"answer, answer unknown.\nContext:\n{ctx}\nQ: {q} A:")
    return f"Q: {q} A:"

tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval(); torch.set_num_threads(2)
print("loaded", MODEL, "augmented docs", len(docs), "questions", len(questions), flush=True)

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

def atomic(obj, path):
    t = path + ".tmp"; json.dump(obj, open(t, "w")); os.replace(t, path)

def main():
    out_path = f"{DATA}/results_poison_llm_flan_t5_small.json"
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {r["qid"] for r in rows}
    for qi, q in enumerate(questions):
        if q["qid"] in done: continue
        rng = random.Random(9000 + qi)  # separate seed stream from the main pilot's 4000+qi
        pidx = poison_llm_idx[q["qid"]]
        fillers = rng.sample([i for i in by_kind["irrelevant"]], 3)
        ctx = [pidx] + fillers
        rng.shuffle(ctx)
        sc_full = bm25.get_scores(re.findall(r"\w+", q["question"].lower()))
        cs = sorted([float(sc_full[i]) for i in ctx], reverse=True)
        feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                    ret_margin=cs[0] - cs[1], ret_min=cs[-1], has_ctx=1.0)
        pr = prompt_for(q["question"], ctx)
        g = greedy(pr); lab = label(g, q["gold"])
        sm = sample(pr, NSAMP)
        se, nclu = discrete_semantic_entropy(sm)
        pt = p_true(q["question"], ctx, g)
        rows.append(dict(qid=q["qid"], regime=q["regime"], cond="poison_llm", gold=q["gold"],
                         answer=g, label=lab, se=se, se_nclusters=nclu, p_true=pt,
                         poison_fake_value=poison_llm[q["qid"]]["fake"],
                         poison_text=poison_llm[q["qid"]]["text"], **feat))
        done.add(q["qid"])
        if qi % 20 == 0:
            print(f"q {qi}/{len(questions)} rows={len(rows)}", flush=True)
            atomic(rows, out_path)
    atomic(rows, out_path)
    n_halluc = sum(1 for r in rows if r["label"] == "hallucination")
    n_correct = sum(1 for r in rows if r["label"] == "correct")
    n_abstain = sum(1 for r in rows if r["label"] == "abstain")
    print(f"DONE rows={len(rows)} hallucination={n_halluc} correct={n_correct} abstain={n_abstain}", flush=True)

if __name__ == "__main__":
    main()
