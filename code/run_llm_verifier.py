"""t9: local LLM-as-verifier baseline (Reviewer 1, revision).

Unlike run_llm_judge.py (which is shown the gold answer and is used only for
offline label validation / kappa against human annotators), this script asks
an independent model to rate its own confidence that a candidate answer is
correct WITHOUT ever showing it the gold answer. That makes its output a
legitimate deployment-time risk signal, fairly comparable to semantic entropy
and P(True) on the same risk-coverage curve (those also never see gold).

Setup: same as run_llm_judge.py (ollama pull qwen2.5:14b; pip install requests)
Usage:
  python run_llm_verifier.py --in data/results_se_flan_t5_small_joined.json --out data/verifier_qwen_se_flan_t5_small.json [--limit N]
"""
import json, os, re, sys, time, argparse
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:14b")

VERIFIER_PROMPT = """You are an independent verifier checking another AI system's answer. You are NOT told the correct answer.

Question: {q}
Candidate answer: {ans}

Judge only from how the answer is phrased and how plausible it sounds as a specific, verifiable fact. Do not rely on your own world knowledge of the answer, since you may not know it and the question may concern a fictional or invented subject.
If the candidate declines to answer or says it does not know, reply exactly: ABSTAIN
Otherwise, reply with ONLY a number from 0 to 100: your confidence that this candidate answer is correct.
Confidence:"""

def verify(q, ans):
    r = requests.post(f"{HOST}/api/chat", json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": VERIFIER_PROMPT.format(q=q, ans=ans)}],
        "options": {"temperature": 0, "num_predict": 6}}, timeout=120)
    r.raise_for_status()
    out = r.json()["message"]["content"].strip()
    if "ABSTAIN" in out.upper():
        return None
    m = re.search(r"\d+(\.\d+)?", out)
    if not m:
        return None
    return max(0.0, min(100.0, float(m.group(0))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    try:
        requests.get(f"{HOST}/api/tags", timeout=5)
    except Exception:
        sys.exit(f"Cannot reach Ollama at {HOST}. Run `ollama serve` and `ollama pull {MODEL}`.")
    rows = json.load(open(a.inp))
    if a.limit:
        rows = rows[:a.limit]
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["qid"], r["cond"]) for r in out}
    t0 = time.time()
    for i, r in enumerate(rows):
        if (r["qid"], r["cond"]) in done:
            continue
        conf = verify(r["question"], r["answer"])
        rec = dict(r); rec["verifier_conf"] = conf
        out.append(rec)
        if i % 20 == 0:
            print(f"{i}/{len(rows)} t={time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print(f"verified {len(out)} records -> {a.out}")

if __name__ == "__main__":
    main()
