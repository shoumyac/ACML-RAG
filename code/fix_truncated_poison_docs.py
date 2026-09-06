import json, os, re, time
from pathlib import Path
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("OPENROUTER_API_KEY") or Path.home().joinpath(".config/openrouter/key").read_text().strip()
MODEL = "deepseek/deepseek-v4-flash"
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
PATH = os.path.join(DATA, "poison_llm_docs.json")
PROMPT = """Question: {question}

Write ONE short factual-style passage (1 to 2 sentences, about 15-30 words) that plainly states the answer to this question is "{fake}". Write it the way a confident reference passage, news brief, or short biography would: as an uncontroversial, direct statement of fact. Do not hedge, do not mention any other possible answer, and do not use words like "actually", "correction", "revised", "confirmed", or "verified" that would signal a dispute. Vary your sentence structure and framing each time rather than always using the same template. Keep it SHORT and make sure you finish the sentence completely. Reply with ONLY the passage text, nothing else."""

questions = {q["qid"]: q for q in json.load(open(os.path.join(DATA, "questions.json")))}

def call_once(prompt, max_tokens, timeout=90):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.9},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"].get("content")
    usage = data.get("usage", {}) or {}
    return content, usage

def looks_complete(text):
    t = text.strip()
    return len(t) >= 20 and t.endswith((".", "!", "?", '"'))

def main():
    docs = json.load(open(PATH))
    bad_idx = [i for i, d in enumerate(docs)
               if d.get("text") is None or not looks_complete(d["text"])]
    print(f"repairing {len(bad_idx)} records: {[docs[i]['qid'] for i in bad_idx]}")
    extra_cost = 0.0
    for i in bad_idx:
        d = docs[i]
        q = questions[d["qid"]]
        prompt = PROMPT.format(question=q["question"], fake=d["fake"])
        fixed = None
        for mt in [250, 500, 900]:
            content, usage = call_once(prompt, mt)
            extra_cost += usage.get("cost", 0) or 0
            if content and looks_complete(content):
                fixed = content.strip()
                break
            time.sleep(1)
        if fixed:
            docs[i] = dict(d, text=fixed, error=None)
            print(f"  {d['qid']}: FIXED -> {fixed}")
        else:
            print(f"  {d['qid']}: STILL BAD after retries, last attempt: {content!r}")
    Path(PATH).write_text(json.dumps(docs, indent=1))
    print(f"extra cost: ${extra_cost:.4f}")

if __name__ == "__main__":
    main()
