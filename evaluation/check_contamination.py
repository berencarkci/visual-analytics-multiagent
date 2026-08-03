"""Contamination check: SFT questions vs the frozen 60 question benchmark.

Two layer comparison: normalized exact match, then fuzzy similarity (SequenceMatcher >= 0.85)
Any hit is listed and the script exits 1.
sft_train.jsonl is not valid until this passes clean. 
Comparing against test/split questions programmatically does not break the seal: no human reads them, no model trains on them.
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

FUZZY_THRESHOLD = 0.85
#################################


# Normalization:
def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
#################################


# Core check:
def check_contamination(sft_questions: list[str], benchmark_path: str | Path = "evaluation/benchmark.json") -> list[dict]:
    """Return a list of hits, empty list = clean"""
    bench = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    bench_qs = [(q["id"], _norm(q["question"])) for q in bench["questions"]]

    hits = []
    for sq in sft_questions:
        n = _norm(sq)
        for bid, bq in bench_qs:
            if n == bq:
                hits.append({"sft_question": sq, "benchmark_id": bid, "kind": "exact", "score": 1.0})
                continue
            score = SequenceMatcher(None, n, bq).ratio()
            if score >= FUZZY_THRESHOLD:
                hits.append({"sft_question": sq, "benchmark_id": bid, "kind": "fuzzy", "score": round(score, 3)})
    return hits
#################################


# CLI:
def main() -> int:
    path = Path("data/sft_train.jsonl")
    if not path.exists():
        print(f"not found: {path}. run make_sft_data.py first")
        return 1
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        questions.append(user.rsplit("Question:", 1)[-1].strip())

    hits = check_contamination(questions)
    if hits:
        print(f"CONTAMINATION: {len(hits)} overlap(s) with the benchmark:")
        for h in hits:
            print(f"  [{h['kind']} {h['score']}] {h['benchmark_id']}  <->  {h['sft_question'][:70]}")
        return 1
    print(f"clean: {len(questions)} SFT questions, no overlap with the 60 benchmark questions "
          f"(exact + fuzzy@{FUZZY_THRESHOLD})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#################################