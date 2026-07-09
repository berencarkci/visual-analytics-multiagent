"""Stratified dev/test split for the benchmark.

Assigns each benchmark question to 'dev' or 'test' with a seeded, stratified split (by question type x dataset), then rewrites benchmark.json.
Run once, commit the result, and dont touch the test split afterwards.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

BENCHMARK_PATH = Path("evaluation/benchmark.json")
TEST_RATIO = 0.30
SEED = 42

# Split logic:
def assign_splits(questions: list[dict], test_ratio: float, seed: int) -> list[dict]:
    """Assign dev/test per (type, dataset) stratum so every stratum is represented in test"""
    rng = random.Random(seed)

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        strata[(q["type"], q["dataset"])].append(q)

    for key, group in sorted(strata.items()):
        group.sort(key=lambda q: q["id"]) # deterministic order before shuffle
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))# every stratum gets >=1 test question
        for i, q in enumerate(group):
            q["split"] = "test" if i < n_test else "dev"

    return questions
#################################


# Report helper:
def split_report(questions: list[dict]) -> str:
    """Small text report: totals and per type test counts, for a quick sanity check"""
    n_test = sum(1 for q in questions if q["split"] == "test")
    n_dev = len(questions) - n_test

    per_type: dict[str, int] = defaultdict(int)
    for q in questions:
        if q["split"] == "test":
            per_type[q["type"]] += 1

    lines = [f"total={len(questions)}  dev={n_dev}  test={n_test}"]
    for t, c in sorted(per_type.items()):
        lines.append(f"  test[{t}] = {c}")
    return "\n".join(lines)
###################################


# Entry point:
if __name__ == "__main__":
    bench = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    already = [q for q in bench["questions"] if q.get("split")]
    if already:
        raise SystemExit(
            f"{len(already)} questions already have a split. "
            "The test split is frozen, refusing to overwrite. "
            "Delete the split fields manually only if you know what you are doing."
        )

    bench["questions"] = assign_splits(bench["questions"], TEST_RATIO, SEED)
    bench["split_config"] = {"test_ratio": TEST_RATIO, "seed": SEED, "method": "stratified(type x dataset)"}

    BENCHMARK_PATH.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    print(split_report(bench["questions"]))
    print("benchmark.json updated - commit this file to freeze the split.")
#########################