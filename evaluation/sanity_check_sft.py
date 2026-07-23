"""Quick sanity check of the SFT adapter (Task B3/T4, checklist item 3).

Runs 10 dev-split questions through the trained model with the SHORT prompt
and prints each recommendation, so the output can be eyeballed before any
quantitative comparison. This is the "does it produce sane JSON at all" gate,
not a measurement — evaluation/compare_prompt_vs_sft.py does the numbers.

Usage (from the repo root):
    python evaluation/sanity_check_sft.py --adapter outputs/sft-qwen2.5-3b
    python evaluation/sanity_check_sft.py --adapter berencarkci/qwen2.5-3b-va-sft
    python evaluation/sanity_check_sft.py            # no adapter = base model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "agents")

from data_ingestion import load_table, profile_table, schema_summary
from model_client import HFClient
from prompts import SFT_SYSTEM
from schemas import validate_output

N_QUESTIONS = 10
DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}
#################################


# Helpers:
def dev_questions(n: int) -> list[dict]:
    """First n dev-split questions, spread across datasets"""
    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    dev = [q for q in bench["questions"] if q.get("split") == "dev"]
    if not dev:
        raise SystemExit("no dev-split questions found — run evaluation/make_split.py first")

    by_type: dict[str, list] = {}                   # round-robin over question types,
    for q in dev:                                   # so all 7 intents show up in 10 slots
        by_type.setdefault(q["type"], []).append(q)

    picked = []
    while len(picked) < n and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop(0))
    return picked


def schema_texts() -> dict[str, str]:
    return {name: schema_summary(profile_table(load_table(path), name))
            for name, path in DATA_FILES.items()}
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="local path or Hub id of the LoRA adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)
    schemas = schema_texts()
    questions = dev_questions(N_QUESTIONS)

    label = args.adapter or "base model (no adapter)"
    print(f"sanity check: {len(questions)} dev questions | {label}\n")

    n_valid = 0
    for i, q in enumerate(questions, 1):
        messages = [{"role": "system", "content": SFT_SYSTEM},
                    {"role": "user", "content": f"{schemas[q['dataset']]}\n\nQuestion: {q['question']}"}]
        raw = client.generate(messages)
        rec, err = validate_output(raw)

        print(f"{i:2}. [{q['type']:19}] {q['question']}")
        if rec:
            n_valid += 1
            t = rec.transform
            print(f"    -> {rec.chart_type:9} x={rec.x_axis} y={rec.y_axis} "
                  f"| groupby={t.groupby} agg={t.agg} filter={t.filter} "
                  f"sort={t.sort} limit={t.limit}")
            print(f"    insight: {rec.insight}")
        else:
            print(f"    -> INVALID: {err}")
            print(f"    raw: {raw[:160]}")
        print()

    print(f"schema-valid: {n_valid}/{len(questions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################