"""Prompt-only vs SFT preliminary comparison (Task B3/T4, checklist item 4).

Three configurations on the dev split (Decision C2), so the effect of training
can be separated from the effect of the prompt:

    A  base + long frozen prompt   the shipped baseline
    B  SFT  + short prompt          the shipped SFT system
    C  SFT  + long frozen prompt    isolates "how much came from training"

A vs B answers "is the new system better", B vs C answers "was the gain the
training or the prompt". Metrics are single-call, single-agent only: this task
measures the MODEL, not the multi-agent system (that comparison is B5).

The test split is never touched here.

Usage (from the repo root):
    python evaluation/compare_prompt_vs_sft.py --adapter outputs/sft-qwen2.5-3b
    python evaluation/compare_prompt_vs_sft.py --adapter <hub-id> --limit 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "agents")

import pandas as pd

from baseline import recommend
from data_ingestion import load_table, profile_table, schema_summary
from model_client import HFClient
from prompts import SFT_SYSTEM
from schemas import validate_output
from visualization import ALLOWED_CHARTS

DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}
OUT_PATH = Path("evaluation/results/sft_vs_prompt.json")
_NUMBER = re.compile(r"\d")
#################################


# Per-answer scoring (all mechanical, no LLM judge):
def score_answer(rec, df: pd.DataFrame, qtype: str) -> dict:
    """Four checks on one validated recommendation"""
    cols = set(df.columns)
    used = [c for c in (rec.x_axis, rec.y_axis) if c]
    return {
        "columns_exist": all(c in cols or re.match(r"^\w+\(.+\)$", c) for c in used),
        "chart_fits_type": rec.chart_type in ALLOWED_CHARTS.get(qtype, ()),
        # Decision A check: a single-call model cannot compute numbers, so any
        # digit in the insight is an unsupported claim by construction.
        "insight_has_numbers": bool(_NUMBER.search(rec.insight or "")),
        "has_transform": rec.transform.groupby is not None or rec.transform.agg is not None,
    }


def run_config(name: str, client: HFClient, questions: list[dict],
               schemas: dict, frames: dict, mode: str) -> dict:
    """Run one configuration over the dev questions and aggregate the scores"""
    rows, t0 = [], time.time()
    for q in questions:
        schema_text = schemas[q["dataset"]]
        if mode == "long":                          # frozen baseline prompt + retry policy
            result = recommend(client, schema_text, q["question"])
            rec, valid, used_retry = result.recommendation, result.valid, result.used_retry
        else:                                       # short SFT prompt, single call
            messages = [{"role": "system", "content": SFT_SYSTEM},
                        {"role": "user", "content": f"{schema_text}\n\nQuestion: {q['question']}"}]
            rec, err = validate_output(client.generate(messages))
            valid, used_retry = rec is not None, False

        row = {"id": q["id"], "type": q["type"], "valid": valid, "used_retry": used_retry}
        if rec:
            row.update(score_answer(rec, frames[q["dataset"]], q["type"]))
            row["chart_type"] = rec.chart_type
        rows.append(row)

    n = len(rows)
    ok = [r for r in rows if r["valid"]]
    def rate(key, source):
        return round(100 * sum(1 for r in source if r.get(key)) / max(len(source), 1), 1)

    return {
        "config": name,
        "n_questions": n,
        "schema_valid_pct": round(100 * len(ok) / n, 1),
        "used_retry_pct": rate("used_retry", rows),
        "columns_exist_pct": rate("columns_exist", ok),
        "chart_fits_type_pct": rate("chart_fits_type", ok),
        "has_transform_pct": rate("has_transform", ok),
        "insight_has_numbers_pct": rate("insight_has_numbers", ok),
        "seconds_per_question": round((time.time() - t0) / n, 1),
        "rows": rows,
    }
#################################


# Reporting:
METRICS = [
    ("schema_valid_pct", "schema valid", "higher"),
    ("used_retry_pct", "needed retry", "lower"),
    ("columns_exist_pct", "columns exist", "higher"),
    ("chart_fits_type_pct", "chart fits type", "higher"),
    ("has_transform_pct", "has transform", "higher"),
    ("insight_has_numbers_pct", "insight has numbers", "lower"),
    ("seconds_per_question", "sec / question", "lower"),
]


def print_table(results: list[dict]) -> None:
    width = max(len(m[1]) for m in METRICS) + 2
    header = "metric".ljust(width) + "".join(r["config"].rjust(22) for r in results)
    print("\n" + header)
    print("-" * len(header))
    for key, label, direction in METRICS:
        line = f"{label} ({direction})".ljust(width)
        line += "".join(f"{r[key]:>22}" for r in results)
        print(line)
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="local path or Hub id of the SFT adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N dev questions")
    args = ap.parse_args()

    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    questions = [q for q in bench["questions"] if q.get("split") == "dev"]
    if not questions:
        raise SystemExit("no dev-split questions found — run evaluation/make_split.py first")
    if args.limit:
        questions = questions[:args.limit]

    frames = {name: load_table(path) for name, path in DATA_FILES.items()}
    schemas = {name: schema_summary(profile_table(df, name)) for name, df in frames.items()}
    print(f"dev questions: {len(questions)}")

    base_client = HFClient(model_name=args.base, temperature=0.0)
    sft_client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)

    results = [
        run_config("A base+long", base_client, questions, schemas, frames, mode="long"),
        run_config("B sft+short", sft_client, questions, schemas, frames, mode="short"),
        run_config("C sft+long", sft_client, questions, schemas, frames, mode="long"),
    ]
    print_table(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"adapter": args.adapter, "results": results},
                                   indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################