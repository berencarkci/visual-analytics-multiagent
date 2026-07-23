"""Smoke test for the multi agent workflow

Runs the full chain on a set of dev split benchmark questions (test split stays sealed) with the real model, prints compact results and saves the outputs to a JSON file for later side by side comparison with the single agent baseline.

Usage:
    python test_multiagent.py # 8 default dev questions
    python test_multiagent.py --ids retail_001 mall_006
    python test_multiagent.py --question "Compare sales by category" --dataset retail_sales_superstore
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "agents")

from data_ingestion import load_table, profile_table
from model_client import HFClient
from orchestrator import run_workflow, trace_view

# Default smoke set: dev split ids covering all 7 intents and all 3 datasets
DEFAULT_IDS = ["retail_001", "retail_005", "retail_010", "retail_013", "mall_006", "mall_011", "energy_006", "energy_017"]

DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}
#################################


# Question loading:
def load_questions(ids: list[str]) -> list[dict]:
    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in bench["questions"]}
    picked = []
    for qid in ids:
        q = by_id.get(qid)
        if q is None:
            print(f"  ! id not found, skipping: {qid}")
            continue
        if q.get("split") == "test":
            print(f"  ! {qid} is in the sealed test split — skipping (use dev ids only)")
            continue
        picked.append(q)
    return picked
#################################


# Runner:
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--question", default=None, help="ad-hoc question instead of benchmark ids")
    ap.add_argument("--dataset", default="retail_sales_superstore")
    ap.add_argument("--adapter", default=None, help="LoRA adapter: local path or Hub id")
    args = ap.parse_args()

    client = HFClient(adapter=args.adapter)
    frames, profiles = {}, {}
    results = []

    if args.question:
        items = [{"id": "adhoc", "dataset": args.dataset, "question": args.question}]
    else:
        items = load_questions(args.ids)

    for q in items:
        ds = q["dataset"]
        if ds not in frames:
            frames[ds] = load_table(DATA_FILES[ds])
            profiles[ds] = profile_table(frames[ds], ds)

        print(f"\n=== {q['id']} [{ds}] ===\nQ: {q['question']}")
        r = run_workflow(client, frames[ds], profiles[ds], q["question"])

        if not r.ok:
            print(f"  STOPPED: {r.error.error_type} — {r.error.detail[:100]}")
        else:
            rec = r.recommendation
            print(f"  chart: {rec.chart_type} | x={rec.x_axis} y={rec.y_axis}")
            print(f"  insight: {r.insight}")
            for row in trace_view(r.trace):
                print(f"    {row['title']}: {row['summary']}")

        results.append({"id": q["id"], "question": q["question"], "dataset": ds, "ok": r.ok,
                        "recommendation": rec.model_dump() if r.ok else None,
                        "insight": r.insight, "trace": r.trace,
                        "error": r.error.model_dump() if r.error else None})

    out = Path("evaluation/results/smoke_multiagent.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results -> {out} (for baseline comparison later)")


if __name__ == "__main__":
    main()
#################################