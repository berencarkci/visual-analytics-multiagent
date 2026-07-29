"""Pre-training diagnostic sweep: what does the prompt alone achieve?

Run this AFTER the prompt patches and BEFORE training v5. The prompts now
document two mechanisms the model has never seen an example of — threshold_flag
and series — so this run answers the question this project has already asked
twice: is documenting a rule enough, or does it take an example bank?

Both earlier answers were "not enough" (anomaly wording needed 6 -> 43 examples;
the intent modifier rule over-triggered until contrast twins were added). A
third measurement on two fresh mechanisms makes that a finding rather than an
anecdote — and either result is worth reporting.

Five groups:

  A  threshold_flag   new mechanism, no training examples yet
  B  series           new mechanism, no training examples yet
  C  under-trained    day_of_week 5, hour_of_day 5, weekend_flag 4, year 12
                      examples — below the ~20 this project found necessary
  D  subset framing   the relationship-with-a-filter regression (correct in v3,
                      lost in v4)
  E  regression guard behaviours that already work and must survive v5

No score is printed. Read the transforms: which mechanism did the model reach
for, and did the engine accept it.

Usage (from the repo root):
    python evaluation/probe_v5_gaps.py --adapter berencarkci/qwen2.5-3b-va-sft-v4
    python evaluation/probe_v5_gaps.py --adapter <id> --group A B
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "agents")

from data_ingestion import load_table, profile_table
from model_client import HFClient
from orchestrator import run_workflow

OUT_PATH = Path("evaluation/results/v5_gap_probe.json")

DATA_FILES = {
    "retail": "data/retail_sales_superstore.csv",
    "mall": "data/customer_analytics_mall.csv",
    "energy": "data/energy_consumption_hourly.csv",
}

# (group, dataset, question, what a correct plan would use)
PROBES = [
    # --- A: threshold_flag (new) ---
    ("A", "energy", "Compare the average appliance consumption on days when outdoor temperature was below 5 degrees versus other days.",
     "threshold_flag(t_out, 5) — a filter keeps one side and loses the comparison"),
    ("A", "energy", "Does energy use differ when the outdoor humidity is above 80 compared with the rest of the time?",
     "threshold_flag(rh_out, 80)"),
    ("A", "retail", "Compare profit on heavily discounted orders, discount above 0.3, against the others.",
     "threshold_flag(discount, 0.3)"),
    ("A", "retail", "Do large orders, quantity above 5, earn more profit than smaller ones?",
     "threshold_flag(quantity, 5)"),
    ("A", "mall", "Split the customers by whether their annual income is above 60 and compare spending score.",
     "threshold_flag(annual_income_k_usd, 60)"),
    ("A", "mall", "Are customers over 40 different from the younger ones in spending score?",
     "threshold_flag(age, 40) — 'different from the younger ones' is a two-group comparison"),

    # --- B: series / second grouping dimension (new) ---
    ("B", "retail", "Show profit per ship mode, broken down by segment.",
     "groupby=ship_mode, series=segment"),
    ("B", "retail", "What is the average profit of each ship mode divided by segment?",
     "same; this phrasing failed on the NLV corpus"),
    ("B", "retail", "Compare sales by region for each product category.",
     "groupby=region, series=category (or the reverse)"),
    ("B", "retail", "Show the monthly sales for each region.",
     "groupby=month(order_date), series=region"),
    ("B", "retail", "Break the order counts down by segment and ship mode.",
     "two dimensions, agg=count"),
    ("B", "mall", "How does spending score differ by gender across age groups?",
     "groupby=bins(age), series=gender"),
    ("B", "energy", "Show the appliance use by day of week, split by weekend versus weekday.",
     "groupby=day_of_week(date), series=weekend_flag(date)"),

    # --- C: under-trained mechanisms ---
    ("C", "retail", "Which day of the week brings the most sales?", "day_of_week (5 examples)"),
    ("C", "energy", "Compare the average appliance use across the days of the week.", "day_of_week"),
    ("C", "energy", "What time of day uses the most energy?", "hour_of_day (5 examples)"),
    ("C", "energy", "Show how appliance consumption changes through the hours of the day.", "hour_of_day"),
    ("C", "energy", "Do weekends use more energy than weekdays?", "weekend_flag (4 examples)"),
    ("C", "retail", "Compare sales on weekends against weekdays.", "weekend_flag"),
    ("C", "retail", "Compare the total sales of each year.", "year (12 examples)"),
    ("C", "retail", "How did each quarter perform on profit?", "quarter (22 examples)"),

    # --- D: subset framing (the v4 regression) ---
    ("D", "mall", "Show the relationship between age and spending score for female customers only.",
     "relationship, NOT filter_aggregation — v3 got this right, v4 lost it"),
    ("D", "retail", "Is there a link between discount and profit within the Technology category?",
     "relationship with a filter"),
    ("D", "mall", "Among the higher-income customers, is age related to spending score?",
     "relationship with a filter"),

    # --- E: regression guards (must not change in v5) ---
    ("E", "mall", "What is the distribution of ages?", "histogram, no groupby"),
    ("E", "retail", "What is the distribution of the category values?", "bar, groupby=category, agg=count"),
    ("E", "retail", "Which sub-categories are not profitable?", "sort=value_asc"),
    ("E", "retail", "What share of Technology sales comes from each region?",
     "filter=Technology, groupby=region"),
    ("E", "retail", "Compare total sales across product categories.", "plain comparison"),
    ("E", "energy", "Identify days with unusually high total appliance consumption.", "anomaly"),
]
#################################


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="berencarkci/qwen2.5-3b-va-sft-v4")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--group", nargs="*", default=None, help="A B C D E")
    args = ap.parse_args()

    probes = [p for p in PROBES if args.group is None or p[0] in args.group]
    frames = {k: load_table(v) for k, v in DATA_FILES.items()}
    profiles = {k: profile_table(df, k) for k, df in frames.items()}
    client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)

    rows, group_now = [], None
    for group, ds, question, expected in probes:
        if group != group_now:
            group_now = group
            print(f"\n{'=' * 72}\nGROUP {group}\n{'=' * 72}")
        row = {"group": group, "dataset": ds, "question": question, "expected": expected}
        try:
            wf = run_workflow(client, frames[ds], profiles[ds], question)
            row["ok"] = wf.ok
            if wf.recommendation is not None:
                r = wf.recommendation
                row.update(chart=r.chart_type, x=r.x_axis, y=r.y_axis,
                           transform=r.transform.model_dump(), insight=r.insight)
            if wf.error is not None:
                row["error"] = f"{wf.error.error_type}: {wf.error.detail}"[:160]
            row["intent"] = next((m["payload"].get("intent") for m in wf.trace
                                  if m.get("payload_type") == "IntentResult"), None)
            row["notes"] = next((m["payload"].get("notes") for m in wf.trace
                                 if m.get("payload_type") == "TransformPlan"), None)
            if wf.verdict is not None:
                row["eval_passed"] = wf.verdict.passed
                row["eval_issues"] = wf.verdict.issues
        except Exception as exc:
            row.update(ok=False, crashed=True, error=f"{type(exc).__name__}: {exc}"[:160])
            traceback.print_exc(limit=2)
        rows.append(row)

        t = row.get("transform") or {}
        print(f"\n{question}")
        print(f"  want:   {expected}")
        print(f"  intent: {row.get('intent')}  chart: {row.get('chart')}")
        print(f"  groupby={t.get('groupby')}  series={t.get('series')}  "
              f"agg={t.get('agg')}  filter={t.get('filter')}  sort={t.get('sort')}")
        if row.get("notes"):
            print(f"  notes:  {row['notes']}")
        if row.get("error"):
            print(f"  ERROR:  {row['error']}")
        print(f"  insight: {str(row.get('insight'))[:120]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"adapter": args.adapter, "rows": rows},
                                   indent=2, default=str), encoding="utf-8")
    used_new = sum(1 for r in rows
                   if (r.get("transform") or {}).get("series")
                   or "threshold_flag" in str((r.get("transform") or {}).get("groupby")))
    print(f"\n{'=' * 72}")
    print(f"{len(rows)} probes | chain stopped: {sum(1 for r in rows if not r.get('ok'))}")
    print(f"plans reaching for a NEW mechanism (series / threshold_flag): {used_new}")
    print(f"written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################