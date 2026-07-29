"""Pre-training checks for the v5 mechanisms. No GPU, no model, seconds to run.

Every new mechanism and every new guardrail is exercised with a hand-written
plan fed through MockClient, so a failure here is a code failure and not a model
failure. Run this before generating training data: a bank whose targets do not
execute is silently dropped by make_agent_sft_data, and the training run then
teaches nothing while looking perfectly healthy.

Usage (from the repo root):
    python evaluation/check_v5_mechanisms.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "agents")

import pandas as pd

from data_analyst import run_data_analysis
from data_ingestion import profile_table, schema_summary
from messages import IntentResult, StepError
from model_client import MockClient
from schemas import ChartRecommendation, Transform
from supervisor import select_workflow
from transforms import apply_transform

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        _failures.append(name)
#################################


# A table with an id-like column (high cardinality) and two small categoricals:
def make_frame() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append({
            "oid": f"ID-{i:03d}",                       # 40 distinct -> not a colour
            "mode": ["Air", "Truck", "Rail"][i % 3],
            "seg": ["Consumer", "Corporate"][i % 2],
            "temp": float(i % 10),                       # 0..9, threshold at 5
            "profit": float(10 + (i % 7) * 3),
        })
    return pd.DataFrame(rows)


def plan_json(target_columns, **transform) -> str:
    full = {"groupby": None, "series": None, "agg": None,
            "filter": None, "sort": None, "limit": None}
    full.update(transform)
    return json.dumps({"target_columns": target_columns, "transform": full})


def run_plan(df, profile, raw_plan: str, intent: str = "comparison"):
    """Feed one fixed plan through the agent and return (plan, prepared)"""
    client = MockClient([raw_plan, raw_plan])          # second copy covers a retry
    workflow = select_workflow(IntentResult(intent=intent, source="llm"))
    return run_data_analysis(client, df, profile, schema_summary(profile),
                             "test question", workflow)
#################################


def main() -> int:
    df = make_frame()
    profile = profile_table(df, "test")

    print("\n--- engine: threshold_flag ---")
    rec = ChartRecommendation(chart_type="bar", x_axis="temp", y_axis="profit",
                              transform=Transform(groupby="threshold_flag(temp, 5)",
                                                  agg="mean"),
                              reason="-", insight="-")
    out, x, y, ser, notes = apply_transform(df, rec)
    check("threshold_flag splits into exactly two groups", len(out) == 2,
          f"{len(out)} rows: {list(out[x])}")
    check("threshold_flag sets no series", ser is None)

    print("\n--- engine: series ---")
    rec = ChartRecommendation(chart_type="bar", x_axis="mode", y_axis="profit",
                              transform=Transform(groupby="mode", series="seg",
                                                  agg="sum"),
                              reason="-", insight="-")
    out, x, y, ser, notes = apply_transform(df, rec)
    check("series produces one row per (group, series) pair", len(out) == 6,
          f"{len(out)} rows (3 modes x 2 segments)")
    check("series column is reported back", ser == "seg", f"series={ser}")
    check("measure survives as the y column", y == "sum(profit)", f"y={y}")

    print("\n--- engine: series identical to groupby ---")
    rec = ChartRecommendation(chart_type="bar", x_axis="mode", y_axis="profit",
                              transform=Transform(groupby="mode", series="mode",
                                                  agg="sum"),
                              reason="-", insight="-")
    out, x, y, ser, notes = apply_transform(df, rec)
    check("duplicate series is ignored", ser is None and len(out) == 3,
          "; ".join(notes) or "no note")

    print("\n--- guardrail: series listed in target_columns ---")
    plan, prepared = run_plan(df, profile,
                              plan_json(["mode", "seg"], groupby="mode",
                                        series="seg", agg="sum"))
    ok = not isinstance(plan, StepError)
    check("plan survives", ok, "" if ok else str(plan))
    if ok:
        check("series removed from target_columns",
              "seg" not in plan.target_columns, f"target_columns={plan.target_columns}")
        check("removal is recorded in the notes",
              any("target_columns" in n for n in plan.notes), "; ".join(plan.notes))

    print("\n--- guardrail: high-cardinality series ---")
    plan, prepared = run_plan(df, profile,
                              plan_json(["mode", "profit"], groupby="mode",
                                        series="oid", agg="sum"))
    ok = not isinstance(plan, StepError)
    check("plan survives", ok, "" if ok else str(plan))
    if ok:
        check("id-like series is dropped", plan.transform.series is None,
              f"series={plan.transform.series}")
        check("result stays at one row per group", plan.result_rows == 3,
              f"{plan.result_rows} rows")

    print("\n--- guardrail: readable series is kept ---")
    plan, prepared = run_plan(df, profile,
                              plan_json(["mode", "profit"], groupby="mode",
                                        series="seg", agg="sum"))
    ok = not isinstance(plan, StepError)
    if ok:
        check("small categorical series survives", plan.transform.series == "seg",
              f"series={plan.transform.series}")
        check("stats describe the pairs, not a collapsed axis",
              plan.result_rows == 6, f"{plan.result_rows} rows")
        groups = (plan.summary_stats or {}).get("groups") or {}
        check("group labels combine both dimensions",
              any("·" in k for k in groups), f"{list(groups)[:2]}")

    print("\n--- regression: single-value aggregate still works ---")
    plan, prepared = run_plan(df, profile,
                              plan_json(["profit"], agg="mean"))
    ok = not isinstance(plan, StepError)
    if ok:
        check("ungrouped aggregate collapses to one number",
              plan.result_rows == 1 and (plan.summary_stats or {}).get("focus") == "single_value",
              f"stats={ (plan.summary_stats or {}).get('value') }")

    print("\n--- regression: trend is never collapsed ---")
    plan, prepared = run_plan(df, profile, plan_json(["profit"], agg="mean"),
                              intent="trend")
    ok = not isinstance(plan, StepError)
    if ok:
        check("trend keeps every row", plan.result_rows == len(df),
              f"{plan.result_rows} rows")

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################