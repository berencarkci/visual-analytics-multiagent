"""Robustness probe: finds coverage gaps the benchmark cannot see.

The benchmark measures the system on three clean, uniform tables. Real input is
a file the user uploads, and the categorical-distribution bug ("what is the
distribution of cat categories" -> histogram on a text column) showed what that
costs: every distribution example in the training data uses a NUMERIC column,
so the model learned "distribution -> histogram, no groupby" as an
unconditional rule.

This probe is deliberately NOT part of the frozen benchmark. It is a diagnostic
run against the shipped system to decide what the next training set should
cover. Two parts:

  A) TASK COVERAGE   questions from the canonical analytic-task taxonomy
                     (Amar/Eagan/Stasko 2005; VLAT) that the 7-intent taxonomy
                     does not clearly cover: Retrieve Value, Find Extremum,
                     Determine Range, Cluster — plus categorical distribution.

  B) DATA SHAPE      the same system against a hostile table: categorical and
                     boolean columns, missing values, a column literally named
                     "count", non-English column names, numeric-looking text
                     values, few rows, duplicate dates.

Expected behaviour is recorded per question so the output can be read without
re-deriving intent by hand. "graceful" means: no crash, no invented number —
declining or answering a narrower question both count as passes.

Usage (from the repo root):
    python evaluation/probe_robustness.py --adapter berencarkci/qwen2.5-3b-va-sft-v3
    python evaluation/probe_robustness.py --adapter <id> --part A
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "agents")

import pandas as pd

from data_ingestion import load_table, profile_table
from model_client import HFClient
from orchestrator import run_workflow

OUT_PATH = Path("evaluation/results/robustness_probe.json")
HOSTILE_CSV = Path("data/samples/hostile_table.csv")

DATA_FILES = {
    "retail": "data/retail_sales_superstore.csv",
    "mall": "data/customer_analytics_mall.csv",
    "energy": "data/energy_consumption_hourly.csv",
}
#################################


# PART A — analytic tasks the intent taxonomy does not clearly cover:
TASK_PROBES = [
    # --- categorical distribution: the observed failure ---
    ("retail", "What is the distribution of the category values?",
     "categorical_distribution",
     "bar with groupby=category, agg=count — NOT a histogram on a text column"),
    ("retail", "How many rows fall into each shipping mode?",
     "categorical_distribution",
     "bar, groupby=ship_mode, agg=count"),
    ("mall", "What is the distribution of gender in the customer base?",
     "categorical_distribution",
     "bar or pie, groupby=gender, agg=count"),
    # contrast twin: same word, numeric column, histogram IS correct
    ("mall", "What is the distribution of ages?",
     "numeric_distribution_control",
     "histogram on age — control case, must stay correct"),

    # --- Retrieve Value: a single number, not a comparison ---
    ("retail", "What were the total sales in the West region during 2017?",
     "retrieve_value",
     "filter + single aggregate; a one-bar chart is acceptable, an invented number is not"),
    ("mall", "What is the average spending score across all customers?",
     "retrieve_value",
     "single aggregate over the whole table"),

    # --- Find Extremum: the single best/worst, not a ranking ---
    ("retail", "Which single sub-category is the most profitable?",
     "find_extremum",
     "sort value_desc with limit 1, or a full ranking whose insight names the top"),
    ("energy", "Which hour of the day has the highest appliance consumption?",
     "find_extremum",
     "groupby hour_of_day, the insight must name the peak hour"),

    # --- Determine Range: spread bounds ---
    ("retail", "What is the range of order profits?",
     "determine_range",
     "min/max must appear in the insight; histogram or box both fine"),

    # --- Cluster: mall is the classic segmentation dataset ---
    ("mall", "Are there natural groups of customers in this data?",
     "cluster",
     "no clustering mechanism exists — a scatter of income vs spending is the honest fallback"),
    ("mall", "Can the customers be segmented into distinct types?",
     "cluster",
     "same; must not invent cluster counts"),

    # --- multi-metric: two measures at once ---
    ("retail", "Compare sales and profit across regions.",
     "multi_metric",
     "only one y-axis exists; picking one and saying so beats silently dropping the other"),

    # --- year-over-year: two filtered periods ---
    ("retail", "How did 2018 sales compare with 2017?",
     "period_comparison",
     "needs two periods; a yearly groupby is an acceptable answer"),

    # --- negation ---
    ("retail", "Which sub-categories are not profitable?",
     "negation",
     "sort value_asc, or a filter on profit < 0 — value_desc would be the wrong end"),

    # --- proportion within a filtered subset ---
    ("retail", "What share of Technology sales came from the West region?",
     "nested_share",
     "filter to Technology, then compose by region — must not filter to West as well"),

    # --- OUT OF SCOPE: the system cannot do these; graceful failure is the pass ---
    ("retail", "Predict next month's sales.",
     "out_of_scope_forecast",
     "no forecasting exists — showing the historical trend is fine, inventing a forecast is not"),
    ("retail", "Why did profit drop in March?",
     "out_of_scope_causal",
     "no causal analysis — showing the series is fine, asserting a cause is not"),
    ("energy", "Show me the humidity in the basement.",
     "nonexistent_column",
     "no such column — must not silently substitute another one"),
]


# PART B — data-shape stress, run against the hostile table:
SHAPE_PROBES = [
    ("What is the distribution of the cat values?", "categorical_distribution",
     "bar, groupby=cat, agg=count"),
    ("How many records are there for each cat?", "categorical_distribution",
     "same, phrased as a count"),
    ("Compare the total count by cat.", "reserved_word_column",
     "the column is literally named 'count' — must not be confused with the count aggregation"),
    ("How did the amount change over time?", "non_english_column",
     "the date column is named 'tarih'"),
    ("Is there a relationship between amount and score?", "missing_values",
     "score has NaNs — correlation must handle them, not crash"),
    ("What is the distribution of the active flag?", "boolean_column",
     "boolean column, two groups"),
    ("What is the average amount per cat?", "tiny_table",
     "very few rows — statistics are fragile, must not overstate"),
]
#################################


def build_hostile_table() -> pd.DataFrame:
    """A table shaped like real user uploads, unlike the three clean samples"""
    return pd.DataFrame({
        # non-English column name, duplicate dates
        "tarih": ["2024-03-11", "2024-03-21", "2024-03-18", "2024-03-19", "2024-03-02",
                  "2024-03-14", "2024-03-03", "2024-03-02", "2024-03-08", "2024-03-19",
                  "2024-03-02", "2024-03-18", "2024-03-14", "2024-03-19"],
        # column named after an aggregation keyword
        "count": [5, 2, 4, 2, 3, 3, 18, 19, 19, 19, 8, 5, 5, 10],
        # numeric-LOOKING categorical values ("5e" can be read as scientific notation)
        "cat": ["6", "5e", "6", "5e", "6", "5e", "6", "5e", "5e", "6", "5e", "6", "5e", "5e"],
        "amount": [12.5, 8.0, 15.25, 9.5, 11.0, 7.75, 22.0, 19.5, 18.0, 20.25,
                   10.5, 13.0, 9.0, 16.5],
        # missing values
        "score": [3.0, None, 4.5, 2.0, None, 3.5, 5.0, 4.0, None, 4.5, 3.0, None, 2.5, 4.0],
        # boolean
        "active": [True, False, True, True, False, True, False, True, True, False,
                   True, False, True, True],
    })
#################################


def run_probe(client, df, profile, question: str, category: str,
              expected: str, dataset: str) -> dict:
    row = {"dataset": dataset, "question": question, "category": category,
           "expected": expected}
    try:
        wf = run_workflow(client, df, profile, question)
        row["ok"] = wf.ok
        if wf.recommendation is not None:
            r = wf.recommendation
            row.update(chart_type=r.chart_type, x_axis=r.x_axis, y_axis=r.y_axis,
                       transform=r.transform.model_dump(), insight=r.insight)
        if wf.verdict is not None:
            row["eval_passed"] = wf.verdict.passed
            row["eval_issues"] = wf.verdict.issues
            row["eval_warnings"] = wf.verdict.warnings
        if wf.error is not None:
            row["error"] = str(wf.error)[:300]
    except Exception as exc:
        row.update(ok=False, crashed=True, error=f"{type(exc).__name__}: {exc}"[:300])
        traceback.print_exc(limit=2)
    return row
#################################


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--part", default="AB", help="A, B or AB")
    args = ap.parse_args()

    client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)
    rows: list[dict] = []

    if "A" in args.part:
        frames = {k: load_table(v) for k, v in DATA_FILES.items()}
        profiles = {k: profile_table(df, k) for k, df in frames.items()}
        print(f"\n{'=' * 70}\nPART A — task coverage ({len(TASK_PROBES)} probes)\n{'=' * 70}")
        for ds, q, cat, exp in TASK_PROBES:
            r = run_probe(client, frames[ds], profiles[ds], q, cat, exp, ds)
            rows.append(r)
            print(f"\n[{cat}] {q}")
            print(f"  expected: {exp}")
            print(f"  -> {r.get('chart_type')} x={r.get('x_axis')} y={r.get('y_axis')} "
                  f"transform={r.get('transform')}")
            print(f"  insight: {str(r.get('insight'))[:160]}")

    if "B" in args.part:
        df = build_hostile_table()
        HOSTILE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(HOSTILE_CSV, index=False)
        prof = profile_table(df, "hostile")
        print(f"\n{'=' * 70}\nPART B — data shape ({len(SHAPE_PROBES)} probes)\n{'=' * 70}")
        print(f"table written to {HOSTILE_CSV}\ndtypes: {dict(df.dtypes.astype(str))}\n")
        for q, cat, exp in SHAPE_PROBES:
            r = run_probe(client, df, prof, q, cat, exp, "hostile")
            rows.append(r)
            print(f"\n[{cat}] {q}")
            print(f"  expected: {exp}")
            print(f"  -> {r.get('chart_type')} x={r.get('x_axis')} y={r.get('y_axis')} "
                  f"transform={r.get('transform')}")
            print(f"  insight: {str(r.get('insight'))[:160]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"adapter": args.adapter, "rows": rows}, indent=2, default=str), encoding="utf-8")

    crashed = [r for r in rows if r.get("crashed")]
    stopped = [r for r in rows if not r.get("ok")]
    print(f"\n{'=' * 70}\n{len(rows)} probes | crashed: {len(crashed)} | chain stopped: {len(stopped)}")
    print("No pass/fail score here on purpose: several probes have no single correct\n"
          "answer, and the point is to read the outputs and decide what to train.")
    print(f"written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################