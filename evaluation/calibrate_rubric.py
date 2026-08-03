"""Rubric calibration.

Ten hand built cases, each one a failure mode observed in this project, checked against the rubric. 
The point is not code coverage: it is to show that the rubric orders candidates the way a reviewer would, before it is used to label 450 pairs unattended.

Every case states what the reviewer's judgement is and why. 
If the rubric disagrees with the reviewer, the rubric is wrong and gets fixed.
The cases are the specification.

Usage:
    python evaluation/calibrate_rubric.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "agents")
sys.path.insert(0, "evaluation")

from data_ingestion import load_table
from rubric import compare, score_candidate

RETAIL = "data/retail_sales_superstore.csv"
#################################


# Reference answers (the ground truth the candidates are scored against):
REF_TREND = {"x_axis": "order_date", "y_axis": "sales", "chart_family": ["line"],
             "target_columns": ["order_date", "sales"],
             "transform": {"groupby": "month(order_date)", "agg": "sum", "filter": None, "sort": "date_asc", "limit": None}}

REF_RELATION = {"x_axis": "discount", "y_axis": "profit", "chart_family": ["scatter", "box"],
                "target_columns": ["discount", "profit"],
                "transform": {"groupby": None, "agg": None, "filter": None, "sort": None, "limit": None}}

REF_FILTERED = {"x_axis": "order_date", "y_axis": "sales", "chart_family": ["line"],
                "target_columns": ["order_date", "sales"],
                "transform": {"groupby": "month(order_date)", "agg": "sum", "filter": "category == 'Technology'", "sort": "date_asc", "limit": None}}

STATS = {"focus": "correlation", "pearson_r": -0.219, "n": 9994,
         "direction": "negative", "strength": "weak",
         "columns": ["discount", "profit"]}
#################################


# Calibration cases: (name, format, candidate A, candidate B, expected, why)
CASES = [
    ("ghost column loses to real column", "data_analyst",
     {"target_columns": ["order_date", "sales"], "transform": REF_TREND["transform"]},
     {"target_columns": ["order_date", "revenue"], # revenue does not exist
      "transform": REF_TREND["transform"]},
     "a", "A column that is not in the table makes the answer unexecutable."),

    ("correct granularity beats wrong granularity", "data_analyst",
     {"target_columns": ["order_date", "sales"], "transform": REF_TREND["transform"]},
     {"target_columns": ["order_date", "sales"],
      "transform": {"groupby": "day(order_date)", "agg": "sum", "filter": None, "sort": "date_asc", "limit": None}},
     "a", "Daily instead of monthly answers a different question."),

    ("kept filter beats dropped filter", "data_analyst",
     {"target_columns": ["order_date", "sales"], "transform": REF_FILTERED["transform"]},
     {"target_columns": ["order_date", "sales"],
      "transform": {"groupby": "month(order_date)", "agg": "sum", "filter": None, "sort": "date_asc", "limit": None}},
     "a", "Dropping the filter silently widens the question."),

    ("missing sort is a minor flaw, not a wrong answer", "data_analyst",
     {"target_columns": ["order_date", "sales"], "transform": REF_TREND["transform"]},
     {"target_columns": ["order_date", "sales"],
      "transform": {"groupby": "month(order_date)", "agg": "sum", "filter": None, "sort": None, "limit": None}},
     "unclear", "Same numbers, different presentation — too close to auto-label."),

    ("right columns beat wrong columns", "data_analyst",
     {"target_columns": ["discount", "profit"], "transform": REF_RELATION["transform"]},
     {"target_columns": ["order_id", "discount"], # the observed NaN-correlation bug
      "transform": REF_RELATION["transform"]},
     "a", "order_id in a correlation produces NaN; the chain then fails."),

    ("allowed chart beats disallowed chart", "visualization",
     {"chart_type": "scatter", "reason": "A scatter plot shows how two numeric variables move together."},
     {"chart_type": "bar", "reason": "Bars compare the two variables."},
     "a", "bar is not in the allowed list for relationship; guardrails would override it."),

    ("reason naming another chart is penalised", "visualization",
     {"chart_type": "box", "reason": "Grouped boxes show the profit spread per discount level."},
     {"chart_type": "box", "reason": "A scatter plot shows how two numeric variables move together."},
     "unclear", "Same chart, stale reason — a real flaw but a small one."),

    ("grounded insight beats invented number", "insight",
     {"insight": "discount and profit show a weak negative correlation (r=-0.219, n=9994)."},
     {"insight": "discount and profit show a weak negative correlation (r=-0.480, n=9994)."},
     "a", "-0.480 is nowhere in the statistics."),

    ("specific insight beats generic template", "insight",
     {"insight": "discount and profit show a weak negative correlation (r=-0.219, n=9994)."},
     {"insight": "The analysis produced 9994 result rows."},
     "a", "The template says nothing about the question."),

    ("correct intent beats wrong intent", "supervisor",
     {"intent": "filter_aggregation"},
     {"intent": "relationship"}, # the observed energy_016 bug
     "a", "A wrong intent routes the whole chain to the wrong statistics."),
]
#################################


def main() -> int:
    df = load_table(RETAIL)
    refs = {"data_analyst": None, "visualization": None, "insight": None, "supervisor": None}

    passed = 0
    for name, fmt, cand_a, cand_b, expected, why in CASES:
        # pick the reference and context that fits the case
        if "relation" in name or "columns" in name or "chart" in name or "reason" in name:
            ref, intent = REF_RELATION, "relationship"
        elif "filter" in name:
            ref, intent = REF_FILTERED, "filter_aggregation"
        elif "insight" in name or "generic" in name:
            ref, intent = REF_RELATION, "relationship"
        elif fmt == "supervisor":
            ref, intent = {"intent": "filter_aggregation"}, None
        else:
            ref, intent = REF_TREND, "trend"

        stats = STATS if fmt == "insight" else None
        sa = score_candidate(cand_a, ref, fmt, df=df, intent=intent, stats=stats)
        sb = score_candidate(cand_b, ref, fmt, df=df, intent=intent, stats=stats)
        got = compare(sa, sb)

        ok = got == expected
        passed += ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
        print(f"A={sa['total']:5.1f} {sa['dimensions']}")
        print(f"B={sb['total']:5.1f} {sb['dimensions']}")
        print(f"expected={expected} got={got}   ({why})")
        print()

    print(f"{passed}/{len(CASES)} calibration cases behave as a reviewer would")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
#################################