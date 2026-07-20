"""Fault injection tests for the Evaluation Agent.

Feeds deliberately broken payloads straight into run_evaluation and asserts every rule catches its target fault, then one chain level test verifies the orchestrator's targeted retry (bad insight -> retry -> template -> pass).

Usage: python test_eval_injection.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "agents")

import pandas as pd

from data_ingestion import load_table, profile_table
from evaluation import run_evaluation
from messages import ChartDecision, InsightResult, IntentResult, TransformPlan
from model_client import MockClient
from orchestrator import run_workflow
from schemas import ChartRecommendation, Transform
from supervisor import select_workflow

PASS = "OK"
FAIL = "FAIL"
results: list[bool] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append(cond)
    print(f"{PASS if cond else FAIL} {name}" + (f"  [{detail}]" if detail else ""))
#################################


# Shared fixtures:
df = load_table("data/retail_sales_superstore.csv")
profile = profile_table(df, "retail")


def wf(intent):
    return select_workflow(IntentResult(intent=intent, source="llm"))


def make_plan(rows=3, stats=None, groupby="category", agg="sum"):
    return TransformPlan(transform=Transform(groupby=groupby, agg=agg), target_columns=["category", "sales"], result_rows=rows, summary_stats=stats or {})


def make_decision(chart="bar", x="category", y="sales"):
    return ChartDecision(recommendation=ChartRecommendation(chart_type=chart, x_axis=x, y_axis=y, transform=Transform(groupby="category", agg="sum"), reason="t", insight=""))
#################################


# Injections, one per rule:
print("Rule-level injections")

good_stats = {"focus": "group_stats", "n_rows_result": 3,
              "groups": {"A": 10.0, "B": 20.0}, "top_group": "B", "top_value": 20.0,
              "bottom_group": "A", "bottom_value": 10.0, "total": 30.0}
clean = run_evaluation(wf("comparison"), make_plan(stats=good_stats), make_decision(), InsightResult(insight="B leads with 20.0.", supporting_stats=good_stats), df)
check("clean case passes all 7", clean.passed, str(clean.checks))

v = run_evaluation(wf("comparison"), make_plan(stats=good_stats), make_decision(x="ghost_column"), InsightResult(insight="B leads with 20.0.", supporting_stats=good_stats), df)
check("1 schema_valid catches ghost column", not v.checks["schema_valid"])

v = run_evaluation(wf("comparison"), make_plan(rows=0, stats=good_stats), make_decision(), InsightResult(insight="B leads with 20.0.", supporting_stats=good_stats), df)
check("2 execution_ok catches empty result", not v.checks["execution_ok"])

v = run_evaluation(wf("relationship"), make_plan(stats=good_stats), make_decision(chart="pie"), InsightResult(insight="B leads with 20.0.", supporting_stats=good_stats), df)
check("3 chart_intent_fit catches pie-for-relationship", not v.checks["chart_intent_fit"])

v = run_evaluation(wf("comparison"), make_plan(stats=good_stats), make_decision(), InsightResult(insight="B leads with a striking 999.5 total.", supporting_stats=good_stats), df)
check("4 insight_grounded catches invented number", not v.checks["insight_grounded"])

nan_stats = {"focus": "correlation", "pearson_r": float("nan"), "n": 0, "direction": "negative", "strength": "weak", "columns": ["a", "b"]}
v = run_evaluation(wf("relationship"), make_plan(stats=nan_stats, groupby=None, agg=None), make_decision(chart="scatter"), InsightResult(insight="There is a weak negative correlation.", supporting_stats=nan_stats), df)
check("5 stats_health catches NaN correlation", not v.checks["stats_health"])

neg_stats = {"focus": "correlation", "pearson_r": -0.327, "n": 200, "direction": "negative", "strength": "moderate", "columns": ["age", "spend"]}
v = run_evaluation(wf("relationship"), make_plan(stats=neg_stats, groupby=None, agg=None), make_decision(chart="scatter"), InsightResult(insight="age and spend show a moderate positive correlation (r=-0.327, n=200).", supporting_stats=neg_stats), df)
check("6 wording_consistency catches wrong direction", not v.checks["wording_consistency"])

tiny_stats = {"focus": "correlation", "pearson_r": 0.01, "n": 200, "direction": "positive", "strength": "weak", "columns": ["income", "spend"]}
v = run_evaluation(wf("relationship"), make_plan(stats=tiny_stats, groupby=None, agg=None), make_decision(chart="scatter"), InsightResult(insight="There is a weak positive association between income and spend.", supporting_stats=tiny_stats), df)
check("6b negligible-r produces WARNING (passes with note)", v.passed and len(v.warnings) == 1, v.warnings[0] if v.warnings else "")

share_stats = {"focus": "share_stats", "n_rows_result": 1, "shares_pct": {"Kentucky": 100.0}, "top_group": "Kentucky", "top_value": 139, "bottom_group": "Kentucky", "bottom_value": 139, "total": 139, "groups": {"Kentucky": 139}}
v = run_evaluation(wf("composition"), make_plan(rows=1, stats=share_stats), make_decision(chart="pie", x="state", y=None), InsightResult(insight="Kentucky has a share of 100.0%.", supporting_stats=share_stats), df)
check("7 composition_integrity catches single-group share (Kentucky case)", not v.checks["composition_integrity"])
#################################


# Chain level: bad insight -> targeted retry -> template fallback -> pass
print("\nChain-level retry")
def plan_json(cols, **t):
    base = {"groupby": None, "agg": None, "filter": None, "sort": None, "limit": None}
    return json.dumps({"target_columns": cols, "transform": {**base, **t}})

mall = load_table("data/customer_analytics_mall.csv")
client = MockClient([
    '{"intent": "relationship"}',
    plan_json(["age", "spending_score"]),
    '{"chart_type": "scatter", "reason": "two numerics"}',
    '{"insight": "A remarkable correlation of r=0.95 over 5000 customers."}', # invented
    '{"insight": "Still wrong: r=0.88 across 4000 people."}', # retry also invents
])
r = run_workflow(client, mall, profile_table(mall, "mall"), "Is age associated with spending score?", log_dir="/tmp/eval_tr")
check("first insight invented -> verifier already forces template (source)", r.trace[4]["payload"]["source"] == "template_fallback")
check("evaluation PASSED on delivered answer", r.verdict.passed, f"retried={r.verdict.retried_step}")
check("real r=-0.327 in final insight", "-0.327" in r.insight, r.insight)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
#################################