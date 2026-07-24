"""Visualization agent: constrained chart selection with guardrail review.

The LLM (swappable client: prompt only today, SFT/DPO checkpoints later) picks a chart from the intent's allowed list only, guardrails then verify the choice against the actual prepared data (category counts, negative values, discrete x) and correct it when a rule is violated. 
Every correction is recorded in guardrails_applied, the models raw choice stays visible for model level evaluation, the corrected chart is the system level output.
"""

from __future__ import annotations

import json

import pandas as pd

from messages import ChartDecision, StepError, TransformPlan, WorkflowPlan
from model_client import ModelClient
from schemas import ChartRecommendation, ChartType, extract_json_block
from prompts import VIZ_SYSTEM

# Intent -> allowed chart families (first=preferred, benchmark aligned):
ALLOWED_CHARTS: dict[str, list[str]] = {
    "trend": ["line", "bar"],
    "comparison": ["bar", "box"],
    "composition": ["pie", "bar"],
    "relationship": ["scatter", "box"],
    "distribution": ["histogram", "box"],
    "filter_aggregation": ["bar", "line"],
    "anomaly": ["line", "box"],
}
#################################


def _build_viz_messages(question: str, intent: str, data_summary: str,
                        allowed: list[str], feedback: str | None = None) -> list[dict]:
    user = (f"Question: {question}\nIntent: {intent}\n"
            f"Prepared data: {data_summary}\nAllowed chart types: {allowed}")
    if feedback:
        user += f"\n\n{feedback}"
    return [{"role": "system", "content": VIZ_SYSTEM},
            {"role": "user", "content": user}]
#################################


# Data facts the guardrails check against:
def _data_facts(raw_df: pd.DataFrame, prepared_df: pd.DataFrame, plan: TransformPlan) -> dict:
    x = plan.target_columns[0] if plan.target_columns else None
    y = plan.target_columns[1] if len(plan.target_columns) > 1 else None

    if plan.transform.groupby:
        n_categories = int(len(prepared_df)) # after groupby: one row per group
    elif x is not None and x in raw_df.columns:
        n_categories = int(raw_df[x].nunique())
    else:
        n_categories = 0

    numeric_cols = prepared_df.select_dtypes("number")
    has_negative = bool((numeric_cols < 0).any().any()) if not numeric_cols.empty else False

    x_discrete_numeric = bool(x is not None and x in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[x]) and raw_df[x].nunique() <= 12)
    return {"x": x, "y": y, "n_categories": n_categories, "has_negative": has_negative, "x_discrete_numeric": x_discrete_numeric, "n_rows": int(len(prepared_df))}
#################################


# Guardrail review (checks against real data, not hoped for model obedience):
def _apply_guardrails(chart: str, facts: dict, allowed: list[str]) -> tuple[str, list[str]]:
    applied: list[str] = []

    if chart not in allowed:
        applied.append(f"{chart}->{allowed[0]}: not allowed for this intent")
        chart = allowed[0]

    if chart == "pie" and facts["n_categories"] > 5:
        applied.append(f"pie->bar: {facts['n_categories']} categories (pie readable up to 5)")
        chart = "bar"

    if chart == "pie" and facts["has_negative"]:
        applied.append("pie->bar: metric contains negative values")
        chart = "bar"

    if chart == "scatter" and facts["x_discrete_numeric"]:
        applied.append(f"scatter->box: x takes few discrete values (overplotting)")
        chart = "box"

    return chart, applied
#################################


# Agent entry point:
def run_visualization(client: ModelClient, question: str, workflow: WorkflowPlan, plan: TransformPlan, raw_df: pd.DataFrame, prepared_df: pd.DataFrame,
                      feedback: str | None = None) -> ChartDecision | StepError:
    """LLM picks from the allowed list then guardrails verify against the data

    Returns a ChartDecision whose recommendation is render ready (source column convention + the Data Analyst's transform). 
    The insight field is left empty, the orchestrator fills it from the Insight Agent's result.
    """
    allowed = ALLOWED_CHARTS[workflow.intent]
    facts = _data_facts(raw_df, prepared_df, plan)
    summary = (f"{facts['n_rows']} rows; x={facts['x']} ({facts['n_categories']} categories), "
               f"y={facts['y']}; negatives={facts['has_negative']}")

    messages = _build_viz_messages(question, workflow.intent, summary, allowed, feedback)
    chart, reason, source = None, "", "llm"
    for attempt in range(2): # one retry on unparseable output
        raw = client.generate(messages)
        block = extract_json_block(raw)
        if block:
            try:
                data = json.loads(block)
                if data.get("chart_type") in ChartType.__args__:
                    chart = data["chart_type"]
                    reason = str(data.get("reason", ""))[:200]
                    break
            except Exception:
                pass
        if attempt == 0:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": 'Invalid. Return ONLY {"chart_type": "...", "reason": "..."}'},
            ]
            source = "llm_retry"

    applied: list[str] = []
    if chart is None: # both attempts failed go with the safe default
        chart = allowed[0]
        reason = "Default choice: model output was invalid twice."
        applied.append(f"invalid_llm_output->{allowed[0]}: default applied")

    chart, guardrail_notes = _apply_guardrails(chart, facts, allowed)
    applied += guardrail_notes

    rec = ChartRecommendation(
        chart_type=chart,
        x_axis=facts["x"] or "",
        y_axis=facts["y"],
        transform=plan.transform, # Data Analyst's plan, not touched
        reason=reason,
        insight="", # Insight Agent fills this
    )
    return ChartDecision(recommendation=rec, guardrails_applied=applied)
#################################