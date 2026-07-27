"""Data Analyst agent: column mapping, safe transform execution, real stats.

Plans the data preparation with a small LLM call (validated against the
actual schema, one retry with error feedback), executes it through the
shared transform engine, then computes REAL summary statistics according
to the workflow's insight_focus. Those numbers are what the Insight Agent
will ground its statements in — the structural fix for the prompt-only
baseline's unsupported-insight weakness.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from data_ingestion import TableProfile
from messages import StepError, TransformPlan, WorkflowPlan
from model_client import ModelClient
from prompts import PLAN_SYSTEM
from schemas import Transform, extract_json_block
from transforms import ColumnNotFoundError, apply_transform, measure_base_columns
from schemas import ChartRecommendation  # transform execution reuses its container

# LLM planning prompt (data preparation ONLY — no chart choice, no insight):


def _build_plan_messages(schema_text: str, question: str, intent: str,
                         feedback: str | None = None) -> list[dict]:
    user = f"{schema_text}\n\nQuestion intent: {intent}\nQuestion: {question}"
    if feedback:
        user += f"\n\n{feedback}"
    return [{"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": user}]
#################################


# Plan validation against the real schema:
# Field-slot reminder fed back on a format error: small models sometimes put a
# value in the wrong Transform slot (e.g. agg="date_asc", which belongs in sort).
_FIELD_HINT = ('Field reminder: agg must be sum/mean/count/count_distinct; '
               'sort must be date_asc/value_desc; date_asc/value_desc are NOT agg values; '
               'groupby is a column or derived expression; filter is a pandas query string.')


def _validate_plan(raw: str, profile: TableProfile) -> tuple[TransformPlan | None, str | None]:
    """Parse LLM output, check schema conformity AND column existence"""
    block = extract_json_block(raw)
    if block is None:
        return None, "No JSON object found. Return ONLY the JSON object."
    try:
        data = json.loads(block)
        # sort only affects display order, so an invalid value is repaired and
        # recorded (like a visualization guardrail) instead of failing the plan
        dropped_sort = None
        tf = data.get("transform") or {}
        if tf.get("sort") not in (None, "date_asc", "value_desc"):
            dropped_sort = tf["sort"]
            tf["sort"] = None
            data["transform"] = tf
        plan = TransformPlan(
            transform=Transform(**(data.get("transform") or {})),
            target_columns=list(data.get("target_columns") or []),
        )
        if dropped_sort:
            plan.notes.append(f"dropped invalid sort value {dropped_sort!r}")
    except Exception as e:
        return None, f"Invalid plan format: {e}. {_FIELD_HINT}"

    valid_cols = {c.name for c in profile.columns}
    # A derived measure names the columns it needs inside the expression, so it
    # is validated by its base columns rather than as a literal column name.
    referenced: set[str] = set()
    for col in plan.target_columns:
        referenced |= set(measure_base_columns(col))
    t = plan.transform
    for expr in (t.groupby, t.filter):
        if expr:
            cleaned = re.sub(r"'[^']*'", "", str(expr))
            referenced |= {tok for tok in re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", cleaned)
                           if tok in valid_cols or "." in tok or tok.islower()}
    unknown = [c for c in referenced
               if c not in valid_cols
               and c not in {"month", "quarter", "week", "day", "year", "hour_of_day",
                             "day_of_week", "weekend_flag", "bins", "threshold_flag",
                             "and", "or", "in", "not"}]
    if unknown:
        return None, f"Unknown column(s): {unknown}. Use only columns from the schema."
    return plan, None
#################################


# Summary statistics per insight focus (REAL numbers for the Insight Agent):
def _compute_stats(focus: str, raw_df: pd.DataFrame, transformed: pd.DataFrame,
                   x_col: str, y_col: str | None, plan: TransformPlan) -> dict:
    s: dict = {"focus": focus, "n_rows_input": int(len(raw_df)), "n_rows_result": int(len(transformed))}
    try:
        if focus in ("group_stats", "share_stats") and y_col and y_col in transformed.columns:
            vals = transformed.set_index(x_col)[y_col].dropna()
            total = float(vals.sum())
            s |= {"groups": {str(k): round(float(v), 3) for k, v in vals.items()},
                  "top_group": str(vals.idxmax()), "top_value": round(float(vals.max()), 3),
                  "bottom_group": str(vals.idxmin()), "bottom_value": round(float(vals.min()), 3),
                  "total": round(total, 3)}
            if focus == "share_stats" and total > 0:
                s["shares_pct"] = {str(k): round(100 * float(v) / total, 1) for k, v in vals.items()}
        elif focus == "trend_stats" and y_col and y_col in transformed.columns:
            ser = transformed.set_index(x_col)[y_col].dropna()
            first, last = float(ser.iloc[0]), float(ser.iloc[-1])
            s |= {"first_x": str(ser.index[0]), "first_value": round(first, 3),
                  "last_x": str(ser.index[-1]), "last_value": round(last, 3),
                  "change_pct": round(100 * (last - first) / abs(first), 1) if first else None,
                  "peak_x": str(ser.idxmax()), "peak_value": round(float(ser.max()), 3),
                  "trough_x": str(ser.idxmin()), "trough_value": round(float(ser.min()), 3)}
        elif focus == "correlation":
            cols = [c for c in plan.target_columns if c in raw_df.columns][:2]
            if len(cols) == 2:
                sub = raw_df[cols].apply(pd.to_numeric, errors="coerce").dropna()
                r = float(sub[cols[0]].corr(sub[cols[1]]))
                strength = ("strong" if abs(r) >= 0.6 else
                            "moderate" if abs(r) >= 0.3 else "weak")
                s |= {"pearson_r": round(r, 3), "n": int(len(sub)),
                      "direction": "positive" if r > 0 else "negative",
                      "strength": strength, "columns": cols}
        elif focus == "distribution_stats":
            col = y_col if (y_col and y_col in transformed.columns) else x_col
            ser = pd.to_numeric(transformed[col], errors="coerce").dropna()
            s |= {"column": col, "mean": round(float(ser.mean()), 3),
                  "median": round(float(ser.median()), 3), "std": round(float(ser.std()), 3),
                  "min": round(float(ser.min()), 3), "max": round(float(ser.max()), 3),
                  "q1": round(float(ser.quantile(0.25)), 3), "q3": round(float(ser.quantile(0.75)), 3)}
        elif focus == "outlier_detection" and y_col and y_col in transformed.columns:
            ser = transformed.set_index(x_col)[y_col].dropna()
            q1, q3 = float(ser.quantile(0.25)), float(ser.quantile(0.75))
            iqr = q3 - q1
            hi, lo = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            outliers = ser[(ser > hi) | (ser < lo)]
            top = outliers.abs().sort_values(ascending=False).head(3)
            s |= {"iqr_low": round(lo, 3), "iqr_high": round(hi, 3),
                  "n_outliers": int(len(outliers)),
                  "top_outliers": {str(k): round(float(ser[k]), 3) for k in top.index}}
    except Exception as e:  # stats must never kill the pipeline
        s["stats_error"] = str(e)
    return s
#################################


# Agent entry point:
def run_data_analysis(client: ModelClient, df: pd.DataFrame, profile: TableProfile,
                      schema_text: str, question: str, workflow: WorkflowPlan,
                      feedback: str | None = None
                      ) -> tuple[TransformPlan | StepError, pd.DataFrame | None]:
    """Plan (LLM, 1 retry) -> execute safely -> compute focus stats

    Returns (TransformPlan, prepared_df) on success or (StepError, None).
    """
    messages = _build_plan_messages(schema_text, question, workflow.intent, feedback)
    first = client.generate(messages)
    plan, err = _validate_plan(first, profile)

    if plan is None:  # one retry with the error fed back
        retry = messages + [
            {"role": "assistant", "content": first},
            {"role": "user", "content": f"Your plan was rejected: {err} Return ONLY a corrected JSON object."},
        ]
        second = client.generate(retry)
        plan, err = _validate_plan(second, profile)
        if plan is None:
            return StepError(agent="data_analyst", error_type="invalid_llm_output",
                             detail=err or "unknown", recoverable=False), None
        plan.plan_source = "llm_retry"

    # PLAN GUARDRAIL (mechanical, mirrors the visualization guardrails):
    # a share/composition question needs the WHOLE data; a filter that pins the
    # groupby column to one value collapses the composition to a single 100% group.
    # Small models sometimes do this despite the prompt rule — so we enforce it here.
    t = plan.transform
    if (workflow.intent == "composition" and t.filter and t.groupby
            and re.search(rf"\b{re.escape(str(t.groupby))}\b\s*==", str(t.filter))):
        plan.transform = t.model_copy(update={"filter": None})
        plan.notes.append(
            f"plan guardrail: filter '{t.filter}' removed — share questions need all groups")

    # execute through the shared engine (reusing the recommendation container)
    x = plan.target_columns[0] if plan.target_columns else None
    y = plan.target_columns[1] if len(plan.target_columns) > 1 else None
    carrier = ChartRecommendation(chart_type="bar", x_axis=x or "", y_axis=y,
                                  transform=plan.transform, reason="-", insight="-")
    try:
        prepared, x_col, y_col, notes = apply_transform(df, carrier)
    except ColumnNotFoundError as e:
        return StepError(agent="data_analyst", error_type="missing_column",
                         detail=str(e), recoverable=False), None
    except Exception as e:
        return StepError(agent="data_analyst", error_type="execution_error",
                         detail=str(e), recoverable=False), None

    if prepared.empty:
        return StepError(agent="data_analyst", error_type="empty_result",
                         detail=f"Transform left 0 rows (filter: {plan.transform.filter})",
                         recoverable=True), None

    plan.notes = plan.notes + notes          # keep plan-repair notes, then transform notes
    plan.result_rows = int(len(prepared))
    plan.summary_stats = _compute_stats(workflow.insight_focus, df, prepared, x_col, y_col, plan)
    return plan, prepared
#################################