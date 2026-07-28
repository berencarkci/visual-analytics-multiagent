"""Evaluation agent: rule based consistency review of the final answer.

Deliberately LLM free: the last reviewer in the chain must itself be deterministic.
A reviewer that can hallucinate cannot be trusted to catch hallucinations. 
Seven mechanical checks cover schema validity, execution success, chart intent fit, insight groundedness, statistics health, wording consistency and composition integrity. 
The same checks will be applied to single agent baseline outputs later (shared reviewer = fair comparison, like the shared renderer).
"""

from __future__ import annotations

import math
import re

import pandas as pd

from insight import verify_grounded
from messages import ChartDecision, EvalVerdict, InsightResult, TransformPlan, WorkflowPlan
from visualization import ALLOWED_CHARTS

# Rule 6 vocabulary (correlation wording must match the computed r):
_STRENGTH_WORDS = ("weak", "moderate", "strong")
_DIRECTION_WORDS = ("positive", "negative")
#################################


# The seven checks:
def _check_schema(decision: ChartDecision, df: pd.DataFrame) -> str | None:
    rec = decision.recommendation
    for col in (rec.x_axis, rec.y_axis):
        if col and col not in df.columns and not re.match(r"^\w+\(.+\)$", col):
            return f"recommended column '{col}' not in the dataset"
    return None


def _check_execution(plan: TransformPlan) -> str | None:
    if plan.result_rows is None or plan.result_rows <= 0:
        return "transform produced no rows"
    return None


def _check_chart_intent(workflow: WorkflowPlan, decision: ChartDecision) -> str | None:
    chart = decision.recommendation.chart_type
    allowed = ALLOWED_CHARTS[workflow.intent]
    if chart not in allowed:
        return f"chart '{chart}' not allowed for intent '{workflow.intent}' (allowed: {allowed})"
    return None


def _check_grounded(ins: InsightResult) -> str | None:
    ok, problems = verify_grounded(ins.insight, ins.supporting_stats)
    if not ok:
        return f"insight contains numbers not in the computed stats: {problems}"
    return None


def _check_stats_health(plan: TransformPlan) -> str | None:
    s = plan.summary_stats
    if s.get("focus") == "correlation":
        r = s.get("pearson_r")
        if r is None or (isinstance(r, float) and math.isnan(r)) or s.get("n", 0) == 0:
            return "correlation stats are unhealthy (NaN r or n=0). insight cannot be trusted"
    if "stats_error" in s:
        return f"stats computation failed: {s['stats_error']}"
    return None


def _check_wording(plan: TransformPlan, ins: InsightResult) -> tuple[str | None, str | None]:
    """Correlation wording vs computed r, returns (issue, warning)"""
    s = plan.summary_stats
    if s.get("focus") != "correlation":
        return None, None
    text = ins.insight.lower()
    strength, direction, r = s.get("strength"), s.get("direction"), s.get("pearson_r")

    for w in _DIRECTION_WORDS:
        if w in text and direction and w != direction:
            return f"insight says '{w}' but computed direction is '{direction}' (r={r})", None
    for w in _STRENGTH_WORDS:
        if w in text and strength and w != strength:
            return f"insight says '{w}' but computed strength is '{strength}' (r={r})", None
    if r is not None and not (isinstance(r, float) and math.isnan(r)) and abs(r) < 0.1:
        if any(w in text for w in _DIRECTION_WORDS) or "correlat" in text or "associat" in text:
            return None, f"|r|={abs(r)} is negligible, wording may overstate the relationship"
    return None, None


def _check_composition(workflow: WorkflowPlan, plan: TransformPlan) -> str | None:
    if workflow.intent == "composition" and (plan.result_rows or 0) == 1:
        return "share question collapsed to a single group (100% share is meaningless)"
    return None
#################################

def _check_insight_informative(ins: InsightResult) -> str | None:
    """Warn if the insight collapsed to the last-resort generic template

    "The analysis produced N result rows" means no focus template matched and
    the statement carries no real finding — a soft signal, not a hard failure.
    """
    text = ins.insight.lower()
    if ins.source == "template_fallback" and "result row" in text:
        return "insight fell back to the generic template (no finding computed)"
    return None
#################################

def _check_data_sufficiency(plan: TransformPlan) -> str | None:
    """Warn when the statistics rest on fewer rows than the user handed in

    Missing values are dropped before a correlation is computed, so an insight
    can report "over 10 rows" for a 14-row upload without the user ever learning
    that 4 were discarded — and at that size the coefficient is not trustworthy
    anyway. The Data Analyst records both facts; this is what surfaces them.
    Warning, not an issue: dropping incomplete rows is correct behaviour, the
    problem is only that it happens invisibly.
    """
    s = plan.summary_stats or {}
    parts: list[str] = []
    dropped = s.get("n_rows_dropped_missing")
    if dropped:
        parts.append(f"{dropped} row(s) dropped for missing values")
    if s.get("caution"):
        parts.append(str(s["caution"]))
    return "; ".join(parts) if parts else None

# Agent entry point:
def run_evaluation(workflow: WorkflowPlan, plan: TransformPlan, decision: ChartDecision, ins: InsightResult, df: pd.DataFrame) -> EvalVerdict:
    """Run all seven checks, hard issues block, warnings ride along with the answer"""
    issues: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    for name, problem in [
        ("schema_valid", _check_schema(decision, df)),
        ("execution_ok", _check_execution(plan)),
        ("chart_intent_fit", _check_chart_intent(workflow, decision)),
        ("insight_grounded", _check_grounded(ins)),
        ("stats_health", _check_stats_health(plan)),
        ("composition_integrity", _check_composition(workflow, plan)),
    ]:
        checks[name] = problem is None
        if problem:
            issues.append(f"{name}: {problem}")

    w_issue, w_warn = _check_wording(plan, ins)
    checks["wording_consistency"] = w_issue is None
    if w_issue:
        issues.append(f"wording_consistency: {w_issue}")
    if w_warn:
        warnings.append(w_warn)

    info_warn = _check_insight_informative(ins)
    checks["insight_informative"] = info_warn is None
    if info_warn:
        warnings.append(info_warn)

    data_warn = _check_data_sufficiency(plan)
    checks["data_sufficiency"] = data_warn is None
    if data_warn:
        warnings.append(data_warn)

    return EvalVerdict(passed=len(issues) == 0, issues=issues, warnings=warnings, checks=checks)
#################################