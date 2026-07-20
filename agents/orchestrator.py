"""Orchestrator: wires the multi agent workflow end to end.

question -> Supervisor (intent + workflow) -> Data Analyst (plan + stats) -> Visualization (guarded chart) -> Insight (grounded statement).
Every step is wrapped in an AgentMessage and logged to the trace, a StepError stops the chain gracefully and is itself logged. 
Evaluation (rule based consistency review).

If the evaluation fails, the orchestrator reruns only the step the failed
checks point at (one targeted retry), re-evaluates, and delivers either way a second failure ships the answer with the verdict attached instead of blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_analyst import run_data_analysis
from evaluation import run_evaluation
from data_ingestion import TableProfile, profile_table, schema_summary
from insight import run_insight
from messages import AgentMessage, EvalVerdict, StepError
from model_client import ModelClient
from schemas import ChartRecommendation
from supervisor import classify_intent, select_workflow
from trace import TraceLogger
from visualization import run_visualization

# Workflow result container:
@dataclass
class WorkflowResult:
    ok: bool
    recommendation: ChartRecommendation | None = None
    insight: str | None = None
    error: StepError | None = None
    verdict: EvalVerdict | None = None
    trace: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Alias for ok, lets UI code treat baseline and workflow results uniformly"""
        return self.ok
#################################

# Which agent did the failed checks point at (deepest cause selected):
_BLAME_ORDER = [
    ("data_analyst", ("execution_ok", "stats_health", "composition_integrity", "schema_valid")),
    ("visualization", ("chart_intent_fit",)),
    ("insight", ("insight_grounded", "wording_consistency")),
]


def _blame(verdict: EvalVerdict) -> str | None:
    failed = {name for name, ok in verdict.checks.items() if not ok}
    for agent, rules in _BLAME_ORDER:
        if failed & set(rules):
            return agent
    return None
#################################


# End to end run:
def run_workflow(client: ModelClient, df: pd.DataFrame, profile: TableProfile, question: str, log_dir: str = "logs") -> WorkflowResult:
    """Full multi agent pass for one question, never raises, always returns a trace"""
    logger = TraceLogger(log_dir=log_dir)
    # accept anything: if the caller passed schema text (or None) instead of a TableProfile, build the profile here. removes a whole class of caller bugs
    if not isinstance(profile, TableProfile):
        profile = profile_table(df, "dataset")
    schema_text = schema_summary(profile)
    step = 0

    def log(agent: str, payload) -> None:
        nonlocal step
        step += 1
        logger.log(AgentMessage.wrap(agent, step, payload))

    # 1-2) Supervisor: intent + workflow
    intent = classify_intent(client, question)
    log("supervisor", intent)
    workflow = select_workflow(intent)
    log("supervisor", workflow)

    # 3-5) Data Analyst -> Visualization -> Insight, wrapped so a targeted retry blamed on the analyst can rerun the dependent steps too:
    def run_analysis_chain():
        plan, prepared = run_data_analysis(client, df, profile, schema_text, question, workflow)
        if isinstance(plan, StepError):
            return plan, None, None, None
        log("data_analyst", plan)
        decision = run_visualization(client, question, workflow, plan, df, prepared)
        log("visualization", decision)
        ins = run_insight(client, question, plan.summary_stats)
        log("insight", ins)
        return plan, prepared, decision, ins

    plan, prepared, decision, ins = run_analysis_chain()
    if isinstance(plan, StepError):
        log("data_analyst", plan)
        return WorkflowResult(ok=False, error=plan, trace=logger.get_trace())

    # 6) Evaluation: rule based review
    verdict = run_evaluation(workflow, plan, decision, ins, df)
    log("evaluation", verdict)

    # Targeted single retry if the review failed:
    if not verdict.passed:
        blamed = _blame(verdict)
        if blamed == "data_analyst":
            plan2, prepared2, decision2, ins2 = run_analysis_chain()
            if not isinstance(plan2, StepError):
                plan, prepared, decision, ins = plan2, prepared2, decision2, ins2
        elif blamed == "visualization":
            decision = run_visualization(client, question, workflow, plan, df, prepared)
            log("visualization", decision)
        elif blamed == "insight":
            ins = run_insight(client, question, plan.summary_stats)
            log("insight", ins)

        verdict = run_evaluation(workflow, plan, decision, ins, df)
        verdict.retried_step = blamed
        verdict.retry_helped = verdict.passed
        log("evaluation", verdict)

    rec = decision.recommendation.model_copy(update={"insight": ins.insight})
    return WorkflowResult(ok=True, recommendation=rec, insight=ins.insight, verdict=verdict, trace=logger.get_trace())
#################################


# Trace presentation helper (Agent Trace tab, no chain of thought, only payloads):
_SUMMARIES = {
    "IntentResult": lambda p: f"intent = {p['intent']} (source: {p['source']})",
    "WorkflowPlan": lambda p: f"insight focus = {p['insight_focus']}",
    "TransformPlan": lambda p: (f"{p['result_rows']} rows after transform" + (f"; notes: {'; '.join(p['notes'])}" if p['notes'] else "")),
    "ChartDecision": lambda p: (f"chart = {p['recommendation']['chart_type']}" + (f"; guardrails: {'; '.join(p['guardrails_applied'])}"
                                    if p['guardrails_applied'] else "; no guardrail needed")),
    "InsightResult": lambda p: f"source = {p['source']}",
    "EvalVerdict": lambda p: (("PASSED" if p["passed"] else "FAILED") + (f"; issues: {'; '.join(p['issues'])}" if p["issues"] else "") + (f"; warnings: {'; '.join(p['warnings'])}" if p["warnings"] else "") + (f"; retried: {p['retried_step']}" if p.get("retried_step") else "")),
    "StepError": lambda p: f"{p['error_type']}: {p['detail'][:80]}",
}


def trace_view(trace: list[dict]) -> list[dict]:
    """[{title, summary, detail_json}] ready for the Agent Trace tab"""
    rows = []
    for msg in trace:
        fn = _SUMMARIES.get(msg["payload_type"], lambda p: "")
        rows.append({
            "title": f"Step {msg['step']} — {msg['agent']} ({msg['payload_type']})",
            "summary": fn(msg["payload"]),
            "detail_json": msg["payload"],
        })
    return rows
#################################