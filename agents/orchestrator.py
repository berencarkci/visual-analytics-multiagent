"""Orchestrator: wires the multi agent workflow end to end.

question -> Supervisor (intent + workflow) -> Data Analyst (plan + stats) -> Visualization (guarded chart) -> Insight (grounded statement).
Every step is wrapped in an AgentMessage and logged to the trace, a StepError stops the chain gracefully and is itself logged. 
The Evaluation Agent joins this chain later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_analyst import run_data_analysis
from data_ingestion import TableProfile, profile_table, schema_summary
from insight import run_insight
from messages import AgentMessage, StepError
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
    trace: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Alias for ok, lets UI code treat baseline and workflow results uniformly"""
        return self.ok
#################################


# End to end run:
def run_workflow(client: ModelClient, df: pd.DataFrame, profile: TableProfile, question: str, log_dir: str = "logs") -> WorkflowResult:
    """Full multi agent pass for one question, never raises, always returns a trace"""
    logger = TraceLogger(log_dir=log_dir)
    # accept anything: if the caller passed schema text (or None) instead of a TableProfile, build the profile here. removes a whole class of caller bugs
    if not isinstance(profile, TableProfile):
        profile = profile_table(df, "dataset")
    schema_text = schema_summary(profile)

    # 1-2) Supervisor: intent + workflow
    intent = classify_intent(client, question)
    logger.log(AgentMessage.wrap("supervisor", 1, intent))
    workflow = select_workflow(intent)
    logger.log(AgentMessage.wrap("supervisor", 2, workflow))

    # 3) Data Analyst: plan + safe execution + real stats
    plan, prepared = run_data_analysis(client, df, profile, schema_text, question, workflow)
    if isinstance(plan, StepError):
        logger.log(AgentMessage.wrap("data_analyst", 3, plan))
        return WorkflowResult(ok=False, error=plan, trace=logger.get_trace())
    logger.log(AgentMessage.wrap("data_analyst", 3, plan))

    # 4) Visualization: guarded chart decision
    decision = run_visualization(client, question, workflow, plan, df, prepared)
    if isinstance(decision, StepError):
        logger.log(AgentMessage.wrap("visualization", 4, decision))
        return WorkflowResult(ok=False, error=decision, trace=logger.get_trace())
    logger.log(AgentMessage.wrap("visualization", 4, decision))

    # 5) Insight: grounded statement from the computed stats
    ins = run_insight(client, question, plan.summary_stats)
    logger.log(AgentMessage.wrap("insight", 5, ins))

    rec = decision.recommendation.model_copy(update={"insight": ins.insight})
    return WorkflowResult(ok=True, recommendation=rec, insight=ins.insight, trace=logger.get_trace())
#################################


# Trace presentation helper (Agent Trace tab, no chain of thought, only payloads):
_SUMMARIES = {
    "IntentResult": lambda p: f"intent = {p['intent']} (source: {p['source']})",
    "WorkflowPlan": lambda p: f"insight focus = {p['insight_focus']}",
    "TransformPlan": lambda p: (f"{p['result_rows']} rows after transform" + (f"; notes: {'; '.join(p['notes'])}" if p['notes'] else "")),
    "ChartDecision": lambda p: (f"chart = {p['recommendation']['chart_type']}" + (f"; guardrails: {'; '.join(p['guardrails_applied'])}"
                                    if p['guardrails_applied'] else "; no guardrail needed")),
    "InsightResult": lambda p: f"source = {p['source']}",
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