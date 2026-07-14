"""Structured inter-agent message schemas for the multi agent workflow.

Every hop between agents is a typed pydantic payload wrapped in an AgentMessage envelope. 
No free form text travels between agents, each step's output is a validated contract. 
The envelope feeds the trace logger (and the Agent Trace tab) uniformly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, Field

from schemas import ChartRecommendation, Transform

# Intent taxonomy (Must stay identical to benchmark question types):
IntentLabel = Literal["trend", "comparison", "composition", "relationship", "distribution", "filter_aggregation", "anomaly"]
#################################


# Step payloads:
class IntentResult(BaseModel):
    """Supervisor output: classified question intent"""
    intent: IntentLabel
    source: Literal["llm", "rule_fallback"]
    matched_keywords: list[str] = Field(default_factory=list)  # filled by fallback


class WorkflowPlan(BaseModel):
    """Supervisor output: which pipeline variant to run for this intent"""
    intent: IntentLabel
    steps: list[str]
    insight_focus: Literal["group_stats", "trend_stats", "correlation", "distribution_stats", "outlier_detection", "share_stats"]


class TransformPlan(BaseModel):
    """Data Analyst output: the plan and its execution result"""
    transform: Transform
    target_columns: list[str]
    notes: list[str] = Field(default_factory=list)
    plan_source: Literal["llm", "llm_retry"] = "llm"
    result_rows: int | None = None # rows after transform execution
    summary_stats: dict = Field(default_factory=dict) # per insight_focus, feeds Insight Agent


class ChartDecision(BaseModel):
    """Visualization Agent output (skeleton)"""
    recommendation: ChartRecommendation
    guardrails_applied: list[str] = Field(default_factory=list)


class InsightResult(BaseModel):
    """Insight Agent output: statement + the computed stats backing it (skeleton)"""
    insight: str
    supporting_stats: dict = Field(default_factory=dict)


class StepError(BaseModel):
    """Structured failure any agent can return to the Supervisor"""
    agent: str
    error_type: Literal["missing_column", "type_mismatch", "empty_result", "invalid_llm_output", "execution_error"]
    detail: str
    recoverable: bool


class EvalVerdict(BaseModel):
    """Evaluation Agent output (skeleton)"""
    passed: bool
    issues: list[str] = Field(default_factory=list)
#################################


# Envelope:
Payload = Union[IntentResult, WorkflowPlan, TransformPlan, ChartDecision, InsightResult, EvalVerdict, StepError]


class AgentMessage(BaseModel):
    """Uniform envelope every agent emits, the unit the trace logger stores"""
    agent: str # e.g. "supervisor", "data_analyst"
    step: int # phase in the workflow
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload_type: str # class name of the payload, for readable traces
    payload: Payload

    @classmethod
    def wrap(cls, agent: str, step: int, payload: Payload) -> "AgentMessage":
        return cls(agent=agent, step=step, payload_type=type(payload).__name__, payload=payload)
#################################