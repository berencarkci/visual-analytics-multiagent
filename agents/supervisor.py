"""Supervisor agent: intent classification and workflow selection.

Classifies the user question into one of the 7 benchmark aligned intents using a small LLM call, with a keyword rule fallback so the pipeline never stalls on a bad model output. 
Then selects the workflow variant (mainly: what Insight Agent should compute).
"""

from __future__ import annotations

import json
import re

from messages import IntentLabel, IntentResult, WorkflowPlan
from model_client import ModelClient
from prompts import INTENT_SYSTEM



def _build_intent_messages(question: str) -> list[dict]:
    """Extracted so training data generators can reproduce the exact prompt"""
    return [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": f'Q: "{question}"'},
    ]


def _classify_with_llm(client: ModelClient, question: str) -> IntentResult | None:
    """One small LLM call, returns None on any invalid output (fallback takes over)"""
    messages = _build_intent_messages(question)
    raw = client.generate(messages)
    match = re.search(r'\{[^{}]*\}', raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return IntentResult(intent=data.get("intent"), source="llm")
    except Exception:
        return None
#################################


# Keyword rule fallback:
_RULES: list[tuple[IntentLabel, list[str]]] = [
    ("anomaly", ["anomal", "unusual", "outlier", "spike", "strange", "abnormal", "stand out"]),
    ("relationship", ["correlat", "relationship", "associated", "relate", "versus", " vs "]),
    ("distribution", ["distribution", "spread", "histogram", "variab", "clustered"]),
    ("composition", ["share", "proportion", "composition", "mix", "split across", "make up", "breakdown"]),
    ("trend", ["over time", "trend", "monthly", "weekly", "daily", "evolve", "develop", "change over"]),
    ("filter_aggregation", ["top ", "only", "above", "below", "older than", "at night", "in 20"]),
    ("comparison", ["compare", "difference", "across", "between", "highest", "lowest", "which"])]


def _classify_with_rules(question: str) -> IntentResult:
    """Deterministic keyword fallback, if can't match with any default is to go with comparison"""
    q = question.lower()
    for intent, keywords in _RULES:
        hits = [k for k in keywords if k in q]
        if hits:
            return IntentResult(intent=intent, source="rule_fallback", matched_keywords=hits)
    return IntentResult(intent="comparison", source="rule_fallback", matched_keywords=[])
#################################


# Workflow selection:
_INSIGHT_FOCUS = {
    "trend": "trend_stats",
    "comparison": "group_stats",
    "composition": "share_stats",
    "relationship": "correlation",
    "distribution": "distribution_stats",
    "filter_aggregation": "group_stats",
    "anomaly": "outlier_detection",
}

_STEPS = ["data_analysis", "visualization", "insight", "evaluation"]


def classify_intent(client: ModelClient, question: str) -> IntentResult:
    """LLM first, rule fallback second"""
    result = _classify_with_llm(client, question)
    return result if result is not None else _classify_with_rules(question)


def select_workflow(intent_result: IntentResult) -> WorkflowPlan:
    """Same 4 pipeline steps for every intent, what differs is the insight focus"""
    return WorkflowPlan(intent=intent_result.intent, steps=list(_STEPS), insight_focus=_INSIGHT_FOCUS[intent_result.intent])
#################################