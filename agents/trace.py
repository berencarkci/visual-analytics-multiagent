"""Structured trace logging for the multi agent workflow.

Every AgentMessage is written to a timestamped JSONL file (debugging, report examples) and kept in an in memory session list (the Agent Trace tab reads from this). 
We log structured messages, not text lines, which is why this is a small custom logger instead of the logging module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from messages import AgentMessage

# One line summaries per payload type (used by both the console log and the Agent Trace tab, so the two views always tell the same story):
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
 
 
def summarize(payload_type: str, payload: dict) -> str:
    """One human readable line for a logged payload"""
    return _SUMMARIES.get(payload_type, lambda p: "")(payload)
 

# Trace logger:
class TraceLogger:
    """One instance per user question, collects the full workflow trace"""

    def __init__(self, log_dir: str | Path = "logs", verbose: bool = True):
        self.verbose = verbose
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.log_dir / f"trace_{stamp}.jsonl"
        self.messages: list[AgentMessage] = []

    def log(self, msg: AgentMessage) -> None:
        self.messages.append(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg.model_dump_json() + "\n")
        if self.verbose: # live console view of the workflow
            summary = summarize(msg.payload_type, msg.payload.model_dump())
            clock = msg.timestamp.split("T")[1][:8]
            print(f"  [{clock}] step {msg.step}  {msg.agent:13} -> {msg.payload_type:14} | {summary}")

    def get_trace(self) -> list[dict]:
        """Trace as plain dicts, ready for the Agent Trace tab/JSON display"""
        return [m.model_dump() for m in self.messages]
#################################