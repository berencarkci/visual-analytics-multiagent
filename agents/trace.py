"""Structured trace logging for the multi agent workflow.

Every AgentMessage is written to a timestamped JSONL file (debugging, report examples) and kept in an in memory session list (the Agent Trace tab reads from this). 
We log structured messages, not text lines, which is why this is a small custom logger instead of the logging module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from messages import AgentMessage

# Trace logger:
class TraceLogger:
    """One instance per user question, collects the full workflow trace"""

    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.log_dir / f"trace_{stamp}.jsonl"
        self.messages: list[AgentMessage] = []

    def log(self, msg: AgentMessage) -> None:
        self.messages.append(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg.model_dump_json() + "\n")

    def get_trace(self) -> list[dict]:
        """Trace as plain dicts, ready for the Agent Trace tab/JSON display"""
        return [m.model_dump() for m in self.messages]
#################################