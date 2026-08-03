"""Prompt only single agent baseline runner.

Builds the few-shot messages (from prompts.py), calls the model through the client interface,
validates against the schema and retries only once with error feedback (only once because we don't want to build the whole response with our feedbacks)
"""

from __future__ import annotations

from dataclasses import dataclass

from model_client import ModelClient
from prompts import build_messages
from schemas import ChartRecommendation, validate_output

# Result record:
@dataclass
class BaselineResult:
    recommendation: ChartRecommendation | None
    valid: bool
    used_retry: bool
    error: str | None
    raw_first: str
    raw_retry: str | None = None
#################################


# Runner with the single retry policy:
def recommend(client: ModelClient, schema_summary: str, question: str) -> BaselineResult:
    """question + schema summary -> validated recommendation (or recorded failure)

    Policy: extract & validate: on failure retry ONCE feeding the error message back, if still failing record as invalid. 
    Retry usage is stored because "valid on first try" vs "rescued by retry" is itself a metric for the schema validity analysis.
    """
    messages = build_messages(schema_summary, question)

    first = client.generate(messages)
    rec, err = validate_output(first)
    if rec is not None:
        return BaselineResult(rec, True, False, None, first)

    retry_messages = messages + [
        {"role": "assistant", "content": first},
        {"role": "user", "content": (
            f"Your previous answer was rejected: {err} "
            "Return ONLY one valid JSON object matching the required schema."
        )},
    ]
    second = client.generate(retry_messages)
    rec, err2 = validate_output(second)
    if rec is not None:
        return BaselineResult(rec, True, True, None, first, second)

    return BaselineResult(None, False, True, err2, first, second)
#################################