"""Structured output schemas for the Visual Analytics Assistant.

Defines the JSON contract that every model configuration (prompt-only, SFT, DPO) and later the Visualization Agent must follow.
Helpers for parsing and validating raw model output.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# Output contract:
ChartType = Literal["bar", "line", "scatter", "pie", "histogram", "box"]
AggType = Literal["sum", "mean", "count", "count_distinct"]


class Transform(BaseModel):
    """Data preparation step. It mirrors the benchmark ground truth format"""
    groupby: str | None = None
    agg: AggType | None = None
    filter: str | None = None
    sort: Literal["date_asc", "value_desc"] | None = None
    limit: int | None = None


class ChartRecommendation(BaseModel):
    """The full structured answer a model must return for one question"""
    chart_type: ChartType
    x_axis: str
    y_axis: str | None = None # histogram/box doesnt always need a y axis
    transform: Transform = Field(default_factory=Transform)
    reason: str # why this chart fits, the intent explanation
    insight: str  # grounded statement, no unsupported claims

    model_config = {"extra": "forbid"} # unknown fields = schema violation
#################################


# Raw output parsing:
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(text: str) -> str | None:
    """Pull the first JSON object out of raw model text

    Handles common failure modes of small models: markdown code fences, preamble text before the JSON, trailing commentary after it vb.
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced braces
#################################


# Validation entry point:
def validate_output(raw_text: str) -> tuple[ChartRecommendation | None, str | None]:
    """Validate raw model text against the contract

    Returns (recommendation, None) on success or (None, error_message) on failure. 
    The error message is written to be fed back to the model on the retry.
    """
    block = extract_json_block(raw_text)
    if block is None:
        return None, "No JSON object found in the output. Return ONLY a JSON object."

    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON syntax: {e.msg} at position {e.pos}. Return valid JSON."

    try:
        rec = ChartRecommendation.model_validate(data)
    except ValidationError as e:
        issues = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:3])
        return None, f"Schema validation failed: {issues}"

    return rec, None
#################################