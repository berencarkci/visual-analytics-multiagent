"""Insight agent: grounded statements from computed statistics.

The LLM writes a short insight using only the numbers the Data Analyst computed. 
A mechanical verifier then checks every number in the sentence actually exists in the stats dict. 
Any invented number (or an empty/broken output) drops the result to a deterministic template, so an unsupported claim can never reach the user. 
The verifier is also the prototype of the groundedness metric.
"""

from __future__ import annotations

import json
import re

from messages import InsightResult
from model_client import ModelClient
from prompts import INSIGHT_SYSTEM

# The Insight agent gets one sentence to write, so a group dictionary with 48
# months or 138 days in it is not information, it is noise: the model cannot
# use that many numbers and tends to echo the dump back as its "insight".
# top_group / bottom_group / total already carry the answer. Trimming here (at
# the prompt layer, not in the stats themselves) keeps the trace and the
# groundedness check working on the full data.
_MAX_GROUPS_SHOWN = 8


def _compact_stats(stats: dict) -> dict:
    """Trim oversized group dictionaries before they reach the model"""
    out = dict(stats)
    for key in ("groups", "shares_pct"):
        values = out.get(key)
        if isinstance(values, dict) and len(values) > _MAX_GROUPS_SHOWN:
            largest = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
            out[key] = dict(largest[:_MAX_GROUPS_SHOWN])
            out[f"{key}_omitted"] = len(values) - _MAX_GROUPS_SHOWN
    return out

def _build_insight_messages(question: str, stats: dict,
                            feedback: str | None = None) -> list[dict]:
    user = f"Question: {question}\nStatistics: {json.dumps(_compact_stats(stats), ensure_ascii=False)}"
    if feedback:
        user += f"\n\n{feedback}"
    return [{"role": "system", "content": INSIGHT_SYSTEM},
            {"role": "user", "content": user}]
#################################


# Groundedness verification (mechanical, no LLM):
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(obj) -> set[float]:
    """Collect every numeric value reachable in the stats dict (recursively)"""
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(round(float(obj), 3))
        found.add(round(abs(float(obj)), 3)) # sign phrasing tolerance
    elif isinstance(obj, str):
        for m in _NUM_RE.findall(obj):
            try:
                found.add(round(float(m.replace(",", "")), 3))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found |= _numbers_in(k) | _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _numbers_in(v)
    return found


def verify_grounded(text: str, stats: dict) -> tuple[bool, list[str]]:
    """True if every number in the text exists in the stats (rounding tolerant)"""
    allowed = _numbers_in(stats)
    problems: list[str] = []
    for m in _NUM_RE.findall(text):
        try:
            val = round(float(m.replace(",", "")), 3)
        except ValueError:
            continue
        # tolerate the same value at coarser rounding (e.g. 836154.033 -> 836154)
        ok = any(abs(val - a) < 0.5 or (a != 0 and abs(val - round(a)) < 0.5)
                 for a in allowed)
        if not ok:
            problems.append(m)
    return len(problems) == 0, problems
#################################


# Deterministic templates (the guaranteed,grounded fallback):
def _template_insight(question: str, stats: dict) -> str:
    f = stats.get("focus")
    try:
        if f in ("group_stats", "share_stats"):
            base = (f"{stats['top_group']} leads with {stats['top_value']:,}; "
                    f"{stats['bottom_group']} is lowest at {stats['bottom_value']:,}.")
            if "shares_pct" in stats:
                top_share = stats["shares_pct"].get(stats["top_group"])
                base = (f"{stats['top_group']} accounts for {top_share}% of the total; "
                        f"{stats['bottom_group']} has the smallest share.")
            return base
        if f == "trend_stats":
            return (f"From {stats['first_x'][:10]} to {stats['last_x'][:10]} the value "
                    f"changed by {stats['change_pct']}%, peaking at {stats['peak_value']:,} "
                    f"({stats['peak_x'][:10]}).")
        if f == "correlation":
            c = stats.get("columns", ["x", "y"])
            return (f"{c[0]} and {c[1]} show a {stats['strength']} {stats['direction']} "
                    f"correlation (r={stats['pearson_r']}, n={stats['n']}).")
        if f == "distribution_stats":
            return (f"{stats['column']} averages {stats['mean']:,} (median {stats['median']:,}), "
                    f"ranging from {stats['min']:,} to {stats['max']:,}.")
        if f == "outlier_detection":
            if stats.get("n_outliers", 0) == 0:
                return "No outliers were detected outside the IQR bounds."
            top = next(iter(stats.get("top_outliers", {}).items()), (None, None))
            return (f"{stats['n_outliers']} outlier(s) detected beyond the IQR bound of "
                    f"{stats['iqr_high']:,}; the most extreme is {top[1]:,} on {str(top[0])[:10]}.")
    except Exception:
        pass
    return f"The analysis produced {stats.get('n_rows_result', '?')} result rows."
#################################


# Agent entry point:
def run_insight(client: ModelClient, question: str, stats: dict,
                feedback: str | None = None) -> InsightResult:
    """LLM writes from the stats -> verifier checks -> template on any violation"""
    messages = _build_insight_messages(question, stats, feedback)
    raw = client.generate(messages)

    text = None
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            text = str(json.loads(m.group(0)).get("insight", "")).strip()
        except Exception:
            text = None

    if text:
        grounded, problems = verify_grounded(text, stats)
        if grounded and 10 <= len(text) <= 400:
            return InsightResult(insight=text, supporting_stats=stats, source="llm")

    return InsightResult(insight=_template_insight(question, stats), supporting_stats=stats, source="template_fallback")
#################################