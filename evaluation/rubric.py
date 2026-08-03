"""Scoring rubric for preference pair construction.

Two candidate answers to the same question need a defensible ordering before
they can become a (chosen, rejected) pair. This module scores one candidate
against a reference answer across six dimensions and returns a total, so pairs
can be built mechanically and only the close calls reach a human.

Why mechanical scoring at all: the preference labels matter more than the
strength of the models that produced the candidates, so the labelling has to be
consistent. A rule that says "the column is either in the schema or it is not"
never drifts; a human reading 430 JSON blobs does.

The six dimensions, and which formats they apply to:

    dimension              supervisor  data_analyst  visualization  insight  single_call
    schema_validity            -            x              -            -         x
    column_selection           -            x              -            -         x
    transform_correctness      -            x              -            -         x
    chart_appropriateness      -            -              x            -         x
    groundedness               -            -              -            x         x
    clarity                    -            -              x            x         x
    intent_correctness         x            -              -            -         -

Each dimension scores 0 / 1 / 2 (wrong / acceptable / correct). A candidate's
total is the sum over the dimensions that apply to its format, normalised to
0-100 so formats with different dimension counts stay comparable.

Some dimensions are hard gates: a chart type outside the intent's allowed list,
or a column that does not exist, scores 0 regardless of everything else, because
the system would reject that answer anyway.

Usage:
    from rubric import score_candidate, compare
    s = score_candidate(cand, reference, fmt="data_analyst", df=df, intent="trend")
    verdict = compare(score_a, score_b)     # "a" | "b" | "tie" | "unclear"
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, "agents")

import pandas as pd

from visualization import ALLOWED_CHARTS

# A pair is auto-labelled only when the candidates differ by MORE than this many
# raw points. A one-point gap means they differ in a single dimension by one
# grade — a real but weak difference, which makes a noisy training signal, so
# those pairs go to manual review instead.
UNCLEAR_MARGIN = 1             # raw points

DIMENSIONS_BY_FORMAT = {
    "supervisor":    ["intent_correctness"],
    "data_analyst":  ["schema_validity", "column_selection", "transform_correctness"],
    "visualization": ["chart_appropriateness", "clarity"],
    "insight":       ["groundedness", "clarity"],
    "single_call":   ["schema_validity", "column_selection", "transform_correctness",
                      "chart_appropriateness", "groundedness", "clarity"],
}

_DERIVED_RE = re.compile(r"^(\w+)\((.+)\)$")
_GENERIC_PHRASES = ("result rows", "the analysis produced", "the chart shows the data", "see the chart", "as shown")
#################################


# Column helpers:
def _base_columns(expr: str | None) -> list[str]:
    """month(order_date) -> [order_date]; ratio(profit, sales) -> [profit, sales]"""
    if not expr:
        return []
    m = _DERIVED_RE.match(expr.strip())
    if not m:
        return [expr.strip()]
    return [p.strip() for p in m.group(2).split(",")]


def _base_column(expr: str | None) -> str | None:
    """First base column of an expression (single column derivations)"""
    cols = _base_columns(expr)
    return cols[0] if cols else None


def _column_exists(col: str | None, df: pd.DataFrame) -> bool:
    return col is None or all(c in df.columns for c in _base_columns(col))
#################################


# Individual dimensions (each returns 0, 1 or 2):
_VALID_AGGS = (None, "sum", "mean", "count", "count_distinct")
_VALID_SORTS = (None, "date_asc", "date_desc", "value_asc", "value_desc")


def _score_schema_validity(cand: dict, df: pd.DataFrame) -> int:
    """Hard gate: would the system accept this answer at all?

    Covers three ways an answer can be rejected before it ever runs: a column
    that is not in the table, and a value in the agg or sort slot that the
    Transform schema does not allow. The last one is not hypothetical — the
    capability probe caught the model writing agg="ratio(profit, sales)", which
    pydantic refuses, so the whole chain would stop there.
    """
    cols = [cand.get("x_axis"), cand.get("y_axis")]
    cols += cand.get("target_columns") or []
    tf = cand.get("transform") or {}
    cols += [tf.get("groupby")]
    if not all(_column_exists(c, df) for c in cols if c):
        return 0
    if tf.get("agg") not in _VALID_AGGS or tf.get("sort") not in _VALID_SORTS:
        return 0
    return 2


def _score_column_selection(cand: dict, ref: dict, df: pd.DataFrame) -> int:
    """Did the answer pick the columns the question is actually about?"""
    def cols_of(d: dict) -> set[str]:
        out: set[str] = set()
        for c in (d.get("x_axis"), d.get("y_axis")):
            out |= set(_base_columns(c))
        for c in (d.get("target_columns") or []):
            out |= set(_base_columns(c))
        return {c for c in out if c}

    got, want = cols_of(cand), cols_of(ref)
    if not want:
        return 1 # no reference to judge against
    if got == want:
        return 2
    if got - want:
        return 0 # brought a column the answer does not need: this is how order_id ends up in a correlation
    return 1 if got else 0 # subset of the right columns: incomplete, not wrong


def _score_transform_correctness(cand: dict, ref: dict) -> int:
    """groupby and agg carry the meaning, sort and limit are presentation

    A wrong groupby changes what the chart is about (daily vs monthly, category vs state), so it is weighted as a gate. Sort and limit shift only how the same numbers are shown.
    """
    c = cand.get("transform") or {}
    r = ref.get("transform")
    if r is None:
        return 1 # reference-free scoring (live labeling): nothing to compare against, stay neutral
    def norm(v):
        return None if v in ("", None) else str(v).strip().lower()

    core_ok = norm(c.get("groupby")) == norm(r.get("groupby")) and \
              norm(c.get("agg")) == norm(r.get("agg"))
    filter_ok = norm(c.get("filter")) == norm(r.get("filter"))
    detail_ok = norm(c.get("sort")) == norm(r.get("sort")) and \
                norm(c.get("limit")) == norm(r.get("limit"))

    # A reversed sort is not a presentation detail. "Which cities lose the most
    # money" answered with value_desc shows the best performers instead of the
    # worst: the ranking is right and the end shown is wrong, which is a wrong
    # answer to the question asked. A missing sort stays a minor flaw.
    opposites = {("value_asc", "value_desc"), ("value_desc", "value_asc"),
                 ("date_asc", "date_desc"), ("date_desc", "date_asc")}
    sort_reversed = (norm(c.get("sort")), norm(r.get("sort"))) in opposites

    if not core_ok:
        return 0
    if not filter_ok:
        return 0  # a dropped filter answers another question
    if sort_reversed:
        return 0
    return 2 if detail_ok else 1


def _score_chart_appropriateness(cand: dict, ref: dict, intent: str | None) -> int:
    """Hard gate on the allowed list, then reference match"""
    chart = cand.get("chart_type")
    if not chart:
        return 0
    if intent and chart not in ALLOWED_CHARTS.get(intent, ()):
        return 0                                    # the guardrails would override this
    ref_charts = ref.get("chart_family") or ([ref["chart_type"]] if ref.get("chart_type") else [])
    if not ref_charts:
        return 1
    return 2 if chart in ref_charts else 1          # allowed but not the reference pick


def _score_groundedness(cand: dict, stats: dict | None) -> int:
    """Every number in the insight must exist in the computed statistics"""
    text = cand.get("insight")
    if not text:
        return 0
    if stats is None:
        return 1                                    # nothing to verify against
    from insight import verify_grounded
    ok, _ = verify_grounded(text, stats)
    return 2 if ok else 0


def _score_clarity(cand: dict, ref: dict) -> int:
    """Is the prose specific, or a placeholder that says nothing?

    Deliberately conservative: clarity is the one dimension a rule cannot really
    judge, so it only punishes what is clearly bad — empty text, the generic
    fallback template, or a reason naming a chart other than the one chosen.
    """
    text = (cand.get("insight") or cand.get("reason") or "").strip()
    if not text:
        return 0
    low = text.lower()
    if any(p in low for p in _GENERIC_PHRASES):
        return 0
    if len(text.split()) < 4:
        return 0
    chart = cand.get("chart_type")
    if chart:                                       # reason must not describe another chart
        named = [c for c in ("bar", "line", "scatter", "pie", "histogram", "box")
                 if re.search(rf"\b{c}\b", low)]
        if named and chart not in named:
            return 1
    return 2


def _score_intent_correctness(cand: dict, ref: dict) -> int:
    got, want = cand.get("intent"), ref.get("intent")
    if not got:
        return 0
    return 2 if got == want else 0
#################################


# Public API:
def score_candidate(cand: dict, ref: dict, fmt: str, df: pd.DataFrame | None = None,
                    intent: str | None = None, stats: dict | None = None) -> dict:
    """Score one candidate; returns per-dimension scores and a 0-100 total"""
    dims = DIMENSIONS_BY_FORMAT[fmt]
    scores: dict[str, int] = {}

    for dim in dims:
        if dim == "schema_validity":
            scores[dim] = _score_schema_validity(cand, df) if df is not None else 1
        elif dim == "column_selection":
            scores[dim] = _score_column_selection(cand, ref, df) if df is not None else 1
        elif dim == "transform_correctness":
            scores[dim] = _score_transform_correctness(cand, ref)
        elif dim == "chart_appropriateness":
            scores[dim] = _score_chart_appropriateness(cand, ref, intent)
        elif dim == "groundedness":
            scores[dim] = _score_groundedness(cand, stats)
        elif dim == "clarity":
            scores[dim] = _score_clarity(cand, ref)
        elif dim == "intent_correctness":
            scores[dim] = _score_intent_correctness(cand, ref)

    raw = sum(scores.values())
    return {"format": fmt, "dimensions": scores, "raw": raw,
            "total": round(100 * raw / (2 * len(dims)), 1)}


def compare(score_a: dict, score_b: dict, margin: int = UNCLEAR_MARGIN) -> str:
    """Order two scored candidates: 'a', 'b', 'tie' or 'unclear'

    Compared on the RAW point difference, not the normalised total: formats have
    different dimension counts, so the same one-point gap is 16.7 normalised
    points in a three-dimension format and 25 in a two-dimension one. Raw points
    keep "one dimension apart" meaning the same thing everywhere.
    """
    diff = score_a["raw"] - score_b["raw"]
    if diff == 0:
        return "tie"
    if abs(diff) <= margin:
        return "unclear"                            # one dimension apart: weak DPO signal
    return "a" if diff > 0 else "b"
#################################