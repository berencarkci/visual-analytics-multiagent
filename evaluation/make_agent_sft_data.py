"""Agent-format SFT data generator (Task B4/T1).

The single-call generator (make_sft_data.py) teaches one skill: schema +
question -> full chart recommendation. That is what baseline.py asks for, and
training on it turned out to transfer well to the Data Analyst — but it also
broke the Insight agent, which started emitting JSON dumps instead of
sentences (observed in the 3B smoke test).

This script fixes the mismatch by decomposing every single-call example into
the formats the agents actually use at inference time:

    Supervisor     question                       -> {"intent"}
    Data Analyst   schema + intent + question   -> {"target_columns", "transform"}
    Visualization  question + intent + data facts -> {"chart_type", "reason"}
    Insight        question + computed statistics -> {"insight"}

Two design points worth knowing:

  * The prompts are not re-implemented here. This module imports each agent's
    own `_build_*_messages()` function, so a training example is byte-identical
    to what the agent will send at inference. When a prompt changes, the
    training data follows automatically — no silent drift.

  * Insight targets DO contain numbers, unlike the single-call ones. The rule
    was never "avoid numbers", it was "never state a number you cannot see".
    The Insight agent is handed computed statistics, so quoting them is the
    correct behaviour; the transforms are executed here against the real data
    so the target sentences carry real values.

Supervisor exclusion was REVERSED in v3. It was originally left out because it
classified all 16 smoke-test questions correctly — but the dev split later
showed a systematic failure the smoke test never probed: filters named as noun
modifiers ("of the X category", "for Y customers", "at night") are misread as
comparison/trend, and softly-worded anomaly questions slide to trend. DPO on
supervisor pairs did not fix this (preference pairs cannot build a behaviour
SFT never established), so the fix moves here: every source example now also
yields a Supervisor record (question -> intent, known by construction), plus a
small intent-only bank in make_sft_data.py for lure patterns without a clean
executable chart target.

Usage (from the repo root):
    python evaluation/make_agent_sft_data.py
    python evaluation/make_agent_sft_data.py --per-format 250
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "agents")

import pandas as pd

from check_contamination import check_contamination
from data_analyst import _build_plan_messages, _compute_stats
from data_ingestion import load_table, profile_table, schema_summary
from insight import _build_insight_messages
from make_sft_data import (DATASETS, build_handwritten_examples,
                           build_intent_only_examples,
                           build_template_examples, pretty)
from failure_examples import FAILURE_EXAMPLES
from messages import TransformPlan
from schemas import ChartRecommendation
from supervisor import _INSIGHT_FOCUS, _build_intent_messages
from transforms import apply_transform
from visualization import ALLOWED_CHARTS, _build_viz_messages, _data_facts

random.seed(42)

OUT_PATH = Path("data/sft_agents_train.jsonl")
#################################


# Insight targets: real numbers, varied phrasing.
#
# Three variants per focus. One variant would teach the model to reproduce a
# fixed string, which is what the deterministic template fallback already does
# for free; the point of the LLM call is a sentence that reads naturally while
# staying pinned to the computed values.
def _insight_targets(stats: dict, question: str) -> list[str]:
    f = stats.get("focus")
    g = stats.get

    if f in ("group_stats", "share_stats") and g("top_group") is not None:
        top, tv = g("top_group"), g("top_value")
        bot, bv = g("bottom_group"), g("bottom_value")
        out = [f"{top} leads with {tv}, while {bot} sits lowest at {bv}.",
               f"The highest value belongs to {top} ({tv}); the lowest is {bot} ({bv}).",
               f"Across {stats['n_rows_result']} groups, {top} tops the ranking at {tv}."]
        shares = g("shares_pct")
        if shares:
            pct = shares.get(str(top))
            out = [f"{top} accounts for {pct}% of the total ({tv} of {g('total')}).",
                   f"The largest share goes to {top} at {pct}%, with {bot} the smallest.",
                   f"{top} makes up {pct}% of the whole; the total across groups is {g('total')}."]
        return out

    if f == "trend_stats" and g("first_value") is not None:
        fx, fv, lx, lv = g("first_x"), g("first_value"), g("last_x"), g("last_value")
        ch, px, pv = g("change_pct"), g("peak_x"), g("peak_value")
        direction = "rose" if (ch or 0) > 0 else "fell"
        return [f"The series {direction} from {fv} at {fx} to {lv} at {lx}, a change of {ch}%.",
                f"Between {fx} and {lx} the value moved from {fv} to {lv} ({ch}%), peaking at {pv}.",
                f"Starting at {fv} and ending at {lv}, the trend shows a {ch}% change with a high of {pv} at {px}."]

    if f == "correlation" and g("pearson_r") is not None:
        r, n = g("pearson_r"), g("n")
        cols = g("columns") or ["the two variables"]
        a = pretty(cols[0]) if cols else "x"
        b = pretty(cols[1]) if len(cols) > 1 else "y"
        strength, direction = g("strength"), g("direction")
        return [f"{a} and {b} show a {strength} {direction} correlation (r={r}, n={n}).",
                f"The relationship between {a} and {b} is {strength} and {direction}, with r={r} over {n} rows.",
                f"With r={r} across {n} observations, {a} and {b} move together only {strength}ly."]

    if f == "distribution_stats" and g("median") is not None:
        col, med, mean = pretty(g("column")), g("median"), g("mean")
        q1, q3, lo, hi = g("q1"), g("q3"), g("min"), g("max")
        return [f"{col} centres on a median of {med} (mean {mean}), with the middle half between {q1} and {q3}.",
                f"Values run from {lo} to {hi}; half of them fall between {q1} and {q3}, around a median of {med}.",
                f"The distribution of {col} has a median of {med} and a mean of {mean}, spanning {lo} to {hi}."]

    if f == "outlier_detection" and g("n_outliers") is not None:
        n_out, hi, lo = g("n_outliers"), g("iqr_high"), g("iqr_low")
        tops = g("top_outliers") or {}
        if n_out and tops:
            first = list(tops)[0]
            return [f"{n_out} periods fall outside the usual range ({lo} to {hi}); the most extreme is {first} at {tops[first]}.",
                    f"The IQR bounds are {lo} and {hi}; {n_out} values break them, led by {first} ({tops[first]}).",
                    f"{first} stands out at {tops[first]}, one of {n_out} readings beyond the {lo}–{hi} range."]
        return [f"No values fall outside the usual range ({lo} to {hi}).",
                f"Every reading stays within the IQR bounds of {lo} and {hi}.",
                f"The data shows no outliers; all values sit between {lo} and {hi}."]

    return []
#################################


# Running one example through the real transform machinery:
def _prepare(example: dict, frames: dict) -> tuple | None:
    """Execute the target transform on the real data; None if it produces nothing"""
    t = example["target"]
    ds_key = example["dataset"]
    df = frames[ds_key]
    try:
        rec = ChartRecommendation(**t)
        prepared, x_col, y_col, notes = apply_transform(df, rec)
    except Exception:
        return None
    if prepared is None or prepared.empty:
        return None

    target_columns = [c for c in (t["x_axis"], t["y_axis"]) if c]
    plan = TransformPlan(transform=rec.transform, target_columns=target_columns,
                         result_rows=int(len(prepared)), notes=notes)
    return df, prepared, x_col, y_col, plan, target_columns
#################################


# Per-format example builders:
def build_agent_examples(examples: list[dict], frames: dict,
                         schemas: dict) -> dict[str, list[dict]]:
    """Decompose single-call examples into per-agent training records"""
    out: dict[str, list[dict]] = {"supervisor": [], "data_analyst": [],
                                  "visualization": [], "insight": []}
    skipped = 0

    for ex in examples:
        intent = ex.get("intent")
        if intent is None:
            skipped += 1
            continue
        meta_base = {"source": ex.get("source", "template"), "dataset": ex["dataset"],
                     "intent": intent}

        # --- Supervisor (question -> intent; needs no executed transform, so it
        # is built before _prepare and survives transform skips) ---
        out["supervisor"].append({
            "messages": _build_intent_messages(ex["question"])
                        + [{"role": "assistant", "content": json.dumps({"intent": intent})}],
            "meta": dict(meta_base, format="supervisor")})

        prep = _prepare(ex, frames)
        if prep is None:
            skipped += 1
            continue
        raw_df, prepared, x_col, y_col, plan, target_columns = prep
        question, t = ex["question"], ex["target"]

        # --- Data Analyst ---
        out["data_analyst"].append({
            "messages": _build_plan_messages(schemas[ex["dataset"]], question, intent)
                        + [{"role": "assistant", "content": json.dumps(
                            {"target_columns": target_columns, "transform": t["transform"]})}],
            "meta": dict(meta_base, format="data_analyst")})

        # --- Visualization (same summary string the agent builds at runtime) ---
        facts = _data_facts(raw_df, prepared, plan)
        summary = (f"{facts['n_rows']} rows; x={facts['x']} ({facts['n_categories']} categories), "
                   f"y={facts['y']}; negatives={facts['has_negative']}")
        allowed = ALLOWED_CHARTS[intent]
        if t["chart_type"] in allowed:          # never teach a chart the guardrails reject
            out["visualization"].append({
                "messages": _build_viz_messages(question, intent, summary, allowed)
                            + [{"role": "assistant", "content": json.dumps(
                                {"chart_type": t["chart_type"], "reason": t["reason"]})}],
                "meta": dict(meta_base, format="visualization")})

        # --- Insight (real computed statistics -> grounded sentence) ---
        focus = _INSIGHT_FOCUS[intent]
        if intent == "distribution" and t["transform"].get("groupby"):
            # categorical distribution (v4): the target is per-category counts,
            # so the insight must describe groups, not the median of the counts.
            # Mirrors the runtime plan guardrail in data_analyst.
            focus = "group_stats"
        stats = _compute_stats(focus, raw_df, prepared, x_col, y_col, plan)
        variants = _insight_targets(stats, question)
        if variants and "stats_error" not in stats:
            out["insight"].append({
                "messages": _build_insight_messages(question, stats)
                            + [{"role": "assistant", "content": json.dumps(
                                {"insight": random.choice(variants)})}],
                "meta": dict(meta_base, format="insight")})

    if skipped:
        print(f"  (skipped {skipped} examples whose transform produced no usable data)")
    return out
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-format", type=int, default=None,
                    help="cap the number of examples kept per agent format")
    ap.add_argument("--single-call", default="data/sft_train.jsonl",
                    help="single-call set to merge in (pass '' to skip)")
    args = ap.parse_args()

    examples = build_template_examples() + build_handwritten_examples()
    examples += [dict(e, source="failure_targeted") for e in FAILURE_EXAMPLES]

    seen, unique = set(), []
    for e in examples:
        key = (e["dataset"], e["question"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    examples = unique
    print(f"source examples: {len(examples)}")

    frames = {k: load_table(v["path"]) for k, v in DATASETS.items()}
    schemas = {k: schema_summary(profile_table(df, k)) for k, df in frames.items()}

    by_format = build_agent_examples(examples, frames, schemas)

    # Intent-only bank (v3): supervisor-format records for lure patterns with no
    # clean executable chart target. These questions are NOT in the single-call
    # set, so they get their own contamination check here.
    intent_only = build_intent_only_examples()
    hits = check_contamination([e["question"] for e in intent_only])
    if hits:
        print(f"ABORT: {len(hits)} intent-only overlap(s) with the benchmark:")
        for h in hits:
            print(f"  [{h['kind']} {h['score']}] {h['benchmark_id']}  <->  {h['sft_question'][:70]}")
        return 1
    for e in intent_only:
        by_format["supervisor"].append({
            "messages": _build_intent_messages(e["question"])
                        + [{"role": "assistant", "content": json.dumps({"intent": e["intent"]})}],
            "meta": {"source": "intent_only", "dataset": e["dataset"],
                     "intent": e["intent"], "format": "supervisor"}})

    records: list[dict] = []
    for fmt, items in by_format.items():
        random.shuffle(items)
        if args.per_format:
            items = items[:args.per_format]
        records.extend(items)
        print(f"  {fmt:14} {len(items):4}")

    if args.single_call:
        path = Path(args.single_call)
        if not path.exists():
            raise SystemExit(f"single-call set not found: {path} — run make_sft_data.py first")
        n_single = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["meta"] = dict(rec.get("meta", {}), format="single_call")
                records.append(rec)
                n_single += 1
        print(f"  {'single_call':14} {n_single:4}")

    random.shuffle(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(records)} examples -> {OUT_PATH}")
    print("contamination: single-call questions are checked by make_sft_data.py; "
          "intent-only questions are checked above in this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################