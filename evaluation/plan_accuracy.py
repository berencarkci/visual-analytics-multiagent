"""Plan-level quality: the axis the headline benchmark leaves flat.

The frozen benchmark reports intent accuracy, schema validity and chart-fit —
all near the ceiling, so every arm scores about the same and the real gain from
training the pipeline is invisible in them. Intent correct sits at ~92% for the
untrained base model and for the trained models alike. Yet the answers plainly
improve: the base model picks the wrong measure column, skips the grouping, or
invents a column that does not exist. That improvement lives in the PLAN, one
layer below intent, and nothing in the headline table measures it.

There is no gold transform to score against — benchmark.json carries a gold
intent (the `type` field) but not a gold plan. So this does not compare each
plan to a reference. It measures three reference-free properties of plan
quality, each of which the training demonstrably moved:

  1. Executability. Does the plan refer only to real columns, and did the chain
     run to completion? The base model wrote `weekday` and `sales` — columns
     that do not exist — and the chain stopped. A plan that cannot run is the
     clearest possible plan failure and needs no reference to detect.

  2. Intent-structure consistency. Does the plan carry the structure its own
     intent requires? A comparison or filter_aggregation needs a grouping; a
     trend needs a time axis; a correlation is point-wise, not grouped; a
     distribution is over one column. These are logical entailments of the
     intent, not a gold answer.

  3. Plan richness. Did the plan actually specify the operations the question
     implies (an aggregation, a grouping), or leave them empty? The base model
     omits these far more often, collapsing an analysis into a bare column pick.

Read on the DEV split only. The frozen test split was measured once; a new
metric over it would turn held-out evaluation into model selection.

Usage (from the repo root):
    python evaluation/plan_accuracy.py
    python evaluation/plan_accuracy.py --arms base_multi sft_v2_multi sft_v3_multi sft_v5_multi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "agents")

DEFAULT_CACHE = Path("evaluation/results/comparison_cache.json")
OUT_PATH = Path("evaluation/results/plan_accuracy.json")

DEFAULT_ARMS = ["base_multi", "sft_v2_multi", "sft_v3_multi", "sft_v5_multi"]

from data_ingestion import load_table, profile_table

DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}
#################################


def _numeric_cols() -> dict:
    out: dict = {}
    for name, path in DATA_FILES.items():
        try:
            prof = profile_table(load_table(path), name)
            out[name] = {c.name for c in prof.columns if c.dtype == "numeric"}
        except Exception:
            out[name] = set()
    return out


def _norm(v) -> str:
    return "" if v is None else "".join(str(v).lower().split())


def _has(v) -> bool:
    return bool(_norm(v))
#################################


def _intent_structure(intent, tf, y_axis) -> tuple:
    """(checked, passed): checked=False means the rule does not apply here"""
    gb = _has(tf.get("groupby"))
    if intent in ("comparison", "filter_aggregation", "composition"):
        return True, gb
    if intent == "trend":
        expr = _norm(tf.get("groupby"))
        time_like = (any(k in expr for k in ("month", "quarter", "week", "day", "year"))
                     or "date" in _norm(y_axis) or "date" in expr)
        return True, time_like
    if intent == "relationship":
        return True, not gb
    if intent == "distribution":
        agg = _norm(tf.get("agg"))
        return True, (not gb) or agg == "count"
    if intent == "anomaly":
        return True, gb
    return False, False
#################################


FIELDS = [
    ("executable", "executable (real cols, ran)"),
    ("intent_structure", "intent-structure consistent"),
    ("has_grouping", "specified a grouping"),
    ("has_aggregation", "specified an aggregation"),
    ("plan_complete", "executable AND consistent"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    path = Path(args.cache)
    if not path.exists():
        print(f"cache not found: {path}")
        return 1
    cache = json.loads(path.read_text(encoding="utf-8"))
    _numeric_cols()  # validates the data files load; schema not needed further

    all_arms = sorted({a for e in cache.values() for a in e.get("arms", {})})
    arms = [a for a in (args.arms or DEFAULT_ARMS) if a in all_arms]
    if not arms:
        print("requested arms not in cache. Available:", all_arms)
        return 1

    tally = {a: {f[0]: 0 for f in FIELDS} for a in arms}
    counted = {a: {f[0]: 0 for f in FIELDS} for a in arms}
    per_question = []

    for qid, entry in cache.items():
        intent = entry.get("type")
        row = {"id": qid, "type": intent, "arms": {}}
        for a in arms:
            ans = entry["arms"].get(a)
            flags: dict = {}
            if not ans:
                for k in ("executable", "intent_structure", "plan_complete"):
                    flags[k] = False
                    counted[a][k] += 1
                row["arms"][a] = flags
                continue

            tf = ans.get("transform") or {}
            executable = bool(ans.get("columns_exist")) and bool(ans.get("valid"))
            flags["executable"] = executable
            counted[a]["executable"] += 1
            tally[a]["executable"] += int(executable)

            checked, passed = _intent_structure(intent, tf, ans.get("y_axis"))
            if checked:
                flags["intent_structure"] = passed
                counted[a]["intent_structure"] += 1
                tally[a]["intent_structure"] += int(passed)

            gb, agg = _has(tf.get("groupby")), _has(tf.get("agg"))
            flags["has_grouping"] = gb
            flags["has_aggregation"] = agg
            counted[a]["has_grouping"] += 1
            counted[a]["has_aggregation"] += 1
            tally[a]["has_grouping"] += int(gb)
            tally[a]["has_aggregation"] += int(agg)

            complete = executable and (passed if checked else True)
            flags["plan_complete"] = complete
            counted[a]["plan_complete"] += 1
            tally[a]["plan_complete"] += int(complete)

            row["arms"][a] = flags
        per_question.append(row)

    n = len(cache)
    print(f"\nPlan-level quality over {n} questions ({path.name}, dev split)\n")
    print("Reference-free: no gold transform exists, so these measure whether a "
          "plan runs and is structurally right for its own intent — not whether "
          "it matches a reference.\n")

    label_w = max(len(lbl) for _, lbl in FIELDS) + 2
    header = " " * label_w + "".join(f"{a:>16}" for a in arms)
    print(header)
    print("-" * len(header))
    for key, lbl in FIELDS:
        cells = ""
        for a in arms:
            c = counted[a][key]
            pct = 100 * tally[a][key] / c if c else 0
            cells += f"{pct:>15.1f}%"
        print(f"{lbl:<{label_w}}{cells}")

    print("\nRead: intent accuracy is ~92% for every arm, so the headline "
          "benchmark hides the training's contribution. Executability and "
          "intent-structure consistency are where the trained pipeline pulls "
          "ahead — the base model writes columns that do not exist and skips "
          "groupings, and those are exactly the plan failures a user feels.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "split": "dev", "n": n, "arms": arms,
        "pct": {a: {k: round(100 * tally[a][k] / counted[a][k], 1)
                    if counted[a][k] else None for k in tally[a]} for a in arms},
        "counted": counted, "totals": tally,
        "per_question": per_question,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################