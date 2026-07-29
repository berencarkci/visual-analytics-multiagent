"""Capability probe for the schema expansion (Task B4/T2, follow-up).

The smoke test and the dev split predate the expansion, so neither can tell
whether the model actually learned to use derived measures and the two new sort
directions — they contain no question that needs them. This probe asks
questions that do.

Each case states what the model has to produce and why. A case passes only if
the plan is executable AND uses the mechanism the question calls for: answering
"which sub-categories lose money" with a descending sort is not wrong exactly,
but it is not what was taught either, so it counts as a miss.

None of these questions appear in the benchmark or the training set; they are
phrased from scratch here.

Usage (from the repo root):
    python evaluation/probe_new_capabilities.py --adapter berencarkci/qwen2.5-3b-va-sft-v2
    python evaluation/probe_new_capabilities.py                # base model, for contrast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "agents")

from data_analyst import _build_plan_messages
from data_ingestion import load_table, profile_table, schema_summary
from model_client import HFClient
from schemas import Transform
from transforms import apply_transform, measure_base_columns

DATA_FILES = {
    "retail": "data/retail_sales_superstore.csv",
    "energy": "data/energy_consumption_hourly.csv",
}
OUT_PATH = Path("evaluation/results/capability_probe.json")

# (dataset, intent, question, what the answer must contain, why)
CASES = [
    ("retail", "comparison",
     "Which shipping mode keeps customers waiting longest between order and dispatch?",
     {"measure": "days_between"},
     "duration between two date columns is not a column; it has to be derived"),

    ("retail", "trend",
     "Has the gap between ordering and shipping widened over the quarters?",
     {"measure": "days_between", "groupby": "quarter"},
     "same derived duration, this time over a time axis"),

    ("retail", "comparison",
     "Which sub-categories keep the least profit out of every sales dollar?",
     {"measure": "ratio"},
     "margin is profit over sales, a ratio of two columns"),

    ("retail", "filter_aggregation",
     "Show me the ten states at the bottom of the profit ranking.",
     {"sort": "value_asc"},
     "the bottom of a ranking needs ascending order, which only exists since the expansion"),

    ("retail", "filter_aggregation",
     "Which cities lose us the most money overall?",
     {"sort": "value_asc"},
     "losing money means the most negative total, so ascending again"),

    ("retail", "trend",
     "Give me the last six months of revenue, newest first.",
     {"sort": "date_desc"},
     "newest first is a descending time order"),

    ("energy", "comparison",
     "How much more electricity do appliances draw than the lights, per weekday?",
     {"measure": "diff"},
     "a difference between two measured columns"),

    ("retail", "comparison",
     "Which product categories earn the most per item sold?",
     {"measure": "ratio"},
     "revenue per unit is sales over quantity"),
]
#################################


def _parse(raw: str) -> dict | None:
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def check(plan: dict, expect: dict, df) -> tuple[bool, str]:
    """Executable, and using the mechanism the question calls for"""
    tf = plan.get("transform") or {}
    cols = plan.get("target_columns") or []

    if "measure" in expect:
        derived = [c for c in cols if "(" in str(c)]
        if not derived:
            return False, "no derived measure in target_columns"
        if not any(str(c).startswith(expect["measure"]) for c in derived):
            return False, f"derived measure is {derived}, expected {expect['measure']}(...)"

    if "sort" in expect and tf.get("sort") != expect["sort"]:
        return False, f"sort is {tf.get('sort')}, expected {expect['sort']}"

    if "groupby" in expect and expect["groupby"] not in str(tf.get("groupby") or ""):
        return False, f"groupby is {tf.get('groupby')}, expected {expect['groupby']}(...)"

    # executability: the plan has to survive the real engine
    try:
        from schemas import ChartRecommendation
        x = cols[0] if cols else None
        y = cols[1] if len(cols) > 1 else None
        rec = ChartRecommendation(chart_type="bar", x_axis=x, y_axis=y,
                                  transform=Transform(**tf), reason="probe", insight="probe")
        out, _, _, _series, notes = apply_transform(df, rec)
        if out is None or out.empty:
            return False, "plan executed to an empty result"
        if any("skipped" in n for n in notes):
            return False, f"engine dropped part of the plan: {notes}"
    except Exception as e:
        return False, f"not executable: {type(e).__name__}: {e}"

    return True, "ok"
#################################


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    frames = {k: load_table(v) for k, v in DATA_FILES.items()}
    schemas = {k: schema_summary(profile_table(df, k)) for k, df in frames.items()}
    client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)

    print(f"capability probe | {args.adapter or 'base model'}\n" + "=" * 72)
    rows, passed = [], 0
    for ds, intent, question, expect, why in CASES:
        messages = _build_plan_messages(schemas[ds], question, intent)
        plan = _parse(client.generate(messages))
        if plan is None:
            ok, detail = False, "unparseable output"
        else:
            ok, detail = check(plan, expect, frames[ds])
        passed += ok

        print(f"\n{'PASS' if ok else 'FAIL'}  [{intent}] {question}")
        print(f"      needs: {expect}   ({why})")
        if plan:
            tf = plan.get("transform") or {}
            print(f"      got  : cols={plan.get('target_columns')} "
                  f"gb={tf.get('groupby')} agg={tf.get('agg')} sort={tf.get('sort')} "
                  f"limit={tf.get('limit')}")
        if not ok:
            print(f"      why  : {detail}")
        rows.append({"question": question, "intent": intent, "expect": expect,
                     "plan": plan, "passed": ok, "detail": detail})

    print(f"\n{'=' * 72}\n{passed}/{len(CASES)} capability cases passed")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"adapter": args.adapter, "passed": passed,
                                    "total": len(CASES), "rows": rows},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################