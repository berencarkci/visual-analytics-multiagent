"""Multi arm comparison: training axis x architecture axis.

Two independent experiments share one runner here, because they share one
expensive resource a T4 that can hold exactly one 3B model at a time.

  Training axis      base -> SFT-v2 -> SFT-v3, and (on the v2 branch) DPO.
                     DPO adapters were trained on top of SFT-v2, so they are
                     compared against v2, never against v3: mixing the two
                     would confound "preference optimisation" with "more SFT
                     data". The two experiments are reported separately.

  Architecture axis  single-agent (one call) vs multi-agent vs multi-agent
                     without the Evaluation reviewer. The ablation arm needs
                     orchestrator.run_workflow(use_evaluator=False).

Why one script: every arm loads a 3B model. Grouping arms by adapter means the
weights are loaded once per model instead of once per arm, which is the
difference between ~90 minutes and a session timeout.

Checkpointing: each arm writes evaluation/results/arms/<arm_id>.json the moment
it finishes, and every question is flushed to that file as it completes. A
killed session resumes with the same command — finished arms are skipped.

Measurement note (read before comparing columns):
    insight_invented_numbers is only meaningful for single call arms. A
    single call model never sees the data, so a figure it states is unsupported
    unless the prompt supplied it. Multi-agent arms are handed computed
    statistics and are supposed to quote them, so the same metric would flag
    correct behaviour. Multi-agent groundedness is therefore read from the
    Evaluation agent's insight_grounded check instead. The two are printed in
    separate rows and must not be compared to each other.

Usage (from the repo root):
    python evaluation/compare_arms.py # dev split, all arms
    python evaluation/compare_arms.py --limit 4 # smoke test first
    python evaluation/compare_arms.py --arms sft_v3_multi
    python evaluation/compare_arms.py --report # merge + print, no GPU
    python evaluation/compare_arms.py --split test # once, after freeze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "agents")

from baseline import recommend
from compare_prompt_vs_sft import free_client, score_answer
from data_ingestion import load_table, profile_table, schema_summary
from model_client import HFClient
from orchestrator import run_workflow
from prompts import SFT_SYSTEM
from schemas import validate_output

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SFT_V2 = "berencarkci/qwen2.5-3b-va-sft-v2"
SFT_V3 = "berencarkci/qwen2.5-3b-va-sft-v3"
DPO_ALL = "berencarkci/qwen2.5-3b-va-dpo"
DPO_REAL = "berencarkci/qwen2.5-3b-va-dpo-real"

DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}

RESULTS_DIR = Path("evaluation/results/arms")
REPORT_PATH = Path("evaluation/results/arm_comparison.json")
CACHE_PATH = Path("evaluation/results/comparison_cache.json")

# Arm registry.
#
# mode:
#   single_long   frozen few shot prompt + one retry (the shipped baseline)
#   single_short  short SFT prompt, one call, no retry
#   multi         full pipeline with the Evaluation reviewer
#   multi_noeval  same pipeline, reviewer off (ablation)
#
# Order matters: arms are executed grouped by adapter, in first appearance
# order, so the cheapest diagnostics land first if the session dies early.
ARMS: list[dict] = [
    {"id": "base_single", "adapter": None, "mode": "single_long",
     "label": "base + long prompt", "axis": "both"},
    {"id": "base_multi", "adapter": None, "mode": "multi",
     "label": "base multi-agent", "axis": "architecture"},

    {"id": "sft_v2_single", "adapter": SFT_V2, "mode": "single_short",
     "label": "SFT-v2 single", "axis": "training"},
    {"id": "sft_v2_multi", "adapter": SFT_V2, "mode": "multi",
     "label": "SFT-v2 multi-agent", "axis": "training"},

    {"id": "sft_v3_single", "adapter": SFT_V3, "mode": "single_short",
     "label": "SFT-v3 single", "axis": "training"},
    {"id": "sft_v3_multi", "adapter": SFT_V3, "mode": "multi",
     "label": "SFT-v3 multi-agent", "axis": "both"},
    {"id": "sft_v3_multi_noeval", "adapter": SFT_V3, "mode": "multi_noeval",
     "label": "SFT-v3 multi, no reviewer", "axis": "architecture"},

    {"id": "dpo_all_multi", "adapter": DPO_ALL, "mode": "multi",
     "label": "DPO-all multi-agent", "axis": "training"},
    {"id": "dpo_real_multi", "adapter": DPO_REAL, "mode": "multi",
     "label": "DPO-real multi-agent", "axis": "training"},
]
#################################


# Question loading:
def load_questions(split: str, limit: int | None) -> list[dict]:
    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    qs = [q for q in bench["questions"] if q.get("split") == split]
    if not qs:
        raise SystemExit(f"no {split}-split questions found — run evaluation/make_split.py first")
    return qs[:limit] if limit else qs
#################################


# One question, one arm:
def _intent_from_trace(trace: list[dict]) -> str | None:
    """The Supervisor's label, read back from the trace (no extra model call)"""
    for msg in trace:
        if msg.get("payload_type") == "IntentResult":
            return (msg.get("payload") or {}).get("intent")
    return None


def run_one(mode: str, client, q: dict, schema_text: str, frames: dict,
            profiles: dict) -> dict:
    """Run a single question through one arm; never raises"""
    row: dict = {"id": q["id"], "type": q["type"], "dataset": q["dataset"],
                 "question": q["question"]}
    df = frames[q["dataset"]]
    t0 = time.time()

    try:
        if mode == "single_long":
            res = recommend(client, schema_text, q["question"])
            rec, row["valid"], row["used_retry"] = res.recommendation, res.valid, res.used_retry

        elif mode == "single_short":
            messages = [{"role": "system", "content": SFT_SYSTEM},
                        {"role": "user", "content": f"{schema_text}\n\nQuestion: {q['question']}"}]
            rec, _ = validate_output(client.generate(messages))
            row["valid"], row["used_retry"] = rec is not None, False

        else:  # multi / multi_noeval
            wf = run_workflow(client, df, profiles[q["dataset"]], q["question"],
                              use_evaluator=(mode == "multi"))
            rec = wf.recommendation
            row["valid"] = wf.ok and rec is not None
            row["used_retry"] = bool(wf.verdict and wf.verdict.retried_step)
            row["chain_stopped"] = not wf.ok
            if not wf.ok and wf.error is not None:
                row["error"] = str(getattr(wf.error, "message", wf.error))[:200]
            row["predicted_intent"] = _intent_from_trace(wf.trace)
            row["intent_correct"] = row["predicted_intent"] == q["type"]
            if wf.verdict is not None:
                row["eval_passed"] = wf.verdict.passed
                row["retry_helped"] = bool(wf.verdict.retry_helped)
                checks = wf.verdict.checks or {}
                row["insight_grounded"] = checks.get("insight_grounded")
                row["chart_intent_fit_check"] = checks.get("chart_intent_fit")

    except Exception as exc: # one bad question must not kill an arm
        row.update(valid=False, crashed=True,
                   error=f"{type(exc).__name__}: {exc}"[:200])
        traceback.print_exc(limit=2)
        rec = None

    row["seconds"] = round(time.time() - t0, 1)

    if rec is not None:
        row.update(score_answer(rec, df, q["type"], q["question"], schema_text))
        row["chart_type"] = rec.chart_type
        row["x_axis"], row["y_axis"] = rec.x_axis, rec.y_axis
    return row
#################################


# One arm, all questions (flushed after every question):
def run_arm(arm: dict, client, questions: list[dict], schemas: dict,
            frames: dict, profiles: dict) -> dict:
    out_path = RESULTS_DIR / f"{arm['id']}.json"
    rows: list[dict] = []
    print(f"\n=== {arm['id']}  ({arm['label']}, mode={arm['mode']}) ===")

    for i, q in enumerate(questions, 1):
        row = run_one(arm["mode"], client, q, schemas[q["dataset"]], frames, profiles)
        rows.append(row)
        flag = "ok " if row.get("valid") else "FAIL"
        extra = ""
        if "intent_correct" in row:
            extra = f" intent={'ok' if row['intent_correct'] else 'X ' + str(row.get('predicted_intent'))}"
        print(f"  [{i:2}/{len(questions)}] {flag} {row['id']:<14} {row['seconds']:>5}s{extra}")
        payload = {"arm": arm, "n_done": len(rows), "n_total": len(questions), "rows": rows}
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    payload = {"arm": arm, "n_done": len(rows), "n_total": len(questions),
               "complete": True, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  -> {out_path}")
    return payload
#################################


# Aggregation:
def summarize_arm(payload: dict) -> dict:
    rows = payload["rows"]
    arm = payload["arm"]
    n = len(rows)
    ok = [r for r in rows if r.get("valid")]

    def pct(key, source, want=True):
        if not source:
            return None
        return round(100 * sum(1 for r in source if r.get(key) is want) / len(source), 1)

    s = {
        "id": arm["id"], "label": arm["label"], "mode": arm["mode"],
        "axis": arm["axis"], "n": n,
        "schema_valid_pct": round(100 * len(ok) / n, 1) if n else None,
        "columns_exist_pct": pct("columns_exist", ok),
        "chart_fits_type_pct": pct("chart_fits_type", ok),
        "used_retry_pct": pct("used_retry", rows),
        "seconds_per_question": round(sum(r.get("seconds", 0) for r in rows) / n, 1) if n else None,
    }
    if arm["mode"].startswith("single"):
        # only meaningful without data access — see the module docstring
        s["insight_invented_pct"] = pct("insight_invented_numbers", ok)
    else:
        intent_rows = [r for r in rows if "intent_correct" in r]
        s["intent_correct_pct"] = pct("intent_correct", intent_rows)
        s["chain_stopped_n"] = sum(1 for r in rows if r.get("chain_stopped"))
        graded = [r for r in rows if r.get("insight_grounded") is not None]
        s["insight_grounded_pct"] = pct("insight_grounded", graded)
        if arm["mode"] == "multi":
            s["eval_passed_pct"] = pct("eval_passed", rows)
            s["retry_helped_n"] = sum(1 for r in rows if r.get("retry_helped"))
    return s


ROWS_TO_PRINT = [
    ("schema_valid_pct", "schema valid %"),
    ("columns_exist_pct", "columns exist %"),
    ("chart_fits_type_pct", "chart fits type %"),
    ("intent_correct_pct", "intent correct %"),
    ("insight_grounded_pct", "insight grounded % (multi)"),
    ("insight_invented_pct", "invented numbers % (single)"),
    ("eval_passed_pct", "reviewer passed %"),
    ("used_retry_pct", "needed retry %"),
    ("chain_stopped_n", "chain stopped (n)"),
    ("retry_helped_n", "retry helped (n)"),
    ("seconds_per_question", "sec / question"),
]


def print_table(summaries: list[dict], title: str) -> None:
    if not summaries:
        return
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    width = max(len(lbl) for _, lbl in ROWS_TO_PRINT) + 2
    colw = 20
    print("metric".ljust(width) + "".join(s["id"][:colw - 1].rjust(colw) for s in summaries))
    print("-" * (width + colw * len(summaries)))
    for key, label in ROWS_TO_PRINT:
        if all(s.get(key) is None for s in summaries):
            continue
        line = label.ljust(width)
        for s in summaries:
            v = s.get(key)
            line += ("-" if v is None else str(v)).rjust(colw)
        print(line)
#################################


# Space cache: per question, every arm's answer side by side:
def build_cache(payloads: list[dict]) -> dict:
    cache: dict = {}
    for p in payloads:
        arm_id = p["arm"]["id"]
        for r in p["rows"]:
            entry = cache.setdefault(r["id"], {"question": r["question"],
                                               "dataset": r["dataset"],
                                               "type": r["type"], "arms": {}})
            entry["arms"][arm_id] = {
                "label": p["arm"]["label"],
                "valid": r.get("valid"),
                "chart_type": r.get("chart_type"),
                "x_axis": r.get("x_axis"),
                "y_axis": r.get("y_axis"),
                "insight": r.get("insight"),
                "predicted_intent": r.get("predicted_intent"),
                "seconds": r.get("seconds"),
            }
    return cache
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke test)")
    ap.add_argument("--arms", nargs="*", default=None, help="arm ids to run (default: all)")
    ap.add_argument("--force", action="store_true", help="rerun arms that already have results")
    ap.add_argument("--report", action="store_true", help="merge existing results only, no GPU")
    ap.add_argument("--base", default=BASE_MODEL)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = [a for a in ARMS if args.arms is None or a["id"] in args.arms]

    if not args.report:
        if args.split == "test":
            print("!! test split: this is the frozen, one-shot measurement.\n"
                  "   Prompts and adapters must not change after this runs.\n")
        questions = load_questions(args.split, args.limit)
        frames = {n: load_table(p) for n, p in DATA_FILES.items()}
        profiles = {n: profile_table(df, n) for n, df in frames.items()}
        schemas = {n: schema_summary(pr) for n, pr in profiles.items()}
        print(f"{args.split} questions: {len(questions)} | arms: {len(selected)}")

        # group by adapter so each 3B model is loaded exactly once
        by_adapter: dict = {}
        for a in selected:
            by_adapter.setdefault(a["adapter"], []).append(a)

        for adapter, arms in by_adapter.items():
            todo = [a for a in arms
                    if args.force or not (RESULTS_DIR / f"{a['id']}.json").exists()]
            if not todo:
                print(f"\n(skipping {adapter or 'base'} — all arms already done)")
                continue
            print(f"\nloading model: {adapter or 'base (no adapter)'}")
            client = HFClient(model_name=args.base, adapter=adapter, temperature=0.0)
            try:
                for arm in todo:
                    run_arm(arm, client, questions, schemas, frames, profiles)
            finally:
                free_client(client)      # a T4 cannot hold two 3B models at once

    # ---- merge whatever exists ----
    payloads = []
    for a in ARMS:
        p = RESULTS_DIR / f"{a['id']}.json"
        if p.exists():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
    if not payloads:
        print("no arm results found yet")
        return 1

    summaries = [summarize_arm(p) for p in payloads]
    by_id = {s["id"]: s for s in summaries}

    print_table([by_id[i] for i in ("base_single", "sft_v2_single", "sft_v3_single")
                 if i in by_id], "TRAINING AXIS — single-call")
    print_table([by_id[i] for i in ("base_multi", "sft_v2_multi", "sft_v3_multi")
                 if i in by_id], "TRAINING AXIS — multi-agent (SFT data expansion)")
    print_table([by_id[i] for i in ("sft_v2_multi", "dpo_all_multi", "dpo_real_multi")
                 if i in by_id],
                "TRAINING AXIS — preference optimisation (all share the SFT-v2 base)")
    print_table([by_id[i] for i in ("base_single", "sft_v3_single", "sft_v3_multi_noeval",
                                    "sft_v3_multi") if i in by_id],
                "ARCHITECTURE AXIS — single vs multi vs multi+reviewer")

    REPORT_PATH.write_text(json.dumps(
        {"split": args.split, "base_model": args.base, "summaries": summaries},
        indent=2, default=str), encoding="utf-8")
    CACHE_PATH.write_text(json.dumps(build_cache(payloads), indent=2, default=str),
                          encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}\nSpace cache -> {CACHE_PATH}")

    incomplete = [p["arm"]["id"] for p in payloads if not p.get("complete")]
    if incomplete:
        print(f"\nWARNING: incomplete arms (rerun to finish): {', '.join(incomplete)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################