"""Failure scan over the dev split (Task B4/T1, preparation).

Runs every dev-split question through the full multi-agent chain and records
where the chain had to correct the model. DPO needs real mistakes, and the
smoke test only covers 8 questions — this scan tells us which failure modes are
still alive after SFT, so the preference pairs target what the model actually
gets wrong rather than what it used to get wrong.

Three kinds of evidence are collected, in decreasing severity:

  * Evaluation issues — the reviewer rejected the answer outright.
  * Guardrail corrections — the model picked a chart the data does not support
    and the rule layer overrode it. The raw pick is a ready-made "rejected"
    sample: the model chose it, the system refused it.
  * Plan notes — the transform engine dropped part of the plan (unknown derived
    expression, invalid sort). Same idea one step earlier in the chain.

Usage (from the repo root):
    python evaluation/scan_dev_failures.py --adapter berencarkci/qwen2.5-3b-va-sft
    python evaluation/scan_dev_failures.py                    # base model
    python evaluation/scan_dev_failures.py --split test       # do NOT do this yet
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "agents")

from data_ingestion import load_table, profile_table
from model_client import HFClient
from orchestrator import run_workflow

DATA_FILES = {
    "retail_sales_superstore": "data/retail_sales_superstore.csv",
    "customer_analytics_mall": "data/customer_analytics_mall.csv",
    "energy_consumption_hourly": "data/energy_consumption_hourly.csv",
}
OUT_PATH = Path("evaluation/results/dev_failure_scan.json")
#################################


# Per-question evidence:
def scan_question(client, q: dict, frames: dict, profiles: dict) -> dict:
    """One full chain pass, reduced to the fields a preference pair would need"""
    ds = q["dataset"]
    result = run_workflow(client, frames[ds], profiles[ds], q["question"], log_dir="logs")

    row = {"id": q["id"], "type": q["type"], "dataset": ds, "question": q["question"],
           "ok": result.ok}

    if not result.ok: # chain stopped early
        row["step_error"] = (f"{result.error.error_type}: {result.error.detail}"
                             if result.error else "unknown")
        return row

    v = result.verdict
    rec = result.recommendation
    row.update({
        "passed": v.passed if v else None,
        "issues": v.issues if v else [],
        "warnings": v.warnings if v else [],
        "retried_step": v.retried_step if v else None,
        "retry_helped": v.retry_helped if v else None,
        "chart_type": rec.chart_type if rec else None,
        "x_axis": rec.x_axis if rec else None,
        "y_axis": rec.y_axis if rec else None,
        "transform": rec.transform.model_dump() if rec else None,
        "insight": result.insight,
    })

    # Guardrail corrections and plan notes live in the trace payloads:
    for entry in result.trace:
        payload = entry.get("payload", {})
        if entry.get("agent") == "visualization" and payload.get("guardrails_applied"):
            row.setdefault("guardrails", []).extend(payload["guardrails_applied"])
        if entry.get("agent") == "data_analyst" and payload.get("notes"):
            row.setdefault("plan_notes", []).extend(payload["notes"])
        if entry.get("agent") == "insight" and payload.get("source"):
            row["insight_source"] = payload["source"]
    return row
#################################


# Summary:
def summarize(rows: list[dict]) -> dict:
    issues = collections.Counter()
    for r in rows:
        for i in r.get("issues", []):
            issues[i.split(":")[0]] += 1

    guardrails = collections.Counter()
    for r in rows:
        for g in r.get("guardrails", []):
            guardrails[g.split(":")[0]] += 1

    notes = collections.Counter()
    for r in rows:
        for n in r.get("plan_notes", []):
            notes[n.split(":")[0].split("(")[0].strip()] += 1

    return {
        "n_questions": len(rows),
        "chain_stopped": sum(1 for r in rows if not r["ok"]),
        "evaluation_failed": sum(1 for r in rows if r.get("passed") is False),
        "had_warnings": sum(1 for r in rows if r.get("warnings")),
        "retried": sum(1 for r in rows if r.get("retried_step")),
        "retry_helped": sum(1 for r in rows if r.get("retry_helped")),
        "guardrail_corrected": sum(1 for r in rows if r.get("guardrails")),
        "plan_degraded": sum(1 for r in rows if r.get("plan_notes")),
        "template_fallback": sum(1 for r in rows
                                 if r.get("insight_source") == "template_fallback"),
        "issues_by_rule": dict(issues),
        "guardrails_by_kind": dict(guardrails),
        "plan_notes_by_kind": dict(notes),
    }


def print_report(rows: list[dict], summary: dict) -> None:
    print(f"\n{'='*70}\nSUMMARY  ({summary['n_questions']} questions)\n{'='*70}")
    for k in ["chain_stopped", "evaluation_failed", "had_warnings", "retried",
              "retry_helped", "guardrail_corrected", "plan_degraded", "template_fallback"]:
        print(f"  {k:22} {summary[k]}")

    for label, key in [("Evaluation issues", "issues_by_rule"),
                       ("Guardrail corrections", "guardrails_by_kind"),
                       ("Plan degradations", "plan_notes_by_kind")]:
        if summary[key]:
            print(f"\n  {label}:")
            for name, n in sorted(summary[key].items(), key=lambda kv: -kv[1]):
                print(f"    {n:3}x  {name}")

    print(f"\n{'='*70}\nPAIR CANDIDATES\n{'='*70}")
    for r in rows:
        marks = []
        if not r["ok"]:
            marks.append(f"STOPPED: {r.get('step_error', '')[:50]}")
        if r.get("passed") is False:
            marks.append(f"FAILED: {'; '.join(r.get('issues', []))[:60]}")
        if r.get("guardrails"):
            marks.append(f"GUARDRAIL: {'; '.join(r['guardrails'])[:60]}")
        if r.get("plan_notes"):
            marks.append(f"PLAN: {'; '.join(r['plan_notes'])[:60]}")
        if r.get("insight_source") == "template_fallback":
            marks.append("INSIGHT: fell back to template")
        if marks:
            print(f"\n  {r['id']} [{r['type']}] {r['question'][:58]}")
            for m in marks:
                print(f"      {m}")
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter: local path or Hub id")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.split == "test":
        print("WARNING: the test split is sealed for the final evaluation. "
              "Scanning it here leaks information into the training data.")

    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    questions = [q for q in bench["questions"] if q.get("split") == args.split]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit(f"no {args.split}-split questions found")

    frames = {name: load_table(path) for name, path in DATA_FILES.items()}
    profiles = {name: profile_table(df, name) for name, df in frames.items()}

    label = args.adapter or "base model (no adapter)"
    print(f"scanning {len(questions)} {args.split} questions | {label}")

    rows = []
    client = HFClient(model_name=args.base, adapter=args.adapter, temperature=0.0)
    for i, q in enumerate(questions, 1):
        print(f"  [{i:2}/{len(questions)}] {q['id']}", end="\r")
        rows.append(scan_question(client, q, frames, profiles))

    summary = summarize(rows)
    print_report(rows, summary)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"adapter": args.adapter, "split": args.split,
                                    "summary": summary, "rows": rows},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################