"""DPO preference pair generation (Task B4/T1).

Builds (prompt, chosen, rejected) triples for the three agent formats the
failure catalogue actually points at: Supervisor, Data Analyst, Visualization.

Where the two sides come from
-----------------------------
The chosen side is easy: the SFT set already carries a verified target for
every question, and those targets were validated through the real pydantic
schema.

The rejected side is the hard part, and the dev-split scan explains why. After
SFT the model fails on 3 of 38 dev questions — nowhere near enough real
mistakes to build a few hundred pairs from. So the negatives come from three
sources, in descending order of control and ascending order of realism:

  synthetic (~60%)  the verified target, deliberately corrupted with one of the
                    failure modes observed in this project. Full control over
                    which error each pair teaches, no model call needed.
  base (~30%)       the untrained model's own answer. A real model mistake, but
                    from a different policy than the one being trained.
  sft_temp (~10%)   the SFT model sampled at temperature. On-policy in the strict
                    sense; the least controllable and the least diverse, since a
                    trained model mostly repeats itself even when sampled.

Every candidate is scored with the rubric (evaluation/rubric.py) and the pair is
only kept when the ordering is unambiguous. Ties are dropped; one-point gaps go
to a separate file for manual review, because a weak ordering makes a noisy
training signal.

Usage (from the repo root):
    python evaluation/make_dpo_pairs.py --source synthetic          # no GPU needed
    python evaluation/make_dpo_pairs.py --source base --adapter ""  # base model
    python evaluation/make_dpo_pairs.py --source sft_temp --adapter <hub-id>
    python evaluation/make_dpo_pairs.py --merge                     # combine + report
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "agents")
sys.path.insert(0, "evaluation")

import pandas as pd

from data_analyst import _build_plan_messages
from data_ingestion import load_table, profile_table, schema_summary
from make_sft_data import (DATASETS, build_handwritten_examples,
                           build_template_examples)
from failure_examples import FAILURE_EXAMPLES
from model_client import HFClient
from rubric import compare, score_candidate
from schemas import ChartRecommendation
from supervisor import _build_intent_messages
from transforms import apply_transform
from visualization import ALLOWED_CHARTS, _build_viz_messages, _data_facts
from messages import TransformPlan

random.seed(42)

OUT_DIR = Path("data/dpo")
INTENTS = ["trend", "comparison", "composition", "relationship",
           "distribution", "filter_aggregation", "anomaly"]
#################################


# Synthetic corruption: each function reproduces a failure mode seen in this
# project, so a pair teaches a specific lesson rather than a generic "be better".
def _base_of(expr: str) -> list[str]:
    """ratio(profit, sales) -> ["profit", "sales"]; plain columns pass through"""
    m = re.match(r"^\w+\((.+)\)$", str(expr).strip())
    return [p.strip() for p in m.group(1).split(",")] if m else [str(expr).strip()]


def _corrupt_wrong_column(target: dict, df: pd.DataFrame, **_) -> dict | None:
    """Swap a target column for an unrelated one (the order_id-in-correlation bug)"""
    out = copy.deepcopy(target)
    used = {out.get("x_axis"), out.get("y_axis")}
    others = [c for c in df.columns if c not in used]
    if not others:
        return None
    out["x_axis"] = random.choice(others)
    return out


def _corrupt_drop_filter(target: dict, **_) -> dict | None:
    """Drop the filter but keep everything else (the silently widened question)"""
    if not target["transform"].get("filter"):
        return None
    out = copy.deepcopy(target)
    out["transform"]["filter"] = None
    return out


def _corrupt_granularity(target: dict, **_) -> dict | None:
    """Change the time granularity: month -> day, day -> month, ..."""
    gb = target["transform"].get("groupby") or ""
    swaps = {"month": "day", "day": "month", "week": "quarter",
             "quarter": "week", "year": "month", "hour_of_day": "day_of_week"}
    for src, dst in swaps.items():
        if gb.startswith(f"{src}("):
            out = copy.deepcopy(target)
            out["transform"]["groupby"] = gb.replace(f"{src}(", f"{dst}(", 1)
            return out
    return None


def _corrupt_agg(target: dict, **_) -> dict | None:
    """sum <-> count, mean <-> sum: the same groups, the wrong quantity"""
    agg = target["transform"].get("agg")
    swaps = {"sum": "count", "mean": "sum", "count": "sum",
             "count_distinct": "count"}
    if agg not in swaps:
        return None
    out = copy.deepcopy(target)
    out["transform"]["agg"] = swaps[agg]
    return out


def _corrupt_camel_case(target: dict, **_) -> dict | None:
    """day_of_week(date) -> dayOfWeek(date): the vocabulary mismatch seen live"""
    gb = target["transform"].get("groupby") or ""
    swaps = {"day_of_week": "dayOfWeek", "hour_of_day": "hourOfDay",
             "weekend_flag": "weekendFlag"}
    for src, dst in swaps.items():
        if gb.startswith(f"{src}("):
            out = copy.deepcopy(target)
            out["transform"]["groupby"] = gb.replace(src, dst, 1)
            return out
    return None


def _corrupt_drop_groupby(target: dict, **_) -> dict | None:
    """Remove the aggregation entirely: raw rows where groups were asked for"""
    if not target["transform"].get("groupby"):
        return None
    out = copy.deepcopy(target)
    out["transform"]["groupby"] = None
    out["transform"]["agg"] = None
    return out


def _corrupt_agg_slot(target: dict, **_) -> dict | None:
    """Put the derived measure in the agg slot instead of target_columns

    Observed in the capability probe: asked for profit margin, the model wrote
    agg="ratio(profit, sales)". The Transform schema only accepts the four
    aggregation names, so pydantic rejects the plan and the chain stops.
    """
    y = target.get("y_axis") or ""
    if "(" not in str(y):
        return None
    out = copy.deepcopy(target)
    out["y_axis"] = _base_of(y)[0]
    out["transform"]["agg"] = y                     # the expression lands in the wrong slot
    return out


def _corrupt_sort_direction(target: dict, **_) -> dict | None:
    """Flip the sort direction: right ranking, wrong end of it

    Observed in the probe: "which cities lose us the most money" came back with
    value_desc, which lists the best performers when the question asked for the
    worst.
    """
    flip = {"value_asc": "value_desc", "value_desc": "value_asc",
            "date_asc": "date_desc", "date_desc": "date_asc"}
    sort = target["transform"].get("sort")
    if sort not in flip:
        return None
    out = copy.deepcopy(target)
    out["transform"]["sort"] = flip[sort]
    return out


def _corrupt_split_derived(target: dict, **_) -> dict | None:
    """Drop the derived measure and list its base columns separately

    Observed in the probe: asked how much more appliances draw than lights, the
    model listed both columns instead of diff(appliances, lights), so nothing
    computes the comparison the question is about.
    """
    y = target.get("y_axis") or ""
    if "(" not in str(y):
        return None
    bases = _base_of(y)
    if len(bases) < 2:
        return None
    out = copy.deepcopy(target)
    out["y_axis"] = bases[0]
    out["extra_columns"] = bases[1:]                # carried into target_columns below
    return out


def _corrupt_chart(target: dict, intent: str, **_) -> dict | None:
    """Pick a chart outside the intent's allowed list"""
    allowed = ALLOWED_CHARTS.get(intent, [])
    outside = [c for c in ("bar", "line", "scatter", "pie", "histogram", "box")
               if c not in allowed]
    if not outside:
        return None
    out = copy.deepcopy(target)
    out["chart_type"] = random.choice(outside)
    return out


def _corrupt_filter_to_one(target: dict, **_) -> dict | None:
    """Composition question filtered down to a single group (the Kentucky bug)"""
    gb = target["transform"].get("groupby")
    if not gb or "(" in gb:
        return None
    out = copy.deepcopy(target)
    out["transform"]["filter"] = f"{gb} == 'X'"
    return out


PLAN_CORRUPTIONS = [_corrupt_wrong_column, _corrupt_drop_filter, _corrupt_granularity,
                    _corrupt_agg, _corrupt_camel_case, _corrupt_drop_groupby,
                    _corrupt_filter_to_one, _corrupt_agg_slot, _corrupt_sort_direction,
                    _corrupt_split_derived]
# Only applicable when the target carries a derived measure:
MEASURE_CORRUPTIONS = [_corrupt_agg_slot, _corrupt_split_derived]

VIZ_CORRUPTIONS = [_corrupt_chart]
#################################


# Prompt construction (imports the agents' own builders — no drift):
def build_prompt(fmt: str, question: str, intent: str, schema_text: str,
                 viz_summary: str | None, allowed: list[str] | None) -> list[dict]:
    if fmt == "supervisor":
        return _build_intent_messages(question)
    if fmt == "data_analyst":
        return _build_plan_messages(schema_text, question, intent)
    if fmt == "visualization":
        return _build_viz_messages(question, intent, viz_summary, allowed)
    raise ValueError(fmt)


def as_completion(fmt: str, answer: dict, target_columns: list[str] | None = None) -> str:
    """The assistant turn the agent would have to produce for this format"""
    if fmt == "supervisor":
        return json.dumps({"intent": answer["intent"]})
    if fmt == "data_analyst":
        return json.dumps({"target_columns": target_columns or [],
                           "transform": answer["transform"]})
    if fmt == "visualization":
        return json.dumps({"chart_type": answer["chart_type"],
                           "reason": answer.get("reason", "")})
    raise ValueError(fmt)
#################################


# Source examples, prepared once:
def load_source_examples() -> list[dict]:
    examples = build_template_examples() + build_handwritten_examples()
    examples += [dict(e, source="failure_targeted") for e in FAILURE_EXAMPLES]
    seen, unique = set(), []
    for e in examples:
        key = (e["dataset"], e["question"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def prepare_context(ex: dict, frames: dict, schemas: dict) -> dict | None:
    """Run the target transform so the Visualization prompt gets real data facts"""
    try:
        rec = ChartRecommendation(**ex["target"])
        df = frames[ex["dataset"]]
        prepared, x_col, y_col, _series, notes = apply_transform(df, rec)
    except Exception:
        return None
    if prepared is None or prepared.empty:
        return None

    target_columns = [c for c in (ex["target"]["x_axis"], ex["target"]["y_axis"]) if c]
    plan = TransformPlan(transform=rec.transform, target_columns=target_columns,
                         result_rows=int(len(prepared)), notes=notes)
    facts = _data_facts(df, prepared, plan)
    summary = (f"{facts['n_rows']} rows; x={facts['x']} ({facts['n_categories']} categories), "
               f"y={facts['y']}; negatives={facts['has_negative']}")
    return {"df": df, "schema_text": schemas[ex["dataset"]], "summary": summary,
            "target_columns": target_columns}
#################################


# Pair assembly:
def make_pair(fmt: str, ex: dict, ctx: dict, rejected_answer: dict,
              source: str, corruption: str | None = None) -> dict | None:
    """Score both sides with the rubric; return the pair only if the order is clear"""
    intent = ex["intent"]
    target = ex["target"]
    allowed = ALLOWED_CHARTS.get(intent, [])

    ref = dict(target, target_columns=ctx["target_columns"],
               chart_family=[target["chart_type"]], intent=intent)
    chosen_answer = dict(target, target_columns=ctx["target_columns"], intent=intent)

    kwargs = dict(df=ctx["df"], intent=intent)
    s_chosen = score_candidate(chosen_answer, ref, fmt, **kwargs)
    s_rejected = score_candidate(rejected_answer, ref, fmt, **kwargs)
    verdict = compare(s_chosen, s_rejected)

    prompt = build_prompt(fmt, ex["question"], intent, ctx["schema_text"],
                          ctx["summary"], allowed)
    pair = {
        "format": fmt,
        "source": source,
        "dataset": ex["dataset"],
        "intent": intent,
        "question": ex["question"],
        "prompt": prompt,
        "chosen": as_completion(fmt, chosen_answer, ctx["target_columns"]),
        "rejected": as_completion(fmt, rejected_answer,
                                  rejected_answer.get("target_columns", ctx["target_columns"])),
        "corruption": corruption,               # which failure mode this pair teaches
        "scores": {"chosen": s_chosen, "rejected": s_rejected},
        "verdict": verdict,
    }
    return pair
#################################


# Source: synthetic corruption (no model needed)
def generate_synthetic(examples: list[dict], frames: dict, schemas: dict,
                       target_n: int) -> tuple[list[dict], list[dict]]:
    clear, unclear = [], []
    pool = examples[:]
    random.shuffle(pool)

    # One pair per question, cycling through the formats. Taking all three
    # formats from every question would reach the target with a third of the
    # questions, and prompt diversity is what makes preference data generalise —
    # 270 pairs over 270 questions beats 270 pairs over 90.
    rotation = ["data_analyst", "visualization", "supervisor"]

    for i, ex in enumerate(pool):
        if len(clear) >= target_n:
            break
        ctx = prepare_context(ex, frames, schemas)
        if ctx is None:
            continue
        intent = ex["intent"]
        fmt = rotation[i % len(rotation)]
        pair = None

        if fmt == "data_analyst":
            # Two corruptions only make sense on a derived measure, and there
            # are few such targets, so they are tried first when the target has
            # one. The rest rotate: with a random order the rare corruptions
            # almost never get their turn, because _corrupt_wrong_column applies
            # to everything and always wins the draw.
            has_measure = "(" in str(ex["target"].get("y_axis") or "")
            start = (i // len(rotation)) % len(PLAN_CORRUPTIONS)
            ordered = PLAN_CORRUPTIONS[start:] + PLAN_CORRUPTIONS[:start]
            if has_measure:
                ordered = MEASURE_CORRUPTIONS + [f for f in ordered
                                                 if f not in MEASURE_CORRUPTIONS]
            for fn in ordered:
                bad = fn(ex["target"], df=ctx["df"], intent=intent)
                if bad is None:
                    continue
                bad_cols = [c for c in (bad["x_axis"], bad["y_axis"]) if c]
                bad_cols += bad.pop("extra_columns", [])   # _corrupt_split_derived
                pair = make_pair("data_analyst", ex, ctx,
                                 dict(bad, target_columns=bad_cols, intent=intent),
                                 source="synthetic", corruption=fn.__name__)
                break
        elif fmt == "visualization":
            for fn in VIZ_CORRUPTIONS:
                bad = fn(ex["target"], df=ctx["df"], intent=intent)
                if bad is None:
                    continue
                pair = make_pair("visualization", ex, ctx, dict(bad, intent=intent),
                                 source="synthetic", corruption=fn.__name__)
                break
        else:
            wrong = random.choice([i for i in INTENTS if i != intent])
            pair = make_pair("supervisor", ex, ctx, {"intent": wrong},
                             source="synthetic", corruption="wrong_intent")

        if pair and pair["verdict"] == "a":
            clear.append(pair)
        elif pair and pair["verdict"] == "unclear":
            unclear.append(pair)

    return clear, unclear
#################################


# Source: model-generated candidates (needs a GPU)
def _parse_json(raw: str) -> dict | None:
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate_from_model(examples: list[dict], frames: dict, schemas: dict,
                        client: HFClient, source: str, target_n: int,
                        formats: list[str]) -> tuple[list[dict], list[dict]]:
    clear, unclear = [], []
    pool = examples[:]
    random.shuffle(pool)

    for i, ex in enumerate(pool):
        if len(clear) >= target_n:
            break
        ctx = prepare_context(ex, frames, schemas)
        if ctx is None:
            continue
        intent = ex["intent"]
        allowed = ALLOWED_CHARTS.get(intent, [])

        for fmt in [formats[i % len(formats)]]:      # rotate: one format per question
            prompt = build_prompt(fmt, ex["question"], intent, ctx["schema_text"],
                                  ctx["summary"], allowed)
            parsed = _parse_json(client.generate(prompt))
            if not parsed:
                continue

            if fmt == "supervisor":
                cand = {"intent": parsed.get("intent")}
            elif fmt == "data_analyst":
                cand = {"x_axis": None, "y_axis": None,
                        "target_columns": parsed.get("target_columns") or [],
                        "transform": parsed.get("transform") or {}}
            else:
                cand = {"chart_type": parsed.get("chart_type"),
                        "reason": parsed.get("reason", "")}

            pair = make_pair(fmt, ex, ctx, dict(cand, intent=cand.get("intent", intent)),
                             source=source)
            if pair and pair["verdict"] == "a":
                clear.append(pair)
            elif pair and pair["verdict"] == "unclear":
                unclear.append(pair)

        if (i + 1) % 20 == 0:
            print(f"  {i+1} questions scanned, {len(clear)} clear pairs", flush=True)

    return clear, unclear
#################################


# Contamination guard:
def check_against_benchmark(pairs: list[dict]) -> list[str]:
    """DPO prompts must not contain benchmark questions (dev or test)"""
    bench = json.loads(Path("evaluation/benchmark.json").read_text(encoding="utf-8"))
    bench_qs = {q["question"].strip().lower() for q in bench["questions"]}
    return [p["question"] for p in pairs if p["question"].strip().lower() in bench_qs]
#################################


# Main:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "base", "sft_temp"], default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--target", type=int, default=270,
                    help="how many clear pairs to collect from this source")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--merge", action="store_true", help="combine per-source files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        all_pairs, all_unclear = [], []
        for f in sorted(OUT_DIR.glob("pairs_*.jsonl")):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            (all_unclear if "unclear" in f.name else all_pairs).extend(rows)

        hits = check_against_benchmark(all_pairs)
        if hits:
            print(f"ABORT: {len(hits)} benchmark question(s) leaked into the pairs:")
            for h in hits[:5]:
                print(f"  {h}")
            return 1

        random.shuffle(all_pairs)
        out = OUT_DIR / "dpo_train.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps({"prompt": p["prompt"], "chosen": p["chosen"],
                                    "rejected": p["rejected"],
                                    "meta": {k: p[k] for k in
                                             ("format", "source", "dataset", "intent")}},
                                   ensure_ascii=False) + "\n")

        import collections
        print(f"merged {len(all_pairs)} pairs -> {out}")
        print("  by format:", dict(collections.Counter(p["format"] for p in all_pairs)))
        print("  by source:", dict(collections.Counter(p["source"] for p in all_pairs)))
        print("  by intent:", dict(collections.Counter(p["intent"] for p in all_pairs)))
        print("  by corruption:", dict(collections.Counter(
            p.get("corruption") or "-" for p in all_pairs)))
        print(f"  unique questions: {len({p['question'] for p in all_pairs})}")
        print(f"  unclear (manual review): {len(all_unclear)}")
        print("contamination check: clean")
        return 0

    if not args.source:
        raise SystemExit("pass --source synthetic|base|sft_temp, or --merge")

    examples = load_source_examples()
    frames = {k: load_table(v["path"]) for k, v in DATASETS.items()}
    schemas = {k: schema_summary(profile_table(df, k)) for k, df in frames.items()}
    print(f"source questions: {len(examples)} | target: {args.target} clear pairs")

    if args.source == "synthetic":
        clear, unclear = generate_synthetic(examples, frames, schemas, args.target)
    else:
        adapter = args.adapter if args.source == "sft_temp" else None
        temp = args.temperature if args.source == "sft_temp" else 0.0
        client = HFClient(model_name=args.base_model, adapter=adapter, temperature=temp)
        clear, unclear = generate_from_model(
            examples, frames, schemas, client, args.source, args.target,
            formats=["data_analyst", "visualization", "supervisor"])

    for name, rows in [(f"pairs_{args.source}.jsonl", clear),
                       (f"pairs_{args.source}_unclear.jsonl", unclear)]:
        path = OUT_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(rows):4} -> {path}")

    import collections
    print("  by format:", dict(collections.Counter(p["format"] for p in clear)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################