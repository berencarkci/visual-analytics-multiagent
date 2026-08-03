"""Comparison and benchmark tabs, rendered from precomputed result files.

Running nine model configurations over the benchmark takes about an hour on a T4, which no free Space can do on demand. 
The measurement therefore runs offline and ships as JSON. 
These tabs are a reader for it, not a live evaluation.

Two files, both produced by the comparison runner in the development repo:

    results/arm_comparison.json    per-arm metric summaries
    results/comparison_cache.json  every arm's answer to every question

The adapter serving the Ask Your Data tab is newer than this measurement, on purpose. 
The benchmark run is frozen: remeasuring after later fixes would turn a held-out evaluation into model selection, which is exactly what a held-out split exists to prevent.

Usage from the app:
    from comparison import build_comparison_tab, build_benchmark_tab
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"
ARM_SUMMARY_PATH = RESULTS_DIR / "arm_comparison.json"
CACHE_PATH = RESULTS_DIR / "comparison_cache.json"

# Metric rows, in reading order. 
# A metric absent from every arm in a group is dropped from that table rather than shown as a column of dashes.
METRIC_ROWS = [
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

# Each group answers one question. 
# The preference optimisation group excludes SFT-v3 deliberately: both DPO adapters were trained on top of SFT-v2, so comparing them against v3 would confound preference optimisation with the larger SFT dataset.
GROUPS = [
    ("Training axis — single call",
     ["base_single", "sft_v2_single", "sft_v3_single"],
     "One model call per question. Nothing is executed, so a plan can look valid on paper; the arms barely separate here."),
    ("Training axis — multi-agent (SFT data expansion)",
     ["base_multi", "sft_v2_multi", "sft_v3_multi"],
     "The transform runs against the real data, so planning errors surface. This is where training shows."),
    ("Training axis — preference optimisation (all share the SFT-v2 base)",
     ["sft_v2_multi", "dpo_all_multi", "dpo_real_multi"],
     "Both DPO adapters were trained on top of SFT-v2 and are compared against it, never against SFT-v3."),
    ("Architecture axis — single vs multi vs multi + reviewer",
     ["base_single", "sft_v3_single", "sft_v3_multi_noeval", "sft_v3_multi"],
     "Same weights, different pipelines. The reviewer arm isolates what the Evaluation agent contributes."),
]
#################################


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
#################################


# Benchmark tab:
def _metric_tables(summary: dict) -> str:
    by_id = {s["id"]: s for s in summary.get("summaries", [])}
    split = summary.get("split", "?")
    n = next((s.get("n") for s in summary.get("summaries", [])), "?")

    out = [f"Measured on the **{split} split** ({n} questions), one greedy pass per "
           f"arm. Base model: `{summary.get('base_model', '?')}`.\n"]

    for title, arm_ids, note in GROUPS:
        arms = [by_id[a] for a in arm_ids if a in by_id]
        if not arms:
            continue
        out.append(f"### {title}\n\n*{note}*\n")
        header = "| metric | " + " | ".join(a["id"] for a in arms) + " |"
        sep = "|---" * (len(arms) + 1) + "|"
        rows = [header, sep]
        for key, label in METRIC_ROWS:
            if all(a.get(key) is None for a in arms):
                continue
            cells = ["–" if a.get(key) is None else str(a[key]) for a in arms]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("\n".join(rows) + "\n")
    return "\n".join(out)


def build_benchmark_tab() -> None:
    summary = _load(ARM_SUMMARY_PATH)
    if summary is None:
        gr.Markdown("*Result file not found — `results/arm_comparison.json` is missing from this deployment.*")
        return

    gr.Markdown("Nine configurations over the same questions, same protocol, one greedy pass each. Two independent experiments share the run: what training data buys, and what the agent architecture buys.")
    gr.Markdown(_metric_tables(summary))
    gr.Markdown(
        "---\n"
        "**How to read this.** Single-call arms and multi-agent arms are not directly comparable on groundedness: a single-call model never sees the data, so any figure it states is unsupported unless the prompt supplied it (*invented numbers*), while a multi-agent insight is handed computed statistics and is supposed to quote them (*insight grounded*). " \
        "The two rows measure different things.\n\n"
        "**Result.** Targeted SFT data produced a consistent held-out gain; preference optimisation did not. " \
        "DPO was trained on 430 pairs, evaluated on the same frozen splits, and showed no measurable improvement in either direction. "
        "That is reported rather than dropped: DPO polishes a behaviour the model already has, and the gaps in this task were missing behaviours, which is what supervised data is for.\n\n"
        "*The adapter serving the Ask Your Data tab is newer than this measurement. "
        "The benchmark run is frozen on purpose so remeasuring after later fixes would turn a held-out evaluation into model selection.*"
    )
#################################


# Comparison tab:
def _format_answer(arm_id: str, a: dict) -> str:
    if not a.get("valid"):
        return f"**{arm_id}** — chain stopped, no answer produced."
    axes = a.get("x_axis") or "?"
    if a.get("y_axis"):
        axes += f" / {a['y_axis']}"
    intent = f" · intent `{a['predicted_intent']}`" if a.get("predicted_intent") else ""
    insight = a.get("insight") or "*(no insight in this mode)*"
    return (f"**{arm_id}** — `{a.get('chart_type')}` on `{axes}`{intent} "
            f"· {a.get('seconds')}s\n\n> {insight}")


def _render_question(qid: str, cache: dict, arm_ids: list[str]) -> str:
    entry = cache.get(qid)
    if entry is None:
        return "*Question not found.*"
    head = (f"### {entry['question']}\n\n"
            f"*dataset:* `{entry['dataset']}` · *reference intent:* `{entry['type']}`\n")
    blocks = [_format_answer(a, entry["arms"][a]) for a in arm_ids if a in entry["arms"]]
    if not blocks:
        return head + "\n*No arms selected.*"
    return head + "\n\n---\n\n" + "\n\n".join(blocks)


def build_comparison_tab() -> None:
    cache = _load(CACHE_PATH)
    if cache is None:
        gr.Markdown("*Result file not found — `results/comparison_cache.json` is missing from this deployment.*")
        return

    qids = list(cache)
    all_arms = sorted({a for e in cache.values() for a in e["arms"]})
    default_arms = [a for a in ("base_multi", "sft_v2_multi", "sft_v3_multi", "dpo_all_multi") if a in all_arms] or all_arms[:4]

    def label_of(qid: str) -> str:
        e = cache[qid]
        return f"[{e['type']}] {e['question'][:70]}"

    choices = [(label_of(q), q) for q in qids]

    gr.Markdown(
        "The same question answered by every configuration, from the frozen benchmark run. " \
        "Answers are precomputed because running nine 3B configurations live is not something a free Space can do."
    )
    with gr.Row():
        q_dd = gr.Dropdown(choices=choices, value=qids[0], label="Question", scale=3)
        arm_sel = gr.CheckboxGroup(choices=all_arms, value=default_arms, label="Configurations", scale=2)
    out_md = gr.Markdown(_render_question(qids[0], cache, default_arms))

    for control in (q_dd, arm_sel):
        control.change(lambda q, a: _render_question(q, cache, a), [q_dd, arm_sel], [out_md])

    gr.Markdown(
        "---\n"
        "*`base_*` is the untrained model, `sft_*` are supervised checkpoints, `dpo_*` add preference optimisation on top of SFT-v2. " \
        "`*_single` answers in one call; `*_multi` runs the full agent pipeline; `*_noeval` runs it without the reviewer.*"
    )
#################################