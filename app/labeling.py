"""Preference labeling tab for the Space.

Generates two candidate answers to the same question, shows them side by side without saying which is which, and records the labeller's choice. 
Every label is also scored by the rubric, so the session doubles as a measurement: how often does the mechanical labelling used for the 412 auto built pairs agree with a human?

Two bias controls that matter more than they look:

  * The two candidates are shuffled before display. 
    Without it a labeller drifts toward whichever slot the "first" answer occupies, and the agreement number measures that habit instead of the rubric.
  * The rubric's own verdict is never shown before the choice is made. 
    Seeing it first would anchor the labeller and inflate agreement.

Persistence is deliberately belt and braces: labels are appended to a local JSONL immediately, and can be pushed to a Hub dataset on demand. 
A Space restarts without warning and its disk does not survive that, so a label that exists only in memory is a label that will be lost.

Usage from the app:
    from labeling import build_labeling_tab
    with gr.Tab("Preference Labeling"):
        build_labeling_tab(CLIENT, SAMPLE_DATASETS)
"""

from __future__ import annotations

# ZeroGPU needs every GPU touching entry point marked. Outside a Space the package does not exist, so the decorator degrades to a no-op and the module still imports for local runs and tests.
try:
    import spaces
    _gpu_task = spaces.GPU(duration=90)
except ImportError:
    def _gpu_task(fn):
        return fn

import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

# The Space flattens the layout: labeling.py, rubric.py and agents/ all sit in the app root. 
# In the repo they are app/labeling.py, evaluation/rubric.py and agents/. Searching both the module's own directory and its parent covers each.
_HERE = Path(__file__).resolve().parent
for _dir in (_HERE, _HERE / "agents", _HERE.parent, _HERE.parent / "agents", _HERE.parent / "evaluation"):
    if _dir.is_dir():
        sys.path.insert(0, str(_dir))

from data_analyst import _build_plan_messages
from data_ingestion import load_table, profile_table, schema_summary
from rubric import compare, score_candidate
from supervisor import _build_intent_messages
from visualization import ALLOWED_CHARTS, _build_viz_messages

LABELS_PATH = Path("labels/preference_labels.jsonl")
HUB_DATASET = os.environ.get("LABEL_DATASET", "berencarkci/va-preference-labels")
ANNOTATOR = os.environ.get("ANNOTATOR", "beren")

# Sampling temperature for candidate generation. Greedy decoding would return the same answer twice and there would be nothing to compare.
CANDIDATE_TEMPERATURE = 0.9

FORMATS = ["data_analyst", "visualization", "supervisor"]
INTENTS = ["trend", "comparison", "composition", "relationship", "distribution", "filter_aggregation", "anomaly"]
#################################


# Candidate generation:
def _parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_prompt(fmt: str, question: str, intent: str, schema_text: str, summary: str, allowed: list[str]) -> list[dict]:
    if fmt == "supervisor":
        return _build_intent_messages(question)
    if fmt == "data_analyst":
        return _build_plan_messages(schema_text, question, intent)
    return _build_viz_messages(question, intent, summary, allowed)


@_gpu_task
def _sample_two(client, prompt: list[dict]) -> list[dict]:
    """Two independent samples, identical or unparseable ones are dropped"""
    original = client.temperature
    client.temperature = CANDIDATE_TEMPERATURE
    try:
        raws = [client.generate(prompt) for _ in range(2)]
    finally:
        client.temperature = original

    parsed = [p for p in (_parse_json(r) for r in raws) if p]
    if len(parsed) == 2 and json.dumps(parsed[0], sort_keys=True) == \
                            json.dumps(parsed[1], sort_keys=True):
        return [] # sampled the same answer twice
    return parsed
#################################


# Rubric scoring of a live pair (no ground truth available):
def _score_pair(cand_a: dict, cand_b: dict, fmt: str, df: pd.DataFrame,
                intent: str) -> tuple[dict, dict, str]:
    """Score both candidates against each other's context, not a reference

    There is no verified target for a freely typed question, so the rubric runs in reference free mode: the dimensions that need a reference (column match, transform match) fall back to their neutral grade, and the hard gates (schema validity, allowed chart list, groundedness) still fire. 
    That is enough to catch the answers the system itself would reject.
    """
    ref: dict = {"intent": intent}
    kwargs = dict(df=df, intent=intent)
    sa = score_candidate(cand_a, ref, fmt, **kwargs)
    sb = score_candidate(cand_b, ref, fmt, **kwargs)
    return sa, sb, compare(sa, sb)
#################################


# State handling. Gradio state carries a dict so the click handlers stay pure.
def generate_candidates(df, schema_text: str, question: str, fmt: str, intent: str, client) -> tuple:
    if df is None:
        return ("Load a dataset first.", "", "", None, _stats_markdown())
    if not question.strip():
        return ("Type a question first.", "", "", None, _stats_markdown())

    fmt = random.choice(FORMATS) if fmt == "auto" else fmt
    intent = random.choice(INTENTS) if intent == "auto" else intent
    allowed = ALLOWED_CHARTS.get(intent, [])
    summary = f"{len(df)} rows; columns: {', '.join(list(df.columns)[:6])}"

    prompt = _build_prompt(fmt, question.strip(), intent, schema_text, summary, allowed)
    parsed = _sample_two(client, prompt)
    if len(parsed) < 2:
        return ("The model produced identical or unparseable answers. Try again or rephrase the question.", "", "", None, _stats_markdown())

    order = [0, 1]
    random.shuffle(order) # position bias control
    cand_a, cand_b = parsed[order[0]], parsed[order[1]]

    state = {
        "prompt": prompt, "format": fmt, "intent": intent,
        "question": question.strip(),
        "cand_a": cand_a, "cand_b": cand_b,
        "df_columns": list(df.columns),
    }
    header = (f"**Format:** `{fmt}`  ·  **Intent:** `{intent}`  ·  "
              f"Pick the better answer, or mark them equal.")
    return (header, json.dumps(cand_a, indent=2), json.dumps(cand_b, indent=2), state, _stats_markdown())
#################################


# Label persistence:
def _load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    return [json.loads(l) for l in LABELS_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _stats_markdown() -> str:
    labels = _load_labels()
    if not labels:
        return "*No labels yet.*"

    decisive = [l for l in labels if l["meta"]["human_choice"] in ("a", "b") and l["meta"]["rubric_choice"] in ("a", "b")]
    agreed = sum(1 for l in decisive if
                 l["meta"]["human_choice"] == l["meta"]["rubric_choice"])
    by_choice: dict[str, int] = {}
    for l in labels:
        c = l["meta"]["human_choice"]
        by_choice[c] = by_choice.get(c, 0) + 1

    lines = [f"**{len(labels)} labels** — " +
             ", ".join(f"{k}: {v}" for k, v in sorted(by_choice.items()))]
    if decisive:
        pct = 100 * agreed / len(decisive)
        lines.append(f"**Rubric agreement:** {agreed}/{len(decisive)} ({pct:.0f}%) "
                     f"on pairs where both gave a verdict")
    return "\n\n".join(lines)


def save_label(choice: str, state: dict | None, df) -> tuple[str, str]:
    if not state:
        return "Generate a pair first.", _stats_markdown()

    fmt, intent = state["format"], state["intent"]
    sa, sb, rubric_choice = _score_pair(state["cand_a"], state["cand_b"], fmt, df, intent)

    if choice in ("a", "b"):
        chosen = state["cand_a"] if choice == "a" else state["cand_b"]
        rejected = state["cand_b"] if choice == "a" else state["cand_a"]
    else: # tie / both_bad: kept for the
        chosen = rejected = None  # agreement stats, not for training

    row = {
        "prompt": state["prompt"],
        "chosen": json.dumps(chosen) if chosen else None,
        "rejected": json.dumps(rejected) if rejected else None,
        "meta": {
            "source": "live_labeling",
            "format": fmt,
            "intent": intent,
            "question": state["question"],
            "annotator": ANNOTATOR,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "human_choice": choice,
            "rubric_choice": rubric_choice,
            "rubric_scores": {"a": sa, "b": sb},
            "usable": choice in ("a", "b"),
        },
    }

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    note = {"a": "Labelled: A preferred.", "b": "Labelled: B preferred.", "tie": "Labelled: equal.", "both_bad": "Labelled: both poor."}[choice]
    if rubric_choice in ("a", "b") and choice in ("a", "b"):
        note += "  (rubric agreed)" if rubric_choice == choice else "  (rubric disagreed)"
    return note, _stats_markdown()
#################################


# Export:
def export_labels() -> str | None:
    return str(LABELS_PATH) if LABELS_PATH.exists() else None


def push_to_hub() -> str:
    """Best effort: needs HF_TOKEN in the Space secrets with write access"""
    if not LABELS_PATH.exists():
        return "Nothing to push yet."
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HUB_DATASET, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(LABELS_PATH), path_in_repo="preference_labels.jsonl", repo_id=HUB_DATASET, repo_type="dataset")
        return f"Pushed {len(_load_labels())} labels to {HUB_DATASET}"
    except Exception as exc: # token missing, offline, quota...
        return (f"Push failed ({type(exc).__name__}: {exc}). Use the download button instead, the local file is intact.")
#################################


# Tab construction:
def build_labeling_tab(client, sample_datasets: dict) -> None:
    gr.Markdown(
        "Two candidate answers to the same question, generated by sampling the model twice. "
        "Pick the better one. "
        "The order is shuffled and the rubric's own verdict is hidden until after the choice, so the agreement number measures the rubric rather than the interface."
    )

    with gr.Row():
        with gr.Column(scale=1):
            sample_dd = gr.Dropdown(choices=list(sample_datasets), value=list(sample_datasets)[0], label="Dataset")
            load_btn = gr.Button("Load dataset", variant="secondary")
            fmt_dd = gr.Dropdown(choices=["auto"] + FORMATS, value="auto", label="Agent format")
            intent_dd = gr.Dropdown(choices=["auto"] + INTENTS, value="auto", label="Question intent")
            question_box = gr.Textbox(label="Question", lines=2, placeholder="e.g. Which weekdays use the most energy?")
            gen_btn = gr.Button("Generate two candidates", variant="primary")
            stats_md = gr.Markdown(_stats_markdown())

        with gr.Column(scale=2):
            header_md = gr.Markdown()
            with gr.Row():
                cand_a_box = gr.Code(label="Candidate A", language="json")
                cand_b_box = gr.Code(label="Candidate B", language="json")
            with gr.Row():
                a_btn = gr.Button("A is better", variant="primary")
                b_btn = gr.Button("B is better", variant="primary")
                tie_btn = gr.Button("Equal")
                bad_btn = gr.Button("Both poor")
            result_md = gr.Markdown()
            with gr.Row():
                dl_btn = gr.Button("Download labels")
                push_btn = gr.Button("Push to Hub")
            dl_file = gr.File(label="labels JSONL", interactive=False)

    df_state = gr.State(value=None)
    pair_state = gr.State(value=None)

    def _load(name):
        df = load_table(sample_datasets[name])
        return df, schema_summary(profile_table(df, name))

    schema_state = gr.State(value="")
    load_btn.click(_load, [sample_dd], [df_state, schema_state])

    gen_btn.click(lambda df, schema, q, f, i: generate_candidates(df, schema, q, f, i, client),
        [df_state, schema_state, question_box, fmt_dd, intent_dd],
        [header_md, cand_a_box, cand_b_box, pair_state, stats_md],
    )

    for btn, choice in [(a_btn, "a"), (b_btn, "b"), (tie_btn, "tie"), (bad_btn, "both_bad")]:
        btn.click(lambda s, d, c=choice: save_label(c, s, d), [pair_state, df_state], [result_md, stats_md])

    dl_btn.click(export_labels, None, dl_file)
    push_btn.click(push_to_hub, None, result_md)
#################################