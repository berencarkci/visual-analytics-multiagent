"""Hugging Face Space app for the Visual Analytics Assistant.

Gradio interface: the Ask Your Data tab runs either the single agent baseline or
the multi agent workflow (mode selector) end to end. The Agent Trace tab shows
the step by step execution of the most recent multi agent question.

Environment:
    APP_MODE=mock   -> MockClient, instant canned answer for UI testing
    APP_MODE=live   -> HFClient (default)
    VA_ADAPTER=<id> -> which LoRA adapter to serve; empty string = base model
"""

from __future__ import annotations

import html as html_lib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

import gradio as gr
import pandas as pd

from baseline import recommend
from chart_render import render_chart
from data_ingestion import load_table, profile_table, schema_summary
from model_client import HFClient, MockClient
from orchestrator import run_workflow, trace_view
from labeling import build_labeling_tab

# The comparison tabs read pre-computed result files; they light up wherever
# those files are deployed and stay a placeholder where they are not.
try:
    from comparison import build_benchmark_tab, build_comparison_tab
    _HAS_COMPARISON = True
except Exception:
    _HAS_COMPARISON = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DATASETS = {
    "Retail Sales (Superstore)": DATA_DIR / "retail_sales_superstore.csv",
    "Customer Analytics (Mall)": DATA_DIR / "customer_analytics_mall.csv",
    "Energy Consumption (hourly)": DATA_DIR / "energy_consumption_hourly.csv",
}

DEFAULT_ADAPTER = "berencarkci/qwen2.5-3b-va-sft-v5"

SINGLE, MULTI = "Single agent", "Multi agent"
#################################


# Model client (single global, loaded once and kept in memory):
# Mock mode has to answer four different agents in order — intent, plan, chart,
# insight — not just the single-call format, or the multi agent pipeline stops
# at the Data Analyst with an empty plan.
_MOCK_SINGLE = (
    '{"chart_type": "bar", "x_axis": "category", "y_axis": "sales", '
    '"transform": {"groupby": "category", "series": null, "agg": "sum", "filter": null, '
    '"sort": "value_desc", "limit": null}, "reason": "Mock mode: canned answer for UI testing.", '
    '"insight": "This is a mock response, set APP_MODE=live for the real model."}'
)
_MOCK_CYCLE = [
    '{"intent": "comparison"}',
    '{"target_columns": ["category", "sales"], "transform": {"groupby": "category", '
    '"series": null, "agg": "sum", "filter": null, "sort": "value_desc", "limit": null}}',
    '{"chart_type": "bar", "reason": "Mock mode: canned chart choice."}',
    '{"insight": "Mock mode: no real statistics were computed."}',
]


def make_client():
    if os.environ.get("APP_MODE", "live") == "mock":
        return MockClient([_MOCK_SINGLE] + _MOCK_CYCLE * 6)
    adapter = os.environ.get("VA_ADAPTER", DEFAULT_ADAPTER)
    return HFClient(adapter=adapter or None)


CLIENT = make_client()
SERVING = ("mock mode" if os.environ.get("APP_MODE", "live") == "mock"
           else (os.environ.get("VA_ADAPTER", DEFAULT_ADAPTER) or "base model (no adapter)"))
_NO_TRACE = {"info": "Single agent mode does not produce a trace — switch to Multi agent."}
#################################


# Layout and the progress indicator. Every colour is read from the active Gradio
# theme through a CSS variable, so nothing here pins a palette.
CSS = """
.gradio-container { max-width: 1240px !important; }

#app-header h1 { margin-bottom: 6px; letter-spacing: -0.02em; }
.va-sub { color: var(--body-text-color-subdued); font-size: 0.9rem; line-height: 1.55; }
.va-tag {
  display: inline-block; padding: 2px 10px; margin-top: 10px;
  border: 1px solid var(--border-color-primary); border-radius: 999px;
  font-size: 0.78rem; color: var(--body-text-color-subdued);
  font-family: var(--font-mono);
}

/* the chart lands in a card, so an empty result does not look broken */
#chart-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-primary);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 8px; min-height: 80px;
}
#answer-box { font-size: 1.02rem; line-height: 1.6; padding-top: 4px; }

/* busy state: a real spinner, in the space the chart will occupy */
#status-line .va-busy {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 15px; padding: 34px 16px; margin: 8px 0;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.va-ring {
  width: 40px; height: 40px; border-radius: 50%;
  border: 3px solid var(--border-color-primary);
  border-top-color: var(--color-accent);
  animation: va-spin 0.8s linear infinite;
}
@keyframes va-spin { to { transform: rotate(360deg); } }
.va-busy-text {
  color: var(--body-text-color-subdued); font-size: 0.93rem;
  text-align: center; line-height: 1.5;
  animation: va-fade 1.6s ease-in-out infinite;
}
@keyframes va-fade { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

/* idle state: one quiet line */
#status-line .va-idle {
  display: flex; align-items: center; gap: 9px;
  padding: 7px 2px; font-size: 0.89rem;
  color: var(--body-text-color-subdued);
}
.va-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--color-accent); flex: none;
}
"""


def _busy(text: str) -> str:
    return (f'<div class="va-busy"><div class="va-ring"></div>'
            f'<div class="va-busy-text">{text}</div></div>')


def _idle(text: str = "") -> str:
    if not text:
        return ""
    return f'<div class="va-idle"><span class="va-dot"></span><span>{text}</span></div>'
#################################


# Data loading:
def load_selected(sample_name: str, uploaded_file):
    """Load the chosen dataset -> (df, profile, schema text, preview, status)"""
    try:
        if uploaded_file is not None:
            df = load_table(uploaded_file.name)
            source = Path(uploaded_file.name).name
        else:
            df = load_table(SAMPLE_DATASETS[sample_name])
            source = sample_name
        # profiled once here rather than on every question: it walks every column
        profile = profile_table(df, source)
        status = _idle(f"{source} — {len(df):,} rows x {df.shape[1]} columns")
        return df, profile, schema_summary(profile), df.head(10), status
    except Exception as e:
        return None, None, f"Could not load the file: {e}", None, _idle(f"Load failed: {e}")
#################################


def _render_html(df: pd.DataFrame, rec) -> tuple[str, list[str]]:
    """Figure -> self contained iframe (scripts run inside an iframe, no CDN dependency)"""
    fig, notes = render_chart(df, rec)
    fig_h = int(fig.layout.height or 430)
    raw = fig.to_html(full_html=True, include_plotlyjs=True, default_height=fig_h)
    html = ('<iframe srcdoc="' + html_lib.escape(raw)
            + f'" style="width:100%;height:{fig_h + 35}px;border:none;"></iframe>')
    return html, notes


# Question handling.
#
# Written as a generator so the interface can report progress: Gradio pushes
# every yield straight to the outputs. Without it the whole pipeline — up to six
# model calls in multi agent mode — looks like a frozen page.
def ask(df: pd.DataFrame | None, profile, schema_text: str, question: str, mode: str):
    if df is None:
        yield (_idle("No dataset loaded."), "", "Please load a dataset first.",
               "", "", "", None)
        return
    if not question.strip():
        yield (_idle("No question typed."), "", "Please type a question.",
               "", "", "", None)
        return

    yield (_busy("Reading the table…"), "", "", "", "", "", None)
    question = question.strip()
    retry_note = False
    eval_note = ""

    if mode == SINGLE:
        yield (_busy("Asking the model…"), "", "", "", "", "", None)
        result = recommend(CLIENT, schema_text, question)
        trace_data = _NO_TRACE
        if not result.valid:
            yield (_idle("No valid recommendation."), "",
                   f"The model could not produce a valid recommendation, even after a retry. "
                   f"Error: {result.error}", "", result.raw_first or "", "", trace_data)
            return
        rec = result.recommendation
        retry_note = result.used_retry
    else:
        yield (_busy("Running the agents<br>"
                     "supervisor · data analyst · visualization · insight"),
               "", "", "", "", "", None)
        if profile is None:
            profile = profile_table(df, "dataset")
        result = run_workflow(CLIENT, df, profile, question)
        trace_data = trace_view(result.trace)
        if not result.valid:
            yield (_idle("Workflow stopped."), "",
                   f"The workflow stopped at **{result.error.agent}** — "
                   f"{result.error.error_type}: {result.error.detail}",
                   "", "", "", trace_data)
            return
        rec = result.recommendation
        if result.verdict and not result.verdict.passed:
            eval_note = "\n\n*Evaluation flagged: " + "; ".join(result.verdict.issues) + "*"
        elif result.verdict and result.verdict.warnings:
            eval_note = "\n\n*Note: " + "; ".join(result.verdict.warnings) + "*"

    yield (_busy("Drawing the chart…"), "", "", "", "", "", None)
    try:
        chart_html, notes = _render_html(df, rec)
    except Exception as e:
        yield (_idle("Could not render."), "",
               f"The recommendation was schema valid but could not be rendered on this "
               f"dataset: {e}", rec.model_dump_json(indent=2), "", "", trace_data)
        return

    if mode == SINGLE:
        answer_md = ("*Insight withheld: in single agent mode the model never sees any "
                     "computed statistics, so its insight is unverifiable and frequently "
                     "invented. The raw model output is still in the JSON below.*"
                     f"\n\n**Why this chart:** {rec.reason}")
    else:
        answer_md = f"**Insight:** {rec.insight}\n\n**Why this chart:** {rec.reason}"
    answer_md += eval_note
    if retry_note:
        answer_md += "\n\n*The first model output was invalid; this answer came from the retry.*"

    notes_md = ('<span class="va-sub">Render notes: ' + "; ".join(notes) + "</span>") if notes else ""
    yield (_idle("Done."), chart_html, answer_md, rec.model_dump_json(indent=2),
           "", notes_md, trace_data)
#################################


# UI layout:
with gr.Blocks(title="Visual Analytics Assistant", css=CSS) as demo:
    with gr.Row(elem_id="app-header"):
        gr.Markdown(
            "# Visual Analytics Assistant\n"
            "Upload a table, ask a question in plain language, get a chart and an insight "
            "grounded in numbers that were actually computed.\n\n"
            "<span class='va-sub'>Single agent = one model call, no access to the data. "
            "Multi agent = supervisor, data analyst, visualization and insight agents, with a "
            "deterministic reviewer checking the result.</span><br>"
            f"<span class='va-tag'>serving {SERVING}</span>"
        )

    with gr.Tab("Ask Your Data"):
        with gr.Row():
            with gr.Column(scale=1):
                sample_dd = gr.Dropdown(choices=list(SAMPLE_DATASETS),
                                        value="Retail Sales (Superstore)",
                                        label="Sample dataset")
                upload = gr.File(label="…or upload your own (CSV / Excel / JSON / JSONL)")
                load_btn = gr.Button("Load dataset", variant="secondary")
                schema_box = gr.Textbox(label="Schema summary (what the model sees)",
                                        lines=11, interactive=False)
            with gr.Column(scale=2):
                preview = gr.Dataframe(label="Data preview (first 10 rows)",
                                       interactive=False)
                question_box = gr.Textbox(label="Your question", lines=2,
                                          placeholder="e.g. Compare total sales across product categories.")
                with gr.Row():
                    mode_radio = gr.Radio([SINGLE, MULTI], value=MULTI, label="Mode", scale=2)
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                status_md = gr.Markdown(elem_id="status-line")
                with gr.Group(elem_id="chart-panel"):
                    chart_out = gr.HTML()
                answer_out = gr.Markdown(elem_id="answer-box")
                notes_out = gr.Markdown()
                with gr.Accordion("Structured recommendation (JSON)", open=False):
                    json_out = gr.Code(language="json")
                with gr.Accordion("Raw model output (shown on failure)", open=False):
                    raw_out = gr.Textbox(lines=6, interactive=False)

        df_state = gr.State(value=None)
        profile_state = gr.State(value=None)
        trace_state = gr.State(value=None)

        load_btn.click(load_selected, [sample_dd, upload],
                       [df_state, profile_state, schema_box, preview, status_md])
        # a freshly uploaded file is loaded without a second click
        upload.change(load_selected, [sample_dd, upload],
                      [df_state, profile_state, schema_box, preview, status_md])

        _ASK_IO = dict(
            fn=ask,
            inputs=[df_state, profile_state, schema_box, question_box, mode_radio],
            outputs=[status_md, chart_out, answer_out, json_out, raw_out,
                     notes_out, trace_state],
        )
        ask_btn.click(
            lambda: gr.update(interactive=False, value="Working…"), None, [ask_btn]
        ).then(**_ASK_IO).then(
            lambda: gr.update(interactive=True, value="Ask"), None, [ask_btn]
        )
        question_box.submit(
            lambda: gr.update(interactive=False, value="Working…"), None, [ask_btn]
        ).then(**_ASK_IO).then(
            lambda: gr.update(interactive=True, value="Ask"), None, [ask_btn]
        )

    with gr.Tab("Agent Trace"):
        gr.Markdown(
            "Step by step execution of the most recent multi agent question: the plan, the "
            "columns chosen, the aggregation, the chart decision, any guardrail correction, "
            "and where the insight came from. Structured payloads only — no raw chain of thought."
        )
        refresh_btn = gr.Button("Show latest trace", variant="secondary")
        trace_out = gr.JSON(label="Workflow trace")
        refresh_btn.click(lambda t: t or {"info": "Run a multi agent question first."},
                          [trace_state], [trace_out])

    with gr.Tab("Model Comparison"):
        if _HAS_COMPARISON:
            build_comparison_tab()
        else:
            gr.Markdown("*Result files are not deployed here — see the repository for the "
                        "frozen benchmark comparison.*")

    with gr.Tab("Benchmark Results"):
        if _HAS_COMPARISON:
            build_benchmark_tab()
        else:
            gr.Markdown("*Result files are not deployed here — see the repository for the "
                        "metric tables.*")

    with gr.Tab("Preference Labeling"):
        build_labeling_tab(CLIENT, SAMPLE_DATASETS)

    with gr.Tab("Methodology"):
        gr.Markdown(
            "**Datasets.** Three public tabular datasets ship as samples (retail sales, "
            "customer analytics, energy consumption); provenance and cleaning steps are "
            "documented in the repository. Any uploaded table is profiled the same way.\n\n"
            "**Model.** Qwen2.5-3B-Instruct with a LoRA adapter trained by SFT + QLoRA over "
            "five agent output formats. A DPO variant was trained on 430 preference pairs and "
            "evaluated on the same frozen splits; it showed no measurable gain, and that is "
            "reported rather than dropped.\n\n"
            "**Modes.** Single agent answers in one call and never sees the data, which is why "
            "its insight is withheld here. Multi agent plans the transform, executes it, "
            "computes statistics, and writes an insight that is checked against those numbers "
            "before it is shown.\n\n"
            "**Guardrails.** Chart choices are verified against the prepared data — category "
            "counts, negative values, crowded axes — and corrected where a rule is violated. "
            "Corrections are displayed, not hidden.\n\n"
            "**Scope.** The assistant does not forecast and does not explain causes. Asked to, "
            "it shows the relevant history instead of inventing an answer.\n\n"
            "**Latency.** Multi agent makes several model calls per question, so it is slower "
            "than single agent. What that buys is an answer whose numbers were computed rather "
            "than generated."
        )

if __name__ == "__main__":
    demo.launch()
#################################