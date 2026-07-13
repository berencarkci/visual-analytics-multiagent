"""Hugging Face Space app for the Visual Analytics Assistant.

Gradio interface: the Ask Your Data tab runs the single agent baseline end to end (dataset -> schema -> question -> validated recommendation -> chart). 
The other tabs are placeholders that later blocks will fill.

APP_MODE environment variable selects the model client:
    live (default) -> HFClient, real local model
    mock -> MockClient, instant canned answer for UI testing
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

# Sample datasets shipped with the repo:
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DATASETS = {
    "Retail Sales (Superstore)": DATA_DIR / "retail_sales_superstore.csv",
    "Customer Analytics (Mall)": DATA_DIR / "customer_analytics_mall.csv",
    "Energy Consumption (hourly)": DATA_DIR / "energy_consumption_hourly.csv",
}
#################################


# Model client selection (single global, loaded once and kept in memory):
_MOCK_ANSWER = (
    '{"chart_type": "bar", "x_axis": "category", "y_axis": "sales", '
    '"transform": {"groupby": "category", "agg": "sum", "filter": null, '
    '"sort": "value_desc", "limit": null}, "reason": "Mock mode: canned answer for UI testing.", '
    '"insight": "This is a mock response, set APP_MODE=live for the real model."}'
)


def make_client():
    if os.environ.get("APP_MODE", "live") == "mock":
        return MockClient([_MOCK_ANSWER])
    return HFClient()


CLIENT = make_client()
#################################


# Tab 1 - Ask for Datasets
def load_selected(sample_name: str, uploaded_file) -> tuple[pd.DataFrame | None, str, pd.DataFrame | None]:
    """Load the chosen dataset, return (df_state, schema_text, preview)"""
    try:
        if uploaded_file is not None:
            df = load_table(uploaded_file.name)
            source = Path(uploaded_file.name).name
        else:
            path = SAMPLE_DATASETS[sample_name]
            df = load_table(path)
            source = sample_name
        summary = schema_summary(profile_table(df, source))
        return df, summary, df.head(10)
    except Exception as e:
        return None, f"Could not load the file: {e}", None


def ask(df: pd.DataFrame | None, schema_text: str, question: str):
    """Question -> model -> validated recommendation -> chart + texts"""
    if df is None:
        return "", "Please load a dataset first.", "", "", ""
    if not question.strip():
        return "", "Please type a question.", "", "", ""

    result = recommend(CLIENT, schema_text, question.strip())

    if not result.valid:
        msg = f"The model could not produce a valid recommendation (after retry). Error: {result.error}"
        return "", msg, "", result.raw_first or "", ""

    rec = result.recommendation
    try:
        fig, notes = render_chart(df, rec)
        # gr.HTML injects via innerHTML, which never executes <script> tags, so we ship the chart as a self contained iframe (scripts DO run inside an iframe, and inlining plotly.js removes any CDN/network dependency)
        fig_h = int(fig.layout.height or 430)
        raw = fig.to_html(full_html=True, include_plotlyjs=True, default_height=fig_h)
        chart_html = ('<iframe srcdoc="' + html_lib.escape(raw) + f'" style="width:100%;height:{fig_h + 35}px;border:none;"></iframe>')
    except Exception as e:
        msg = (f"The recommendation was schema valid but could not be rendered on this dataset: {e}")
        return "", msg, rec.model_dump_json(indent=2), "", ""

    answer_md = f"**Insight:** {rec.insight}\n\n**Why this chart:** {rec.reason}"
    if result.used_retry:
        answer_md += "\n\n*Note: the first model output was invalid, this answer came from the retry.*"
    notes_md = ("**Render notes:** " + "; ".join(notes)) if notes else ""

    return chart_html, answer_md, rec.model_dump_json(indent=2), "", notes_md
#################################


# UI layout:
with gr.Blocks(title="Visual Analytics Assistant") as demo:
    gr.Markdown(
        "# Multi Agent Visual Analytics Assistant with Preference Optimization\n"
        "Upload tabular data, ask a question in natural language, get an appropriate chart and a grounded insight. *Single agent baseline version (Multi agent workflow, model comparison and benchmark results are coming in later stages).*"
    )

    with gr.Tab("Ask Your Data"):
        with gr.Row():
            with gr.Column(scale=1):
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLE_DATASETS), value="Retail Sales (Superstore)",
                    label="Sample dataset",
                )
                upload = gr.File(
                    label="...or upload your own (CSV / Excel / JSON / JSONL)",
                )
                load_btn = gr.Button("Load dataset", variant="secondary")
                schema_box = gr.Textbox(label="Schema summary (what the model sees)", lines=10, interactive=False)
            with gr.Column(scale=2):
                preview = gr.Dataframe(label="Data preview (first 10 rows)", interactive=False)
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Compare total sales across product categories.",
                )
                ask_btn = gr.Button("Ask", variant="primary")
                gr.Markdown("**Recommended chart**")
                chart_out = gr.HTML()
                answer_out = gr.Markdown()
                notes_out = gr.Markdown()
                with gr.Accordion("Details: structured recommendation (JSON)", open=False):
                    json_out = gr.Code(language="json")
                with gr.Accordion("Raw model output (only shown on failure)", open=False):
                    raw_out = gr.Textbox(lines=6, interactive=False)

        df_state = gr.State(value=None)

        load_btn.click(load_selected, [sample_dd, upload], [df_state, schema_box, preview])
        ask_btn.click(ask, [df_state, schema_box, question_box], [chart_out, answer_out, json_out, raw_out, notes_out])

    with gr.Tab("Agent Trace"):
        gr.Markdown("*Coming in later:* step by step execution summary of the multi agent workflow, plan, selected columns, aggregation, chart decision, evaluation result.")

    with gr.Tab("Model Comparison"):
        gr.Markdown("*Coming in later:* prompt only vs SFT vs DPO outputs side by side or the same unseen question, plus single agent vs multi agent comparison.")

    with gr.Tab("Benchmark Results"):
        gr.Markdown("*Coming in later:* metric tables, preference win rates and failure case explorer over the frozen 60 question benchmark.")

    with gr.Tab("Preference Labeling"):
        gr.Markdown("*Coming in later:* pairwise labeling interface and two candidate answers per prompt, pick A / B / tie / both poor.")

    with gr.Tab("Methodology"):
        gr.Markdown(
            "**Datasets:** three public tabular datasets (retail sales, customer analytics, energy consumption), provenance and cleaning steps are documented in the repository.\n\n"
            "**Model:** Qwen2.5-3B-Instruct with a frozen few shot prompt (prompt only baseline). SFT and DPO variants will be added and compared on a frozen held out benchmark split.\n\n"
            "**Note on latency:** this Space runs the model on free CPU hardware, responses can take several minutes. Local runs on GPU/MPS answer in seconds."
        )

if __name__ == "__main__":
    demo.launch()
#################################