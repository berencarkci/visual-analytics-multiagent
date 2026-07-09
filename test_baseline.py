"""Quick manual test for the single-agent baseline pipeline.

Usage:
    python test_baseline.py # fast, no model (just a preprepared llm output sent into the pipeline)
    python test_baseline.py --model # real model, end-to-end, opens a chart, preprepared question
    python test_baseline.py --model --question "Show monthly sales trend" # Your own question about the sales dataset
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "agents")

import pandas as pd

from baseline import recommend
from chart_render import render_chart
from model_client import MockClient
from schemas import validate_output

RETAIL = "data/retail_sales_superstore.csv"


# No-model pipeline test:
def test_pipeline() -> bool:
    """MockClient: checks wiring + the three retry outcomes + render"""
    good = ('{"chart_type": "bar", "x_axis": "category", "y_axis": "sales", '
            '"transform": {"groupby": "category", "agg": "sum", "filter": null, '
            '"sort": "value_desc", "limit": null}, "reason": "categorical comparison", '
            '"insight": "Categories differ in total sales."}')
    bad = '{"chart_type": "heatmap", "x_axis": "x"}'
    schema = "Table: 9994 rows x 19 columns\n- category (categorical, 3 categories)\n- sales (numeric, range 1-22638)"
    q = "Compare sales by category"

    r1 = recommend(MockClient([good]), schema, q)
    r2 = recommend(MockClient([bad, good]), schema, q)
    r3 = recommend(MockClient([bad, bad]), schema, q)

    print(f"first-try valid: {r1.valid} (retry={r1.used_retry})")
    print(f"retry rescued: {r2.valid} (retry={r2.used_retry})")
    print(f"double failure: invalid={not r3.valid}")

    fig, notes = render_chart(pd.read_csv(RETAIL), r1.recommendation)
    print(f"render: {len(fig.data)} series, notes={notes or 'none'}")

    ok = r1.valid and not r1.used_retry and r2.valid and r2.used_retry and not r3.valid and len(fig.data) > 0
    print("PIPELINE WORKS" if ok else "SOMETHING WRONG")
    return ok
#################################


# Real-model end-to-end test:
def test_model(question: str) -> None:
    """HFClient: real question -> real model -> validated rec -> chart in browser"""
    from data_ingestion import load_table, profile_table, schema_summary
    from model_client import HFClient

    df = load_table(RETAIL)
    schema = schema_summary(profile_table(df, "retail"))

    print(f"question: {question}\nmodel thinking...")
    r = recommend(HFClient(), schema, question)
    print(f"valid={r.valid} retry={r.used_retry}")

    if r.recommendation:
        print(r.recommendation.model_dump_json(indent=2))
        fig, notes = render_chart(df, r.recommendation)
        print(f"notes: {notes or 'none'}")
        fig.write_html("/tmp/test_grafik.html", auto_open=True)
        print("chart opened in browser.")
    else:
        print(f"  error: {r.error}")
#################################


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="run the real model end-to-end")
    ap.add_argument("--question", default="Compare total sales across product categories.")
    args = ap.parse_args()

    print("[pipeline test]")
    test_pipeline()

    if args.model:
        print("\n[model test]")
        test_model(args.question)