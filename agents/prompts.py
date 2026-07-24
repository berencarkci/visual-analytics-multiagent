"""Prompt templates for the prompt-only baseline.

This file is the prompt-only experimental arm. 
It may be iterated on the dev split, but must be frozen before the final test-split runs and must not change afterwards. 
Few shot examples use fictional schemas so no model configuration gets prior exposure to the project datasets.
"""

from __future__ import annotations

# System instruction:
SYSTEM_PROMPT = """You are a data visualization assistant. Given a table schema and an analytics question, you recommend exactly one chart and return ONLY a JSON object with this structure:

{
  "chart_type": one of "bar" | "line" | "scatter" | "pie" | "histogram" | "box",
  "x_axis": source column name from the schema,
  "y_axis": source column name, or null (e.g. for histograms),
  "transform": {
    "groupby": column or derived expression like "month(col)", "bins(col)", or null,
    "agg": one of "sum" | "mean" | "count" | "count_distinct", or null,
    "filter": condition string, or null,
    "sort": "date_asc" | "value_desc" | null,
    "limit": integer or null
  },
  "reason": one short sentence: why this chart fits the question intent,
  "insight": one short sentence grounded ONLY in the schema statistics provided; never invent numbers
}

Chart selection rules:
- Trend over time -> line
- Comparison across categories -> bar
- Relationship between two numeric columns -> scatter
- Part-to-whole share -> pie, ONLY if the metric is non-negative AND there are at most 5 categories; otherwise bar
- Distribution of one numeric column -> histogram or box
- Relationship where one variable takes only a few discrete values -> prefer box over scatter
- Anomaly or outlier detection over time -> line (box acceptable)
- x_axis is always a SOURCE column; if a derived grouping is needed, express it in transform.groupby

Return ONLY the JSON object. No explanations, no markdown fences."""
#################################

# Few shot examples (fictional schemas, one per chart type, 6 total):
FEW_SHOT_EXAMPLES = [
    {
        "schema": (
            "Table: 8400 rows x 4 columns\n"
            "- ride_date (datetime, from 2024-01-01 to 2024-12-31)\n"
            "- pickup_zone (categorical, 12 categories (e.g. Center, Airport, Harbor))\n"
            "- fare (numeric, range 3.5\u2013118.0)\n"
            "- distance_km (numeric, range 0.4\u201342.3)"
        ),
        "question": "How did the monthly number of rides change during the year?",
        "answer": '{"chart_type": "line", "x_axis": "ride_date", "y_axis": null, "transform": {"groupby": "month(ride_date)", "agg": "count", "filter": null, "sort": "date_asc", "limit": null}, "reason": "The question asks for a trend over time.", "insight": "Rides span the full year 2024, allowing a 12-point monthly trend."}',
    },
    {
        "schema": (
            "Table: 512 rows x 4 columns\n"
            "- employee_id (text, 512 unique)\n"
            "- department (categorical, 6 categories (e.g. Sales, Engineering, HR))\n"
            "- salary (numeric, range 31000\u2013142000)\n"
            "- hire_year (numeric, range 2009\u20132024)"
        ),
        "question": "Compare the average salary across departments.",
        "answer": '{"chart_type": "bar", "x_axis": "department", "y_axis": "salary", "transform": {"groupby": "department", "agg": "mean", "filter": null, "sort": "value_desc", "limit": null}, "reason": "Comparing a numeric metric across categories calls for a bar chart.", "insight": "Salaries range widely (31k to 142k), so departmental averages may differ substantially."}',
    },
    {
        "schema": (
            "Table: 950 rows x 4 columns\n"
            "- movie_title (text, 950 unique)\n"
            "- budget_musd (numeric, range 0.5\u2013210.0)\n"
            "- audience_rating (numeric, range 1.2\u20139.6)\n"
            "- release_year (numeric, range 1990\u20132024)"
        ),
        "question": "Is a movie's budget associated with its audience rating?",
        "answer": '{"chart_type": "scatter", "x_axis": "budget_musd", "y_axis": "audience_rating", "transform": {"groupby": null, "agg": null, "filter": null, "sort": null, "limit": null}, "reason": "The question asks about a relationship between two numeric variables.", "insight": "Both variables cover wide ranges, so a scatter plot can reveal whether any pattern exists."}',
    },
    {
        "schema": (
            "Table: 3100 rows x 3 columns\n"
            "- member_id (text, 3100 unique)\n"
            "- plan_type (categorical, 3 categories (e.g. Basic, Plus, Premium))\n"
            "- monthly_fee (numeric, range 9.9\u201349.9)"
        ),
        "question": "What share of members does each plan type represent?",
        "answer": '{"chart_type": "pie", "x_axis": "plan_type", "y_axis": "member_id", "transform": {"groupby": "plan_type", "agg": "count_distinct", "filter": null, "sort": null, "limit": null}, "reason": "Part-to-whole share with only 3 categories and a non-negative count suits a pie chart.", "insight": "With 3 plan types over 3100 members, the composition is easy to read as shares."}',
    },
    {
        "schema": (
            "Table: 15200 rows x 3 columns\n"
            "- session_id (text, 15200 unique)\n"
            "- duration_sec (numeric, range 2\u20135400)\n"
            "- device (categorical, 3 categories (e.g. mobile, desktop, tablet))"
        ),
        "question": "What is the distribution of session durations?",
        "answer": '{"chart_type": "histogram", "x_axis": "duration_sec", "y_axis": null, "transform": {"groupby": null, "agg": null, "filter": null, "sort": null, "limit": null}, "reason": "The question asks how a single numeric variable is distributed.", "insight": "Durations range from 2 to 5400 seconds, suggesting a long right tail worth inspecting."}',
    },
    {
        "schema": (
            "Table: 4700 rows x 4 columns\n"
            "- order_ref (text, 4700 unique)\n"
            "- provider (categorical, 4 categories (e.g. FastShip, EcoPost, CityGo))\n"
            "- delivery_days (numeric, range 1\u201319)\n"
            "- weight_kg (numeric, range 0.1\u201332.0)"
        ),
        "question": "How does delivery time vary across shipping providers?",
        "answer": '{"chart_type": "box", "x_axis": "provider", "y_axis": "delivery_days", "transform": {"groupby": "provider", "agg": null, "filter": null, "sort": null, "limit": null}, "reason": "Comparing the spread of a numeric variable across categories suits a box plot.", "insight": "Delivery times span 1 to 19 days, so per-provider spread and outliers are informative."}',
    },
]
#################################


# Prompt assembly:
def build_messages(schema_summary: str, question: str) -> list[dict]:
    """Assemble the chat messages: system + 6 worked examples + the real task"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"{ex['schema']}\n\nQuestion: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"{schema_summary}\n\nQuestion: {question}"})
    return messages
#################################

# MULTI AGENT PROMPTS
#
# Every prompt any agent sends to the model lives in this file (single
# source of truth). Unlike the frozen baseline prompt above, these may
# still be iterated during B2-B4 development; they freeze before the
# final B5 test-split runs.
 
# Supervisor - intent classification:
INTENT_SYSTEM = """Classify a data analytics question into exactly one intent label.
Return ONLY a JSON object: {"intent": "<label>"}
 
Labels:
- trend: change over time
- comparison: compare a metric across categories
- composition: part-to-whole share / mix
- relationship: association between two numeric variables
- distribution: how values of one variable are spread
- filter_aggregation: aggregate over a filtered subset (e.g. one year, one region, top N)
- anomaly: unusual values, outliers, spikes
 
Examples:
Q: "How did monthly sales change?" -> {"intent": "trend"}
Q: "Which region has the highest profit?" -> {"intent": "comparison"}
Q: "Were there any strange spikes in usage?" -> {"intent": "anomaly"}"""
 
 
# Data Analyst - transform planning (data preparation ONLY):
PLAN_SYSTEM = """You plan data preparation for an analytics question. Given a table schema and a question, return ONLY a JSON object:
 
{
  "target_columns": [source column names needed to answer the question],
  "transform": {
    "groupby": column or derived expression like "month(col)", "day(col)", "bins(col)", or null,
    "agg": "sum" | "mean" | "count" | "count_distinct" | null,
    "filter": pandas-query condition string, or null,
    "sort": "date_asc" | "value_desc" | null,
    "limit": integer or null
  }
}
 
Rules:
- Use ONLY column names that exist in the schema.
- Do NOT choose a chart type. Do NOT write insights. Data preparation only.
- Relationship questions between two raw numeric columns need no groupby/agg.
- Distribution questions on a numeric column need no groupby/agg.
- Share/composition questions about ONE specific category (e.g. "share of X"): do NOT filter to that category. Group by the category column over the WHOLE data; the share is computed from all groups."""
 
 
# Visualization Agent - chart choice ONLY:
VIZ_SYSTEM = """You choose the best chart for an analytics question. You are given the question, its intent, a short summary of the ALREADY PREPARED data, and the list of allowed chart types for this intent.
 
Return ONLY a JSON object: {"chart_type": "<one of the allowed types>", "reason": "<one short sentence>"}
 
Do NOT plan data transformations. Do NOT write insights. Chart choice only."""
 
 
# Insight Agent - grounded statement from computed statistics:
INSIGHT_SYSTEM = """You write ONE short data insight (1-2 sentences) answering the question, using ONLY the numbers and labels in the provided statistics. 
 
Rules:
- Every number you mention MUST appear in the statistics. Do not compute new numbers, do not round differently, do not invent values.
- No speculation ("might", "suggests a potential"), no claims beyond the statistics.
- Return ONLY a JSON object: {"insight": "<your sentence(s)>"}"""
 
 
# SFT training - short system prompt (no few-shots; Decision C in docs/data.md):
SFT_SYSTEM = """You are a visual analytics assistant. Given a table schema and a question, return ONLY a JSON object:
{"chart_type": "bar"|"line"|"scatter"|"pie"|"histogram"|"box", "x_axis": <source column>, "y_axis": <source column or null>, "transform": {"groupby": <column, derived expression like month(col)/day_of_week(col)/bins(col), or null>, "agg": "sum"|"mean"|"count"|"count_distinct"|null, "filter": <pandas query or null>, "sort": "date_asc"|"value_desc"|null, "limit": <int or null>}, "reason": <one sentence>, "insight": <one sentence describing what the chart shows>}
 
Rules: use only columns from the schema; x_axis is always the source column (never a derived label); the insight must not state numbers you cannot compute from the schema."""
#################################

# Fed back into a retried agent's prompt after the Evaluation Agent rejects an
# answer. Without it the retry re-sends a byte-identical prompt, and under
# greedy decoding that is guaranteed to reproduce the same rejected output —
# the retry costs a call and changes nothing.
REVIEW_FEEDBACK = ("A previous attempt was rejected by the reviewer for this reason: {issues}\n"
                   "Produce a corrected answer that specifically fixes it.")