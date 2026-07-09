"""Chart rendering for the Visual Analytics Assistant.

Turns a validated ChartRecommendation plus a DataFrame into a Plotly figure.
Applies the transform (filter -> derived grouping -> aggregation -> sort -> limit), then dispatches on chart_type. 
Kept deliberately pragmatic at baseline stage, the fully robust transform execution is the Data Analyst Agent's job. 
Problems are recorded in a notes list (future agent trace input) instead of raising, so the demo never crashes on a schema valid but odd recommendation.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px

from schemas import ChartRecommendation

# Derived grouping expressions:
_DERIVED_RE = re.compile(r"^(\w+)\((.+)\)$")


def _resolve_grouping(df: pd.DataFrame, expr: str) -> tuple[pd.Series, str]:
    """Turn a groupby expression into an actual grouping Series

    Plain column name -> the column itself.
    Derived notation   -> month(col), quarter(col), week(col), day(col),
                          year(col), hour_of_day(col), day_of_week(col),
                          weekend_flag(col), bins(col)
    Returns (series, label_for_axis).
    """
    m = _DERIVED_RE.match(expr.strip())
    if not m:
        return df[expr], expr

    func, inner = m.group(1), m.group(2).strip()

    if func in ("month", "quarter", "week", "day", "year"):
        dt = pd.to_datetime(df[inner])
        period = dt.dt.to_period({"month": "M", "quarter": "Q", "week": "W", "day": "D", "year": "Y"}[func])
        return period.dt.to_timestamp(), f"{func}({inner})"
    if func == "hour_of_day":
        return pd.to_datetime(df[inner]).dt.hour, "hour of day"
    if func == "day_of_week":
        return pd.to_datetime(df[inner]).dt.day_name(), "day of week"
    if func == "weekend_flag":
        wd = pd.to_datetime(df[inner]).dt.dayofweek
        return wd.map(lambda d: "weekend" if d >= 5 else "weekday"), "weekend vs weekday"
    if func == "bins":
        return pd.cut(df[inner], bins=5).astype(str), f"{inner} (binned)"

    raise ValueError(f"Unknown derived grouping: {expr}")
#################################


# Transform application:
_AGG_MAP = {"sum": "sum", "mean": "mean", "count": "count", "count_distinct": "nunique"}


def apply_transform(df: pd.DataFrame, rec: ChartRecommendation) -> tuple[pd.DataFrame, str, str | None, list[str]]:
    """Apply filter/groupby/agg/sort/limit, return (df, x_col, y_col, notes)

    Never raises for a recoverable issue: problems are recorded in `notes` and the step is skipped.
    """
    t = rec.transform
    notes: list[str] = []
    out = df.copy()

    # filter (pandas query; year(col) == N rewritten first)
    if t.filter:
        expr = re.sub(r"year\((\w+)\)", r"\1.dt.year", t.filter)
        try:
            if ".dt.year" in expr:
                col = re.search(r"(\w+)\.dt\.year", expr).group(1)
                out[col] = pd.to_datetime(out[col])
            out = out.query(expr)
            if out.empty:
                notes.append(f"filter left 0 rows: {t.filter}")
        except Exception as e:
            notes.append(f"filter skipped ({t.filter}): {e}")

    x_col, y_col = rec.x_axis, rec.y_axis

    # groupby + agg
    if t.groupby and t.agg:
        try:
            grouping, label = _resolve_grouping(out, t.groupby)
            target = y_col if y_col else rec.x_axis
            agg_fn = _AGG_MAP[t.agg]
            out = (out.assign(_grp=grouping)
                      .groupby("_grp", observed=True)[target]
                      .agg(agg_fn)
                      .reset_index()
                      .rename(columns={"_grp": label, target: f"{t.agg}({target})"}))
            x_col, y_col = label, f"{t.agg}({target})"
        except Exception as e:
            notes.append(f"groupby skipped ({t.groupby}): {e}")

    # sort
    if t.sort == "date_asc":
        out = out.sort_values(x_col)
    elif t.sort == "value_desc" and y_col:
        out = out.sort_values(y_col, ascending=False)

    # limit
    if t.limit:
        out = out.head(int(t.limit))

    return out, x_col, y_col, notes
#################################


# Chart dispatch:
def render_chart(df: pd.DataFrame, rec: ChartRecommendation):
    """ChartRecommendation + raw DataFrame -> (Plotly figure, notes)"""
    data, x, y, notes = apply_transform(df, rec)
    title = rec.reason if len(rec.reason) < 80 else rec.reason[:77] + "..."

    if rec.chart_type == "bar":
        fig = px.bar(data, x=x, y=y, title=title)
    elif rec.chart_type == "line":
        fig = px.line(data, x=x, y=y, title=title)
    elif rec.chart_type == "scatter":
        fig = px.scatter(data, x=x, y=y, title=title, opacity=0.6)
    elif rec.chart_type == "pie":
        fig = px.pie(data, names=x, values=y, title=title)
    elif rec.chart_type == "histogram":
        fig = px.histogram(data, x=x, title=title)
    elif rec.chart_type == "box":
        # grouped box (x categorical, y numeric) or single-variable box
        fig = px.box(data, x=x if y else None, y=y if y else x, title=title)
    else:  # pragma: no cover — schema already forbids this
        raise ValueError(f"Unsupported chart type: {rec.chart_type}")

    fig.update_layout(margin=dict(t=50, r=20, b=40, l=50))
    return fig, notes
#################################