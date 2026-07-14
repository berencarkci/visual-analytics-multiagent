"""Shared transform engine for the Visual Analytics Assistant.

Executes the structured transform (filter -> derived grouping -> aggregation -> sort -> limit) on a DataFrame, including the histogram guardrails discovered during Space testing. 
Lives in its own module because both the single agent baseline (via chart_render) and the multi agent Data Analyst import it: one source of truth, no cross-system dependency in either direction.
"""
from __future__ import annotations

import re
import pandas as pd

from schemas import ChartRecommendation

class ColumnNotFoundError(ValueError):
    """Raised when the recommendation references columns missing from the data
    Carries the transform notes so the caller can show what else was skipped.
    """
    def __init__(self, message: str, notes: list[str]):
        super().__init__(message)
        self.notes = notes
        
# Derived grouping expressions:
_DERIVED_RE = re.compile(r"^(\w+)\((.+)\)$")
def _resolve_grouping(df: pd.DataFrame, expr: str) -> tuple[pd.Series, str]:
    """Turn a groupby expression into an actual grouping Series
    Plain column name -> the column itself.
    Derived notation-> month(col), quarter(col), week(col), day(col), year(col), hour_of_day(col), day_of_week(col), weekend_flag(col), bins(col)
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
    Never raises for a recoverable issue: problems are recorded in 'notes' and the step is skipped.
    """
    t = rec.transform
    notes: list[str] = []
    out = df.copy()
    # filter (pandas query, year(col) == N rewritten first)
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
        # histogram over an already numeric x bins the RAW values itself, a groupby+agg would collapse the data first and destroy the distribution. Skip it with a note.
        if (rec.chart_type == "histogram" and rec.x_axis in out.columns and pd.api.types.is_numeric_dtype(out[rec.x_axis])):
            notes.append("groupby/agg ignored: histogram bins the raw numeric values itself")
            t = t.model_copy(update={"groupby": None, "agg": None})

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

    # sort (check if the model referenced columns that do not exist)
    if t.sort == "date_asc" and x_col in out.columns:
        out = out.sort_values(x_col)
    elif t.sort == "value_desc" and y_col and y_col in out.columns:
        out = out.sort_values(y_col, ascending=False)
    elif t.sort:
        notes.append(f"sort skipped: column not found for '{t.sort}'")

    # limit
    if t.limit:
        out = out.head(int(t.limit))
        
    # final existence check: rendering needs real columns
    missing = [c for c in (x_col, y_col) if c is not None and c not in out.columns]
    if missing:
        notes.append(f"columns not in dataset: {missing}")
        raise ColumnNotFoundError(f"Recommended column(s) not found in the dataset: {missing}", notes)
    return out, x_col, y_col, notes
#################################