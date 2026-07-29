"""Shared transform engine for the Visual Analytics Assistant.

Executes the structured transform (filter -> derived grouping -> aggregation -> sort -> limit) on a DataFrame, including the histogram guardrails discovered during Space testing. 
Lives in its own module because both the single agent baseline (via chart_render) and the multi agent Data Analyst import it: one source of truth, no cross system dependency in either direction.
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


# Derived measures on the y axis:
#
# Questions like "average delivery time" or "profit margin" ask for a quantity that is not a column: it has to be computed from two of them first, and only then aggregated. 
# Same notation as the derived groupings, so nothing new has to be added to the schema, y_axis simply accepts an expression as well.
_DERIVED_MEASURES = ("days_between", "ratio", "diff")


def _resolve_measure(df: pd.DataFrame, expr: str) -> tuple[pd.Series, str]:
    """Turn a y axis expression into a Series; plain column names pass through

    days_between(a, b) -> whole days from a to b (delivery time, delay)
    ratio(a, b) -> a / b, guarded against division by zero (margin, unit price)
    diff(a, b) -> a - b
    """
    m = _DERIVED_RE.match(expr.strip())
    if not m:
        return df[expr], expr
    func, inner = m.group(1), m.group(2)
    parts = [p.strip() for p in inner.split(",")]
    if func not in _DERIVED_MEASURES:
        raise ValueError(f"Unknown derived measure: {expr}")
    if len(parts) != 2:
        raise ValueError(f"{func} needs exactly two columns: {expr}")

    a, b = parts
    for col in (a, b):
        if col not in df.columns:
            raise ValueError(f"column not in dataset: {col}")

    if func == "days_between":
        delta = pd.to_datetime(df[b]) - pd.to_datetime(df[a])
        return delta.dt.total_seconds() / 86400, f"days between {a} and {b}"
    if func == "ratio":
        denom = df[b].replace(0, pd.NA) # 0 denominators become NaN, not inf
        return df[a] / denom, f"{a} / {b}"
    return df[a] - df[b], f"{a} - {b}"


def measure_base_columns(expr: str | None) -> list[str]:
    """Which real columns a y axis expression depends on"""
    if not expr:
        return []
    m = _DERIVED_RE.match(expr.strip())
    if not m or m.group(1) not in _DERIVED_MEASURES:
        return [expr.strip()]
    return [p.strip() for p in m.group(2).split(",")]
#################################

# Transform application:
_AGG_MAP = {"sum": "sum", "mean": "mean", "count": "count", "count_distinct": "nunique"}

# Date parts usable inside a filter. Only year() was handled before, so a filter
# like "hour_of_day(date) >= 22" raised inside query() and was skipped with a
# note — the question "how much energy do appliances use at night" then summed
# all 24 hours and answered a different question than the one asked.
_FILTER_PARTS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "quarter": lambda s: s.dt.quarter,
    "hour_of_day": lambda s: s.dt.hour,
    "day_of_week": lambda s: s.dt.day_name(),
    "weekend_flag": lambda s: s.dt.dayofweek >= 5,
}
def apply_transform(df: pd.DataFrame, rec: ChartRecommendation) -> tuple[pd.DataFrame, str, str | None, list[str]]:
    """Apply filter/groupby/agg/sort/limit, return (df, x_col, y_col, notes)
    Never raises for a recoverable issue: problems are recorded in 'notes' and the step is skipped.
    """
    t = rec.transform
    notes: list[str] = []
    out = df.copy()
    # filter (pandas query, year(col) == N rewritten first)
    if t.filter:
        expr = str(t.filter)
        work = out
        try:
            # Materialise every func(col) the filter references as a real column,
            # so pandas.query can see it. Temporary columns are dropped again
            # afterwards; on failure `out` is left untouched.
            for func, col in set(re.findall(r"(\w+)\((\w+)\)", expr)):
                if func in _FILTER_PARTS and col in work.columns:
                    tmp = f"_flt_{func}_{col}"
                    work = work.assign(
                        **{tmp: _FILTER_PARTS[func](pd.to_datetime(work[col]))})
                    expr = expr.replace(f"{func}({col})", tmp)
            work = work.query(expr)
            out = work.drop(columns=[c for c in work.columns if c.startswith("_flt_")])
            if out.empty:
                notes.append(f"filter left 0 rows: {t.filter}")
        except Exception as e:
            notes.append(f"filter skipped ({t.filter}): {e}")
    x_col, y_col = rec.x_axis, rec.y_axis

    # derived measure on y: materialise it as a real column before aggregating.
    # Any parenthesised y axis is treated as a measure attempt, including unknown functions, failing loudly here is better than silently dropping the y axis and aggregating something else instead.
    if y_col and _DERIVED_RE.match(y_col):
        try:
            series, label = _resolve_measure(out, y_col)
            out = out.assign(**{label: series})
            y_col = label
        except Exception as e:
            raise ColumnNotFoundError(f"cannot compute y axis '{y_col}': {e}", notes)

    # groupby + agg
    if t.groupby and t.agg:
        # histogram over an already numeric x bins the raw values itself, a groupby+agg would collapse the data first and destroy the distribution. Skip it with a note.
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
    if t.sort in ("date_asc", "date_desc") and x_col in out.columns:
        out = out.sort_values(x_col, ascending=t.sort == "date_asc")
    elif t.sort in ("value_asc", "value_desc") and y_col and y_col in out.columns:
        out = out.sort_values(y_col, ascending=t.sort == "value_asc")
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