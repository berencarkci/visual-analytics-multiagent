"""Chart rendering for the Visual Analytics Assistant.
Turns a validated ChartRecommendation plus a DataFrame into a Plotly figure. 
Applies the transform (filter -> derived grouping -> aggregation -> sort -> limit), then dispatches on chart_type. 
Problems are recorded in a notes list (future agent trace input) instead of raising, so the demo never crashes on a schema valid but odd recommendation.
"""
from __future__ import annotations
import re
import pandas as pd
import plotly.express as px
from schemas import ChartRecommendation
from transforms import ColumnNotFoundError, apply_transform


# Chart dispatch:
def render_chart(df: pd.DataFrame, rec: ChartRecommendation):
    """ChartRecommendation + raw DataFrame -> (Plotly figure, notes)"""
    data, x, y, series, notes = apply_transform(df, rec)
    title = rec.reason if len(rec.reason) < 80 else rec.reason[:77] + "..."
    has_series = bool(series and series in data.columns)

    if rec.chart_type == "bar":
        n_cat = data[x].nunique() if x in data.columns else 0
        if has_series:
            # grouped bars: the axis carries the first key, colour the second
            fig = px.bar(data, x=x, y=y, color=series, barmode="group", title=title)
            notes.append(f"grouped by '{series}' (colour)")
            if n_cat > 5:
                fig.update_xaxes(tickangle=-40, tickmode="linear")
        elif n_cat > 15:
            # if too many categories use horizontal bars(on too many categories in plotly every other label gets dropped)
            fig = px.bar(data.sort_values(y) if y in data.columns else data, x=y, y=x, orientation="h", title=title)
            fig.update_layout(height=max(450, 26 + 18 * n_cat), yaxis=dict(dtick=1))
            notes.append(f"drawn horizontally: {n_cat} categories (Plotly drops "
                         "every other label on a vertical axis this crowded)")
        else:
            fig = px.bar(data, x=x, y=y, title=title)
            if n_cat > 5:
                fig.update_xaxes(tickangle=-40, tickmode="linear")
    elif rec.chart_type == "line":
        fig = px.line(data, x=x, y=y, color=series if has_series else None, title=title)
        if has_series:
            notes.append(f"one line per '{series}'")
    elif rec.chart_type == "scatter":
        fig = px.scatter(data, x=x, y=y, title=title, opacity=0.6)
    elif rec.chart_type == "pie":
        fig = px.pie(data, names=x, values=y, title=title)
    elif rec.chart_type == "histogram":
        # histogram needs a numeric axis, small models sometimes point it at a categorical column (each label counted once -> meaningless flat bars).
        # If x is not numeric but a numeric y exists draw the distribution over that and say so in the notes.
        hist_col = x
        if hist_col not in data.columns or not pd.api.types.is_numeric_dtype(data[hist_col]):
            if y and y in data.columns and pd.api.types.is_numeric_dtype(data[y]):
                hist_col = y
                notes.append(f"histogram drawn over numeric '{y}' (x '{x}' is not numeric)")
            else:
                raise ColumnNotFoundError(
                    f"Histogram needs a numeric column, '{x}' is not numeric and no numeric y is available.",
                    notes,
                )
        fig = px.histogram(data, x=hist_col, title=title)
    elif rec.chart_type == "box":
        # grouped box (x categorical, y numeric) or single variable box
        fig = px.box(data, x=x if y else None, y=y if y else x, title=title)
    else:  # pragma: no cover, schema already forbids this
        raise ValueError(f"Unsupported chart type: {rec.chart_type}")
    
    fig.update_layout(margin=dict(t=50, r=20, b=40, l=50))
    return fig, notes
#################################