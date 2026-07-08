"""Data ingestion and profiling for the Visual Analytics Assistant.

Loads CSV/Excel files, extracts a typed schema, profiles the data, and produces a compact schema summary suitable for LLM prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

# Loading the file:
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_table(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Load CSV & Excel files into DataFrame with basic normalizations"""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding_errors="replace")
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    df.columns = [str(c).strip() for c in df.columns]
    return df
##################################

# Column Profiling:
@dataclass
class ColumnProfile:
    name: str
    dtype: str # data's type: numeric, categorical, datetime, boolean, text
    pandas_dtype: str
    missing_count: int
    missing_ratio: float
    unique_count: int
    sample_values: list = field(default_factory=list)

    # numeric-only
    min: float | None = None
    max: float | None = None
    mean: float | None = None

    # datetime-only
    date_min: str | None = None
    date_max: str | None = None


@dataclass
class TableProfile:
    source: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str, **kwargs)


_DATE_HINTS = ("date", "time", "tarih", "zaman", "month", "year", "ay", "yil", "yıl")


def _semantic_dtype(series: pd.Series, col_name: str) -> tuple[str, pd.Series]:
    """Infer a semantic type; return (dtype, possibly-converted series)"""
    if pd.api.types.is_bool_dtype(series):
        return "boolean", series
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", series
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", series

    # In object columns try datetime if the name hints at it or values parse correctly for a date
    non_null = series.dropna()
    if len(non_null) > 0:
        name_hints_date = any(h in col_name.lower() for h in _DATE_HINTS)
        try:
            # In object columns check if it is a date. take 200 samples from that column, if all 200 samples in that column parses correctly for a date take it directly as date else if over %95 of those samples parse correctly check if the name of that row indicates that it is a date, if it does take it as a date if it doesnt then dont
            sample = non_null.sample(min(len(non_null), 200), random_state=0)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            parse_ratio = parsed.notna().mean()
            if parse_ratio > 0.95 and (name_hints_date or parse_ratio == 1.0):
                converted = pd.to_datetime(series, errors="coerce", format="mixed")
                return "datetime", converted
        except (ValueError, TypeError):
            pass

    # If it wasnt date check if it is categorical or text. low cardinality is considered to be categorical.
    nunique = series.nunique(dropna=True)
    # If the column has less than 20 or %5 unique rows than that column is accepted as a category
    if nunique <= max(20, 0.05 * len(series)):
        return "categorical", series
    return "text", series


def profile_table(df: pd.DataFrame, source: str = "uploaded") -> TableProfile:
    """Build a full profile for the table. semantic type, missing stats, samples for every column"""
    profile = TableProfile(source=source, n_rows=len(df), n_cols=df.shape[1])

    for col in df.columns:
        series = df[col]
        dtype, converted = _semantic_dtype(series, col)
        non_null = converted.dropna()

        cp = ColumnProfile(
            name=col,
            dtype=dtype,
            pandas_dtype=str(series.dtype),
            missing_count=int(series.isna().sum()),
            missing_ratio=round(float(series.isna().mean()), 4),
            unique_count=int(series.nunique(dropna=True)),
            # take the 5 most frequent value in that column as sample
            sample_values=[str(v) for v in non_null.value_counts().head(5).index],
        )
        # if that column is numeric then take min, max, mean values for that column else if it is a date then take min and max for dates.
        if dtype == "numeric" and len(non_null) > 0:
            cp.min = round(float(non_null.min()), 4)
            cp.max = round(float(non_null.max()), 4)
            cp.mean = round(float(non_null.mean()), 4)
        elif dtype == "datetime" and len(non_null) > 0:
            cp.date_min = str(non_null.min())
            cp.date_max = str(non_null.max())

        profile.columns.append(cp)

    return profile
#################################


# Compact schema summary for LLM prompts:
def schema_summary(profile: TableProfile, max_samples: int = 3) -> str:
    """Render a compact, token efficient schema description for prompts

    Example: sales (numeric, range 12.5–8420.0, 0% missing)
    """
    lines = [f"Table: {profile.n_rows} rows x {profile.n_cols} columns"]
    for c in profile.columns:
        extra = ""
        if c.dtype == "numeric" and c.min is not None:
            extra = f", range {c.min}\u2013{c.max}"
        elif c.dtype == "datetime" and c.date_min:
            extra = f", from {c.date_min[:10]} to {c.date_max[:10]}"
        elif c.dtype == "categorical":
            samples = ", ".join(c.sample_values[:max_samples])
            extra = f", {c.unique_count} categories (e.g. {samples})"
        miss = f", {round(c.missing_ratio * 100, 1)}% missing" if c.missing_count else ""
        lines.append(f"- {c.name} ({c.dtype}{extra}{miss})")
    return "\n".join(lines)


# CLI for quick manual checks
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_ingestion.py <file.csv|file.xlsx>")
        raise SystemExit(1)

    frame = load_table(sys.argv[1])
    prof = profile_table(frame, source=Path(sys.argv[1]).name)
    print(schema_summary(prof))
    print("\nfull profile (JSON)")
    print(prof.to_json(indent=2))
#############################