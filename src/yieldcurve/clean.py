"""Cleaning and standardization utilities for Treasury yield data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EXPECTED_MATURITY_MAP


def treasury_row_to_long(row: pd.Series) -> pd.DataFrame:
    """Convert one Treasury row (wide) into tidy long format."""
    if "Date" not in row.index:
        raise ValueError("Treasury row missing Date field")

    row_date_raw = row["Date"]
    row_date = pd.to_datetime(row_date_raw).date()

    records: list[dict[str, Any]] = []
    for label, maturity_years in EXPECTED_MATURITY_MAP.items():
        if label not in row.index:
            continue

        raw_value = row[label]
        yield_pct = pd.to_numeric(raw_value, errors="coerce")
        if pd.isna(yield_pct):
            continue

        records.append(
            {
                "date": row_date.isoformat(),
                "maturity_years": float(maturity_years),
                "yield_pct": float(yield_pct),
                "yield_decimal": float(yield_pct) / 100.0,
            }
        )

    if not records:
        raise ValueError(f"No valid maturity values for date {row_date}")

    out = pd.DataFrame.from_records(records)
    out = out.sort_values("maturity_years").reset_index(drop=True)
    validate_long_format(out)
    return out


def treasury_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert many Treasury wide rows into one tidy long dataframe."""
    long_frames = [treasury_row_to_long(row) for _, row in df_wide.iterrows()]
    out = pd.concat(long_frames, ignore_index=True)
    out = out.sort_values(["date", "maturity_years"]).reset_index(drop=True)
    validate_long_format(out)
    return out


def long_to_wide_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format to date x maturity matrix in decimal yields."""
    required_cols = {"date", "maturity_years", "yield_decimal"}
    missing = required_cols - set(long_df.columns)
    if missing:
        raise ValueError(f"Long dataframe missing required columns: {sorted(missing)}")

    working = long_df.copy()
    working["date"] = pd.to_datetime(working["date"])

    wide = (
        working.pivot_table(
            index="date",
            columns="maturity_years",
            values="yield_decimal",
            aggfunc="first",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    wide.columns.name = None

    validate_wide_matrix(wide)
    return wide


def validate_long_format(long_df: pd.DataFrame) -> None:
    """Basic long-format checks for ordering and realistic bounds."""
    sorted_maturities = np.array(sorted(set(EXPECTED_MATURITY_MAP.values())))
    if not np.all(np.diff(sorted_maturities) > 0):
        raise ValueError("Configured maturities are not strictly increasing")

    if (long_df["yield_decimal"] < -0.05).any() or (long_df["yield_decimal"] > 0.25).any():
        raise ValueError("Yield decimals outside expected bounds [-5%, 25%]")


def validate_wide_matrix(wide_df: pd.DataFrame) -> None:
    """Ensure wide matrix columns are monotonic maturities."""
    columns = [float(c) for c in wide_df.columns]
    if any(next_col <= col for col, next_col in zip(columns[:-1], columns[1:])):
        raise ValueError("Wide matrix maturity columns are not strictly increasing")
