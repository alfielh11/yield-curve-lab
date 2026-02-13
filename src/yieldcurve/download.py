"""Download US Treasury daily yield curve data from public TextView pages."""

from __future__ import annotations

import logging
from io import StringIO
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from .clean import treasury_row_to_long
from .config import RAW_DATA_DIR, ensure_data_dirs

LOGGER = logging.getLogger(__name__)

TREASURY_TEXTVIEW_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
)


def _normalize_treasury_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(col).strip() for col in df.columns]
    df = df.copy()
    df.columns = columns
    if "Date" not in df.columns:
        raise ValueError("Treasury table does not include a Date column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def fetch_treasury_year_table(
    year: int,
    timeout_seconds: int = 30,
    session: requests.Session | None = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Fetch one calendar year's daily Treasury yield table."""
    ensure_data_dirs()
    url = TREASURY_TEXTVIEW_URL.format(year=year)
    http = session or requests.Session()
    LOGGER.info("Fetching Treasury table for year=%s", year)
    response = http.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"No tables found on Treasury page for year={year}")

    selected: pd.DataFrame | None = None
    for table in tables:
        if "Date" in [str(c).strip() for c in table.columns]:
            selected = table
            break
    if selected is None:
        raise ValueError(f"No Date table found on Treasury page for year={year}")

    normalized = _normalize_treasury_table(selected)
    normalized = normalized.sort_values("Date").reset_index(drop=True)

    if save_raw:
        raw_path = RAW_DATA_DIR / f"treasury_daily_yield_curve_{year}.csv"
        normalized.to_csv(raw_path, index=False)
        LOGGER.info("Saved raw table: %s", raw_path)

    return normalized


def _get_row_for_date(df_year: pd.DataFrame, target_date: date) -> pd.Series | None:
    match = df_year[df_year["Date"].dt.date == target_date]
    if match.empty:
        return None
    return match.iloc[0]


def fetch_latest_curve(
    as_of_date: date | None = None,
    max_lookback_days: int = 14,
    session: requests.Session | None = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Fetch most recent available Treasury curve with lookback fallback."""
    if as_of_date is None:
        as_of_date = datetime.now().date()

    year_cache: dict[int, pd.DataFrame] = {}
    http = session or requests.Session()

    for offset in range(max_lookback_days + 1):
        candidate = as_of_date - timedelta(days=offset)
        year = candidate.year
        if year not in year_cache:
            year_cache[year] = fetch_treasury_year_table(
                year=year,
                session=http,
                save_raw=save_raw,
            )

        row = _get_row_for_date(year_cache[year], candidate)
        if row is not None:
            long_df = treasury_row_to_long(row)
            LOGGER.info("Using curve date %s for as_of_date=%s", candidate, as_of_date)
            return long_df

    raise ValueError(f"No Treasury curve found within {max_lookback_days} days of {as_of_date}")


def fetch_history_curves(
    n_days: int = 252,
    end_date: date | None = None,
    session: requests.Session | None = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Fetch a long-format history of approximately n_days business-day curves."""
    if end_date is None:
        end_date = datetime.now().date()

    start_date = end_date - timedelta(days=int(n_days * 2.3) + 30)
    years = list(range(start_date.year, end_date.year + 1))
    http = session or requests.Session()

    frames: list[pd.DataFrame] = []
    for year in years:
        try:
            frames.append(fetch_treasury_year_table(year, session=http, save_raw=save_raw))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping year %s due to download error: %s", year, exc)

    if not frames:
        raise ValueError("Unable to fetch any Treasury data")

    raw_all = pd.concat(frames, ignore_index=True)
    raw_all = raw_all.drop_duplicates(subset=["Date"]).sort_values("Date")
    mask = (raw_all["Date"].dt.date >= start_date) & (raw_all["Date"].dt.date <= end_date)
    filtered = raw_all.loc[mask].copy()
    filtered = filtered.tail(n_days)

    long_frames: list[pd.DataFrame] = []
    for _, row in filtered.iterrows():
        try:
            long_frames.append(treasury_row_to_long(row))
        except Exception as exc:  # noqa: BLE001
            date_value = row.get("Date")
            LOGGER.warning("Skipping date=%s due to parse error: %s", date_value, exc)

    if not long_frames:
        raise ValueError("No curve rows could be converted to long format")

    out = pd.concat(long_frames, ignore_index=True).sort_values(["date", "maturity_years"])
    return out


def build_wide_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build date x maturity matrix from long format."""
    wide_df = long_df.pivot_table(
        index="date",
        columns="maturity_years",
        values="yield_decimal",
        aggfunc="first",
    ).sort_index()
    wide_df.columns.name = None
    return wide_df
