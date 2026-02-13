from __future__ import annotations

import pandas as pd

from yieldcurve.clean import long_to_wide_matrix, treasury_row_to_long
from yieldcurve.config import EXPECTED_MATURITY_MAP


def test_treasury_row_to_long_has_expected_maturities() -> None:
    row_data = {"Date": "2025-01-02"}
    for i, label in enumerate(EXPECTED_MATURITY_MAP, start=1):
        row_data[label] = str(2.0 + i * 0.1)

    row = pd.Series(row_data)
    long_df = treasury_row_to_long(row)

    assert set(long_df.columns) == {"date", "maturity_years", "yield_pct", "yield_decimal"}
    assert len(long_df) == len(EXPECTED_MATURITY_MAP)
    assert long_df["maturity_years"].is_monotonic_increasing


def test_long_to_wide_matrix_shape() -> None:
    long_df = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "maturity_years": [2.0, 10.0, 2.0, 10.0],
            "yield_decimal": [0.03, 0.04, 0.031, 0.041],
        }
    )

    wide = long_to_wide_matrix(long_df)

    assert wide.shape == (2, 2)
    assert list(wide.columns) == [2.0, 10.0]
