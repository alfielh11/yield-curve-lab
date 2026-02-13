from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.config import PROCESSED_DATA_DIR
from yieldcurve.models import fit_over_dates, fitted_curve_from_params
from yieldcurve.plotting import plot_actual_vs_fitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Nelson-Siegel model over historical curves.")
    parser.add_argument(
        "--wide-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "yield_curve_wide.parquet",
        help="Path to wide matrix parquet file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    wide_df = pd.read_parquet(args.wide_path)
    wide_df.index = pd.to_datetime(wide_df.index)

    params_df = fit_over_dates(wide_df)
    params_path = PROCESSED_DATA_DIR / "nelson_siegel_params.parquet"
    params_df.to_parquet(params_path)
    logging.info("Saved: %s", params_path)

    sample_dates = [wide_df.index[0], wide_df.index[len(wide_df) // 2], wide_df.index[-1]]
    maturities = [float(c) for c in wide_df.columns]

    for dt in sample_dates:
        if dt not in params_df.index:
            continue
        actual = wide_df.loc[dt].dropna()
        params = params_df.loc[dt]
        fitted_values = fitted_curve_from_params(actual.index.to_numpy(dtype=float), params)
        fitted = pd.Series(fitted_values, index=actual.index)

        out_path = PROCESSED_DATA_DIR / f"ns_fit_{dt.date().isoformat()}.png"
        plot_actual_vs_fitted(
            maturities=list(actual.index.astype(float)),
            actual=actual,
            fitted=fitted,
            output_path=out_path,
            title=f"Nelson-Siegel Fit ({dt.date().isoformat()})",
        )
        logging.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
