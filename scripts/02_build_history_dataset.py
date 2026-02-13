from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.clean import long_to_wide_matrix
from yieldcurve.config import PROCESSED_DATA_DIR, ensure_data_dirs
from yieldcurve.download import fetch_history_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical Treasury yield datasets.")
    parser.add_argument("--n-days", type=int, default=252, help="Approximate number of business days.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensure_data_dirs()

    long_df = fetch_history_curves(n_days=args.n_days)
    wide_df = long_to_wide_matrix(long_df)

    long_path = PROCESSED_DATA_DIR / "yield_curve_long.parquet"
    wide_path = PROCESSED_DATA_DIR / "yield_curve_wide.parquet"

    long_df.to_parquet(long_path, index=False)
    wide_df.to_parquet(wide_path)

    logging.info("Saved: %s", long_path)
    logging.info("Saved: %s", wide_path)
    logging.info("Rows (long): %s | Shape (wide): %s", len(long_df), wide_df.shape)


if __name__ == "__main__":
    main()
