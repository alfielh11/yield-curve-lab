from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.config import PROCESSED_DATA_DIR, ensure_data_dirs
from yieldcurve.download import fetch_latest_curve
from yieldcurve.plotting import plot_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest Treasury curve and plot it.")
    parser.add_argument("--lookback", type=int, default=14, help="Maximum lookback days for missing data.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensure_data_dirs()

    curve_long = fetch_latest_curve(max_lookback_days=args.lookback)
    date_label = curve_long["date"].iloc[0]

    csv_path = PROCESSED_DATA_DIR / "latest_curve_long.csv"
    png_path = PROCESSED_DATA_DIR / "latest_curve.png"
    curve_long.to_csv(csv_path, index=False)
    plot_curve(curve_long, png_path, title=f"US Treasury Curve ({date_label})")

    logging.info("Saved: %s", csv_path)
    logging.info("Saved: %s", png_path)


if __name__ == "__main__":
    main()
