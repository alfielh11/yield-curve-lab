from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.config import PROCESSED_DATA_DIR
from yieldcurve.plotting import plot_histogram
from yieldcurve.risk import scenario_pnl, var_es


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy portfolio risk demo on scenario curves.")
    parser.add_argument(
        "--scenario-curves",
        type=Path,
        default=PROCESSED_DATA_DIR / "scenario_curves_parametric.parquet",
        help="Scenario curves parquet output from script 05.",
    )
    parser.add_argument(
        "--wide-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "yield_curve_wide.parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    scenario_curves = pd.read_parquet(args.scenario_curves)
    wide_df = pd.read_parquet(args.wide_path)
    base_curve = wide_df.iloc[-1].dropna()

    exposures = {2.0: 1_000_000.0, 5.0: 1_000_000.0, 10.0: 1_000_000.0}
    pnl = scenario_pnl(base_curve=base_curve, scenario_curves=scenario_curves, exposures=exposures)
    risk = var_es(pnl, confidence=0.95)

    risk_table = pd.DataFrame(
        {
            "metric": ["VaR_95", "ES_95"],
            "value": [risk.var_95, risk.es_95],
        }
    )

    table_path = PROCESSED_DATA_DIR / "risk_metrics.csv"
    risk_table.to_csv(table_path, index=False)

    plot_histogram(
        pnl,
        PROCESSED_DATA_DIR / "portfolio_pnl_distribution.png",
        title="Toy Portfolio P&L Distribution",
        xlabel="P&L",
    )

    print(risk_table.to_string(index=False))
    logging.info("Saved: %s", table_path)


if __name__ == "__main__":
    main()
