from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.config import PROCESSED_DATA_DIR
from yieldcurve.plotting import plot_histogram, plot_scenario_fan
from yieldcurve.scenarios import generate_pca_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate yield curve scenarios from PCA factor moves.")
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wide-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "yield_curve_wide.parquet",
    )
    parser.add_argument(
        "--loadings-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "pca_loadings.parquet",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "pca_scores.parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    wide_df = pd.read_parquet(args.wide_path)
    loadings = pd.read_parquet(args.loadings_path)
    scores = pd.read_parquet(args.scores_path)

    latest_curve = wide_df.iloc[-1].dropna()

    hist = generate_pca_scenarios(
        latest_curve=latest_curve,
        loadings=loadings,
        factor_moves=scores,
        method="historical",
        n_scenarios=args.n_scenarios,
        random_state=args.seed,
    )
    para = generate_pca_scenarios(
        latest_curve=latest_curve,
        loadings=loadings,
        factor_moves=scores,
        method="parametric",
        n_scenarios=args.n_scenarios,
        random_state=args.seed,
    )

    summary = pd.concat([hist.summary, para.summary], axis=0)
    summary_path = PROCESSED_DATA_DIR / "scenario_summary.parquet"
    summary.to_parquet(summary_path)

    hist_curves_path = PROCESSED_DATA_DIR / "scenario_curves_historical.parquet"
    para_curves_path = PROCESSED_DATA_DIR / "scenario_curves_parametric.parquet"
    hist.scenario_curves.to_parquet(hist_curves_path)
    para.scenario_curves.to_parquet(para_curves_path)

    plot_scenario_fan(
        scenario_curves=hist.scenario_curves,
        baseline_curve=latest_curve,
        output_path=PROCESSED_DATA_DIR / "scenario_fan_historical.png",
        title="Historical PCA Scenario Fan",
    )
    plot_scenario_fan(
        scenario_curves=para.scenario_curves,
        baseline_curve=latest_curve,
        output_path=PROCESSED_DATA_DIR / "scenario_fan_parametric.png",
        title="Parametric PCA Scenario Fan",
    )
    plot_histogram(
        summary["y10_change_bp"],
        PROCESSED_DATA_DIR / "y10_change_distribution.png",
        title="Distribution of 10Y Yield Change",
        xlabel="10Y change (bp)",
    )
    plot_histogram(
        summary["s2s10_change_bp"],
        PROCESSED_DATA_DIR / "s2s10_change_distribution.png",
        title="Distribution of 2s10s Spread Change",
        xlabel="2s10s change (bp)",
    )

    logging.info("Saved: %s", summary_path)
    logging.info("Saved: %s", hist_curves_path)
    logging.info("Saved: %s", para_curves_path)


if __name__ == "__main__":
    main()
