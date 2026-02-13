from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yieldcurve.config import PROCESSED_DATA_DIR
from yieldcurve.pca import fit_pca_factors
from yieldcurve.plotting import plot_factor_scores, plot_pca_loadings


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA factors on Treasury yield changes.")
    parser.add_argument(
        "--wide-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "yield_curve_wide.parquet",
        help="Path to wide matrix parquet file.",
    )
    parser.add_argument("--n-components", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    wide_df = pd.read_parquet(args.wide_path)
    wide_df.index = pd.to_datetime(wide_df.index)

    pca_result = fit_pca_factors(wide_df, n_components=args.n_components)

    loadings_path = PROCESSED_DATA_DIR / "pca_loadings.parquet"
    scores_path = PROCESSED_DATA_DIR / "pca_scores.parquet"
    explained_path = PROCESSED_DATA_DIR / "pca_explained_variance.parquet"

    pca_result.loadings.to_parquet(loadings_path)
    pca_result.scores.to_parquet(scores_path)
    pca_result.explained_variance_ratio.to_frame().to_parquet(explained_path)

    plot_pca_loadings(pca_result.loadings, PROCESSED_DATA_DIR / "pca_loadings.png")
    plot_factor_scores(pca_result.scores, PROCESSED_DATA_DIR / "pca_scores.png")

    logging.info("Saved: %s", loadings_path)
    logging.info("Saved: %s", scores_path)
    logging.info("Saved: %s", explained_path)
    logging.info("Explained variance: %s", pca_result.explained_variance_ratio.round(4).to_dict())


if __name__ == "__main__":
    main()
