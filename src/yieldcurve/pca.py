"""PCA factor extraction for yield-curve changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    loadings: pd.DataFrame
    scores: pd.DataFrame
    explained_variance_ratio: pd.Series
    delta_yields: pd.DataFrame


def compute_daily_changes(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily changes in decimal yields."""
    delta = wide_df.sort_index().diff().dropna(how="all")
    delta = delta.interpolate(limit_direction="both").dropna(how="any")
    return delta


def fit_pca_factors(wide_df: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Fit PCA on daily yield changes and return loadings and score series."""
    delta = compute_daily_changes(wide_df)
    if delta.empty:
        raise ValueError("Not enough data to compute daily yield changes")

    n_comp = min(n_components, delta.shape[0], delta.shape[1])
    model = PCA(n_components=n_comp)
    scores_array = model.fit_transform(delta.to_numpy())

    pc_names = [f"PC{i}" for i in range(1, n_comp + 1)]
    maturity_cols = [float(c) for c in delta.columns]

    loadings = pd.DataFrame(
        model.components_,
        index=pc_names,
        columns=maturity_cols,
    )
    scores = pd.DataFrame(scores_array, index=delta.index, columns=pc_names)
    explained = pd.Series(model.explained_variance_ratio_, index=pc_names, name="explained_variance_ratio")

    return PCAResult(
        loadings=loadings,
        scores=scores,
        explained_variance_ratio=explained,
        delta_yields=delta,
    )
