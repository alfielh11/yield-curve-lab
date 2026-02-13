"""Scenario generation utilities based on PCA factor moves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    scenario_curves: pd.DataFrame
    factor_shocks: pd.DataFrame
    summary: pd.DataFrame


def _nearest_maturity(target: float, maturities: list[float]) -> float:
    return min(maturities, key=lambda x: abs(x - target))


def _factor_shocks_historical(
    factor_moves: pd.DataFrame,
    n_scenarios: int,
    random_state: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(factor_moves), size=n_scenarios)
    sampled = factor_moves.iloc[idx].reset_index(drop=True).copy()
    sampled.index.name = "scenario_id"
    return sampled


def _factor_shocks_parametric(
    factor_moves: pd.DataFrame,
    n_scenarios: int,
    random_state: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    mu = factor_moves.mean(axis=0).to_numpy()
    cov = np.cov(factor_moves.to_numpy(), rowvar=False)
    draws = rng.multivariate_normal(mean=mu, cov=cov, size=n_scenarios)
    return pd.DataFrame(draws, columns=factor_moves.columns)


def generate_pca_scenarios(
    latest_curve: pd.Series,
    loadings: pd.DataFrame,
    factor_moves: pd.DataFrame,
    method: str = "historical",
    n_scenarios: int = 1000,
    random_state: int | None = 42,
) -> ScenarioResult:
    """Generate yield-curve scenarios from PCA factor shocks."""
    latest_curve = latest_curve.copy()
    latest_curve.index = latest_curve.index.astype(float)
    maturities = list(latest_curve.index.astype(float))

    pc_cols = list(loadings.index)
    factor_moves = factor_moves[pc_cols]

    if method == "historical":
        factor_shocks = _factor_shocks_historical(factor_moves, n_scenarios, random_state)
    elif method == "parametric":
        factor_shocks = _factor_shocks_parametric(factor_moves, n_scenarios, random_state)
    else:
        raise ValueError("method must be 'historical' or 'parametric'")

    shock_matrix = factor_shocks.to_numpy() @ loadings.loc[pc_cols, maturities].to_numpy()
    delta_yield_df = pd.DataFrame(shock_matrix, columns=maturities)

    scenario_curves = delta_yield_df.add(latest_curve.to_numpy(), axis=1)
    scenario_curves.index.name = "scenario_id"

    m2 = _nearest_maturity(2.0, maturities)
    m10 = _nearest_maturity(10.0, maturities)
    base_2 = float(latest_curve.loc[m2])
    base_10 = float(latest_curve.loc[m10])

    summary = pd.DataFrame(
        {
            "method": method,
            "y10_change_bp": (scenario_curves[m10] - base_10) * 10_000.0,
            "s2s10_change_bp": (
                (scenario_curves[m10] - scenario_curves[m2]) - (base_10 - base_2)
            )
            * 10_000.0,
        }
    )
    summary.index.name = "scenario_id"

    return ScenarioResult(
        scenario_curves=scenario_curves,
        factor_shocks=factor_shocks,
        summary=summary,
    )
