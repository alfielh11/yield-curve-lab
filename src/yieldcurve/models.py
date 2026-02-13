"""Yield curve model implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NelsonSiegelParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    sse: float
    success: bool


def nelson_siegel_yield(
    maturity_years: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Compute Nelson-Siegel yield values (decimal units)."""
    t = np.asarray(maturity_years, dtype=float)
    tau = float(tau)
    x = np.clip(t / tau, 1e-8, None)
    level_slope = (1.0 - np.exp(-x)) / x
    curvature = level_slope - np.exp(-x)
    return beta0 + beta1 * level_slope + beta2 * curvature


def fit_nelson_siegel_for_day(
    maturities: np.ndarray,
    yields_decimal: np.ndarray,
) -> NelsonSiegelParams:
    """Fit Nelson-Siegel parameters for one day's observed curve."""
    maturities = np.asarray(maturities, dtype=float)
    yields_decimal = np.asarray(yields_decimal, dtype=float)

    valid = np.isfinite(maturities) & np.isfinite(yields_decimal)
    maturities = maturities[valid]
    yields_decimal = yields_decimal[valid]

    if maturities.size < 4:
        raise ValueError("At least 4 observed maturities are required for a stable fit")

    init = np.array([
        float(np.nanmean(yields_decimal)),
        -0.01,
        0.01,
        1.5,
    ])

    bounds = [
        (-0.10, 0.20),
        (-0.10, 0.20),
        (-0.10, 0.20),
        (0.05, 10.0),
    ]

    def objective(params: np.ndarray) -> float:
        y_hat = nelson_siegel_yield(maturities, *params)
        residuals = yields_decimal - y_hat
        return float(np.sum(residuals**2))

    result = minimize(
        objective,
        x0=init,
        method="L-BFGS-B",
        bounds=bounds,
    )

    beta0, beta1, beta2, tau = result.x
    return NelsonSiegelParams(
        beta0=float(beta0),
        beta1=float(beta1),
        beta2=float(beta2),
        tau=float(tau),
        sse=float(result.fun),
        success=bool(result.success),
    )


def fit_over_dates(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Fit Nelson-Siegel parameters for each row in a wide date x maturity matrix."""
    records: list[dict[str, float | bool | pd.Timestamp]] = []
    maturities = np.asarray([float(c) for c in wide_df.columns], dtype=float)

    for dt, row in wide_df.sort_index().iterrows():
        y = row.to_numpy(dtype=float)
        try:
            fit = fit_nelson_siegel_for_day(maturities, y)
            records.append(
                {
                    "date": pd.Timestamp(dt),
                    "beta0": fit.beta0,
                    "beta1": fit.beta1,
                    "beta2": fit.beta2,
                    "tau": fit.tau,
                    "sse": fit.sse,
                    "success": fit.success,
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed NS fit for date=%s: %s", dt, exc)

    if not records:
        raise ValueError("No Nelson-Siegel fits succeeded")

    params_df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return params_df


def fitted_curve_from_params(maturities: np.ndarray, params: pd.Series) -> np.ndarray:
    """Generate fitted curve values from a params row."""
    return nelson_siegel_yield(
        maturity_years=np.asarray(maturities, dtype=float),
        beta0=float(params["beta0"]),
        beta1=float(params["beta1"]),
        beta2=float(params["beta2"]),
        tau=float(params["tau"]),
    )
