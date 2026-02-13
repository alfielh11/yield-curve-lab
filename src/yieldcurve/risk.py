"""Simple portfolio risk calculations using scenario curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskSummary:
    var_95: float
    es_95: float


def zcb_price(yield_decimal: float, maturity_years: float) -> float:
    """Price of a zero-coupon bond using continuous compounding."""
    return float(np.exp(-yield_decimal * maturity_years))


def portfolio_value(curve: pd.Series, exposures: dict[float, float]) -> float:
    """Portfolio value of notional-weighted ZCB ladder."""
    curve = curve.copy()
    curve.index = curve.index.astype(float)
    value = 0.0

    maturities_available = list(curve.index)
    for maturity, notional in exposures.items():
        nearest = min(maturities_available, key=lambda x: abs(x - maturity))
        y = float(curve.loc[nearest])
        value += notional * zcb_price(y, maturity)
    return value


def scenario_pnl(
    base_curve: pd.Series,
    scenario_curves: pd.DataFrame,
    exposures: dict[float, float],
) -> pd.Series:
    """Compute scenario P&L relative to base curve."""
    baseline_value = portfolio_value(base_curve, exposures)
    pnl = scenario_curves.apply(lambda row: portfolio_value(row, exposures) - baseline_value, axis=1)
    pnl.name = "pnl"
    return pnl


def var_es(pnl: pd.Series, confidence: float = 0.95) -> RiskSummary:
    """Compute VaR and Expected Shortfall from P&L series."""
    alpha = 1.0 - confidence
    quantile = float(np.quantile(pnl, alpha))
    var_value = -quantile
    tail = pnl[pnl <= quantile]
    es_value = -float(tail.mean()) if len(tail) else var_value
    return RiskSummary(var_95=var_value, es_95=es_value)
