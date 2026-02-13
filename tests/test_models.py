from __future__ import annotations

import numpy as np

from yieldcurve.models import fit_nelson_siegel_for_day, nelson_siegel_yield


def test_nelson_siegel_shape_matches_input() -> None:
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    y = nelson_siegel_yield(maturities, beta0=0.03, beta1=-0.01, beta2=0.01, tau=1.5)
    assert y.shape == maturities.shape
    assert np.all(np.isfinite(y))


def test_fit_nelson_siegel_returns_finite_params() -> None:
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    yields = nelson_siegel_yield(maturities, beta0=0.035, beta1=-0.015, beta2=0.008, tau=1.8)
    yields = yields + np.array([0.0, 0.0001, -0.0001, 0.0001, 0.0, -0.0001, 0.0001])

    params = fit_nelson_siegel_for_day(maturities, yields)

    assert np.isfinite(params.beta0)
    assert np.isfinite(params.beta1)
    assert np.isfinite(params.beta2)
    assert np.isfinite(params.tau)
    assert params.tau > 0.0
