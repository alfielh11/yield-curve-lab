"""Plotting utilities for curves, factors, scenarios, and risk outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_curve(curve_df: pd.DataFrame, output_path: Path, title: str = "Treasury Yield Curve") -> None:
    """Plot one curve from long dataframe containing maturity_years and yield_pct."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve_df["maturity_years"], curve_df["yield_pct"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_fitted(
    maturities: list[float],
    actual: pd.Series,
    fitted: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    """Plot actual vs fitted curve."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(maturities, actual.values * 100.0, marker="o", label="Actual")
    ax.plot(maturities, fitted.values * 100.0, marker="x", label="Nelson-Siegel fit")
    ax.set_title(title)
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_loadings(loadings: pd.DataFrame, output_path: Path) -> None:
    """Plot PCA loadings by maturity for each component."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = loadings.columns.astype(float)
    for pc in loadings.index:
        ax.plot(x, loadings.loc[pc].values, marker="o", label=pc)
    ax.set_title("PCA Loadings")
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Loading")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_factor_scores(scores: pd.DataFrame, output_path: Path) -> None:
    """Plot PCA factor score time series."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in scores.columns:
        ax.plot(scores.index, scores[col], label=col)
    ax.set_title("PCA Factor Scores")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scenario_fan(
    scenario_curves: pd.DataFrame,
    baseline_curve: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    """Plot percentile fan chart of scenario curves."""
    _prepare_output(output_path)
    maturities = baseline_curve.index.astype(float)
    quantiles = scenario_curves.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(maturities, quantiles.loc[0.05], quantiles.loc[0.95], alpha=0.2, label="5-95%")
    ax.fill_between(maturities, quantiles.loc[0.25], quantiles.loc[0.75], alpha=0.3, label="25-75%")
    ax.plot(maturities, quantiles.loc[0.5], label="Median", linewidth=2)
    ax.plot(maturities, baseline_curve.values, label="Baseline", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (decimal)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_histogram(series: pd.Series, output_path: Path, title: str, xlabel: str) -> None:
    """Generic histogram plot."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=40, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
