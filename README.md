# Yield Curve Lab
Project complete 13/02/2026 
macro/rates quant project that builds a small rates engine from public US Treasury data.

This repo shows how to:
- download and clean daily US Treasury yield curve data,
- fit a Nelson-Siegel curve model,
- extract PCA factors from curve changes,
- generate historical and parametric scenarios,
- run a simple portfolio stress test with VaR/ES.

Everything uses free Python libraries and free public data.

## Data Source

US Treasury daily yield curve rates (TextView page):
- `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=YYYY`

No API keys, paid services, or credentials are required.

## Project Structure

- `src/yieldcurve/`: core library code
- `scripts/`: runnable pipeline scripts (1 to 6)
- `tests/`: small sanity tests
- `data/raw/`: raw Treasury pulls (gitignored)
- `data/processed/`: outputs and plots (gitignored)

## Windows PowerShell Setup (Python 3.11+)

1. Clone and enter project:
```powershell
git clone <your-repo-url>
cd yield-curve-lab
```

2. Create and activate virtual environment:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Pipeline (in order)

1. Download latest curve and plot:
```powershell
python scripts/01_download_and_plot_curve.py
```

2. Build historical dataset (default ~252 business days):
```powershell
python scripts/02_build_history_dataset.py --n-days 252
```

3. Fit Nelson-Siegel per day:
```powershell
python scripts/03_fit_nelson_siegel.py
```

4. Run PCA on daily yield changes:
```powershell
python scripts/04_pca_factors.py --n-components 3
```

5. Generate scenario curves:
```powershell
python scripts/05_generate_scenarios.py --n-scenarios 1000 --seed 42
```

6. Run toy portfolio risk demo:
```powershell
python scripts/06_portfolio_risk_demo.py
```

## Test

```powershell
pytest -q
```

## Notebook Walkthrough

After running scripts `01` through `06`, open:

- `notebooks/01_explore_data.ipynb`

It provides a guided walkthrough for new users:
- checks expected output files exist,
- loads and visualizes curve history,
- reviews Nelson-Siegel parameters,
- reviews PCA loadings/scores and explained variance,
- reviews scenario distributions and portfolio risk metrics.

## Outputs

All outputs are written under `data/processed/` (gitignored), including:

- `latest_curve_long.csv`, `latest_curve.png`
- `yield_curve_long.parquet`, `yield_curve_wide.parquet`
- `nelson_siegel_params.parquet`
- `pca_loadings.parquet`, `pca_scores.parquet`, `pca_explained_variance.parquet`
- `scenario_summary.parquet`
- `scenario_curves_historical.parquet`, `scenario_curves_parametric.parquet`
- fan charts, histograms, and risk plots
- `risk_metrics.csv` with VaR(95%) and ES(95%)

## Notes and Common Errors

- `ImportError: lxml not found`:
  - install with `pip install lxml` and rerun.
- Treasury page format changes:
  - parsing may fail if table structure changes; check `src/yieldcurve/download.py` table selection logic.
- Missing latest-day data:
  - script 1 uses lookback (default 14 days) to find the most recent available curve.
- If a date/year fetch fails:
  - the downloader logs warnings and continues where possible.

## Model Notes

- Maturities used:
  - `1 Mo, 2 Mo, 3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr`
- Nelson-Siegel bounds:
  - betas in `[-0.10, 0.20]`, tau in `[0.05, 10]`
- PCA:
  - fit on daily changes `delta_y` in decimal yields
- Scenarios:
  - historical sampling and multivariate normal (parametric)

## License

MIT (see `LICENSE`).
