"""Project configuration and common paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXPECTED_MATURITY_MAP: dict[str, float] = {
    "1 Mo": 1.0 / 12.0,
    "2 Mo": 2.0 / 12.0,
    "3 Mo": 3.0 / 12.0,
    "4 Mo": 4.0 / 12.0,
    "6 Mo": 6.0 / 12.0,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}


def ensure_data_dirs() -> None:
    """Create standard data directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
