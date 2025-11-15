"""Project-wide configuration constants for the data pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_prices.parquet"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_daily.parquet"
