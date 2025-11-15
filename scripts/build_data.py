"""Orchestrates the raw download and daily feature build."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from jepa_worldmodel.data.build_dataset import build_daily_features
from jepa_worldmodel.data.load_raw import download_raw_prices


def main() -> None:
    """Download raw prices and derive the canonical daily dataset."""

    raw_path = download_raw_prices()
    print(f"Raw data written to: {raw_path}")

    processed_path = build_daily_features(raw_path=raw_path)
    print(f"Processed dataset written to: {processed_path}")


if __name__ == "__main__":
    main()
