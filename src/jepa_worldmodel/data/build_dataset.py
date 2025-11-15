"""Feature engineering pipeline for the canonical daily dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def build_daily_features(
    raw_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Compute daily log returns and realized volatility features.

    Parameters
    ----------
    raw_path:
        Path to ``market_prices.parquet``. Defaults to the canonical raw data path.
    output_path:
        Destination for the processed parquet table. Defaults to ``data/processed/market_daily.parquet``.

    Returns
    -------
    Path
        Location of the processed parquet file.
    """

    raw_path = Path(raw_path or RAW_DATA_PATH)
    output_path = Path(output_path or PROCESSED_DATA_PATH)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_path}")

    df = pd.read_parquet(raw_path)
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    df["log_return_1d"] = (
        df.groupby("asset")["close_price"]
        .transform(lambda s: np.log(s).diff())
    )

    df["realized_vol_20d"] = (
        df.groupby("asset")["log_return_1d"]
        .transform(lambda s: s.rolling(window=20, min_periods=5).std())
    )

    output_df = df[["date", "asset", "close_price", "log_return_1d", "realized_vol_20d"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    return output_path
