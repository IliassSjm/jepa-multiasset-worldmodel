"""Download utilities for raw market data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import warnings

import pandas as pd
import yfinance as yf

from ..config import RAW_DATA_PATH

ASSET_TICKERS: Dict[str, str] = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "EuroStoxx 50": "^STOXX50E",
    "Nikkei 225": "^N225",
    "US 10Y yield": "^TNX",
    "German 10Y yield": "DE10YBOND=X",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "VIX": "^VIX",
    "Gold": "GC=F",
}

DEFAULT_START_DATE = "2000-01-01"


def download_raw_prices(
    output_path: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
) -> Path:
    """Download raw daily close prices and store them as a long-format parquet file."""

    output_path = Path(output_path or RAW_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    price_panel = yf.download(
        tickers=list(ASSET_TICKERS.values()),
        start=start_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if price_panel.empty:
        raise RuntimeError("yfinance returned an empty DataFrame. Check tickers or network connectivity.")

    if isinstance(price_panel.columns, pd.MultiIndex):
        level0 = price_panel.columns.get_level_values(0)
        field = next((f for f in ("Adj Close", "Close") if f in level0), None)
        if field is None:
            raise KeyError("Downloaded panel missing both 'Adj Close' and 'Close' columns.")
        adj_close = price_panel[field]
    else:
        adj_close = price_panel

    reverse_map = {v: k for k, v in ASSET_TICKERS.items()}
    adj_close = adj_close.rename(columns=reverse_map)

    missing_assets = [asset for asset in ASSET_TICKERS if asset not in adj_close.columns]
    if missing_assets:
        warnings.warn(
            "Missing ticker data for: " + ", ".join(missing_assets) + ". Columns filled with NaN.",
            RuntimeWarning,
        )
        for asset in missing_assets:
            adj_close[asset] = pd.NA

    adj_close = adj_close[list(ASSET_TICKERS.keys())]

    calendar = pd.date_range(adj_close.index.min(), adj_close.index.max(), freq="B")
    adj_close = adj_close.reindex(calendar).ffill()

    long_df = (
        adj_close.stack(dropna=False)
        .rename("close_price")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )

    long_df.to_parquet(output_path, index=False)
    return output_path
