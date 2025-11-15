"""Typed schemas for raw and derived market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, TypeAlias

__all__ = ["MarketRow", "Tensor4DShape"]


@dataclass(frozen=True, slots=True)
class MarketRow:
    """Canonical daily market row as described in the project README."""

    date: date
    asset: str
    close_price: float
    log_return_1d: Optional[float]
    realized_vol_20d: Optional[float]


Tensor4DShape: TypeAlias = tuple[int, int, int, int]
"""Tuple describing the tensor shape [num_windows, L, n_assets, n_features],
with L = 60, n_assets = 10, and n_features = 3 (log_return_1d, realized_vol_20d,
plus one future feature slot)."""
