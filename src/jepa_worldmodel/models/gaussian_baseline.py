"""Simple Gaussian baseline for multi-asset returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaussianReturnModel:
    """Empirical multivariate normal approximation for asset returns."""

    assets: list[str]
    mean: np.ndarray
    cov: np.ndarray

    MIN_VARIANCE: float = 1e-8

    @classmethod
    def fit(cls, df: pd.DataFrame, return_col: str = "log_return_1d") -> "GaussianReturnModel":
        """Estimate the Gaussian model from a long-format return table."""

        clean_df = df.dropna(subset=[return_col])
        pivot = (
            clean_df.pivot(index="date", columns="asset", values=return_col)
            .dropna(axis=1, how="all")
        )

        assets = pivot.columns.tolist()
        if not assets:
            raise ValueError("No assets with available returns were found.")

        returns = pivot.to_numpy()
        mean = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False)
        cov = cov + np.eye(len(assets)) * cls.MIN_VARIANCE

        return cls(assets=assets, mean=mean, cov=cov)

    def sample_paths(
        self,
        n_steps: int,
        n_scenarios: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Sample IID multivariate Gaussian return paths."""

        rng = np.random.default_rng(random_state)
        samples = rng.multivariate_normal(self.mean, self.cov, size=(n_scenarios * n_steps))
        samples = samples.reshape(n_scenarios, n_steps, len(self.assets))
        return samples
