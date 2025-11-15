# JEPA-based World Models for Multi-Asset Risk Scenarios

## Project statement
We train a JEPA-style world model on a small multi-asset universe to learn market regimes and correlation breakdowns. The model will be benchmarked against standard risk engines to understand when representation learning improves portfolio stress testing.

## Universe (daily data)
- SPX (US equity index)
- NDX (US tech index)
- EuroStoxx 50
- Nikkei 225
- US 10Y yield
- German 10Y yield
- EURUSD
- USDJPY
- VIX
- Gold

All series are daily and aligned on a common trading calendar before feature engineering.

## Data schema (daily)
The canonical table will contain one row per asset per date with the columns `date`, `asset`, `close_price`, `log_return_1d`, and `realized_vol_20d`. Additional derived features will remain synchronized across all assets thanks to the shared calendar alignment.

## Target tensor shape for modeling
For a sliding window of length L (e.g., 60 trading days), the model ingests tensors shaped as `[num_windows, L, n_assets, n_features]` where `n_assets = 10` and `n_features = 3` (log_return_1d, realized_vol_20d, plus capacity for a future feature such as macro surprises or liquidity metrics).

## Planned experiments
**A. Regime probing.** Train JEPA and baseline representational models (PCA, masked Transformer) and evaluate linear probes that discriminate crises vs. normal markets and risk-on vs. risk-off regimes.

**B. Scenario generation.** Generate multi-asset paths with Gaussian and VAR baselines alongside the JEPA-based world model; compare marginal distributions, correlation structures, and tail co-movements.

**C. Portfolio risk demo.** Stress-test a representative 60/40-style portfolio using scenarios from each model and contrast VaR/ES and drawdown profiles against historical crisis episodes.
