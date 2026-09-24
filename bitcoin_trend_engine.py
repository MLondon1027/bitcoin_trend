from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendComparisonResult:
    equity: pd.DataFrame
    drawdowns: pd.DataFrame
    weights: pd.DataFrame
    daily_returns: pd.DataFrame
    annual_returns: pd.DataFrame
    metrics: pd.DataFrame


def _clean_series(values: pd.Series, name: str) -> pd.Series:
    series = pd.to_numeric(values.copy(), errors="coerce").dropna().sort_index()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.loc[~series.index.duplicated(keep="last")]
    series.name = name
    if len(series) < 250:
        raise ValueError(f"{name} requires at least 250 daily observations.")
    return series


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity.div(equity.cummax()).sub(1.0)


def _complete_year_returns(returns: pd.Series) -> pd.Series:
    rows: dict[int, float] = {}
    for year, group in returns.groupby(returns.index.year):
        if len(group) >= 350:
            rows[int(year)] = float((1.0 + group).prod() - 1.0)
    return pd.Series(rows, dtype=float)


def _metric_row(
    returns: pd.Series,
    weights: pd.Series,
    starting_capital: float,
) -> dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    elapsed_years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    complete_years = _complete_year_returns(returns)
    entries = int(((weights > 0.01) & (weights.shift(1).fillna(0.0) <= 0.01)).sum())
    standard_deviation = returns.std(ddof=1)
    return {
        "Ending Value": starting_capital * float(wealth.iloc[-1]),
        "Total Return": float(wealth.iloc[-1] - 1.0),
        "CAGR": float(wealth.iloc[-1] ** (1.0 / elapsed_years) - 1.0),
        "Annualized Volatility": float(standard_deviation * np.sqrt(365)),
        "Sharpe (0% risk-free)": (
            float(returns.mean() / standard_deviation * np.sqrt(365))
            if standard_deviation > 0 else np.nan
        ),
        "Maximum Drawdown": float(_drawdown(wealth).min()),
        "Worst Complete Year": (
            float(complete_years.min()) if not complete_years.empty else np.nan
        ),
        "Losing Complete Years": float((complete_years < 0).sum()),
        "Positive Complete Years": float((complete_years > 0).sum()),
        "BTC Entries": float(entries),
        "Time With BTC Exposure": float((weights > 0.01).mean()),
        "Average BTC Allocation": float(weights.mean()),
    }


def run_trend_comparison(
    btc_close: pd.Series,
    cash_close: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    starting_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
    fast_ema_days: int = 50,
    slow_ema_days: int = 200,
    slope_lookback_days: int = 20,
    volatility_lookback_days: int = 30,
    target_volatility_pct: float = 30.0,
    maximum_btc_allocation_pct: float = 100.0,
) -> TrendComparisonResult:
    """Compare fixed Bitcoin trend/cash rules without look-ahead bias.

    Signals observed at a daily close determine the allocation applied to the
    following close-to-close return. The risk-off asset is an investable
    short-term Treasury ETF or another supplied cash proxy.
    """
    if starting_capital <= 0:
        raise ValueError("Starting capital must be positive.")
    if not 2 <= fast_ema_days < slow_ema_days:
        raise ValueError("Fast EMA must be at least 2 days and shorter than the slow EMA.")
    if slope_lookback_days < 1 or volatility_lookback_days < 2:
        raise ValueError("Lookback periods are too short.")
    if target_volatility_pct <= 0 or not 0 < maximum_btc_allocation_pct <= 100:
        raise ValueError("Volatility target and maximum allocation must be positive.")

    btc = _clean_series(btc_close, "BTC")
    cash_raw = _clean_series(cash_close, "Cash")
    cash = cash_raw.reindex(btc.index).ffill().bfill()
    btc_returns = btc.pct_change(fill_method=None).fillna(0.0)
    cash_returns = cash.pct_change(fill_method=None).fillna(0.0)

    sma_slow = btc.rolling(slow_ema_days).mean()
    ema_fast = btc.ewm(span=fast_ema_days, adjust=False).mean()
    ema_slow = btc.ewm(span=slow_ema_days, adjust=False).mean()
    rising_slow = ema_slow > ema_slow.shift(slope_lookback_days)

    simple_signal = (btc > sma_slow).astype(float)
    dual_signal = (ema_fast > ema_slow).astype(float)
    strict_signal = ((btc > ema_slow) & (ema_fast > ema_slow) & rising_slow).astype(float)

    realized_volatility = (
        btc_returns.rolling(volatility_lookback_days).std(ddof=1) * np.sqrt(365)
    )
    volatility_scalar = (target_volatility_pct / 100.0) / realized_volatility
    maximum_weight = maximum_btc_allocation_pct / 100.0
    volatility_weight = strict_signal * volatility_scalar.clip(0.0, maximum_weight)

    signal_weights = pd.DataFrame(
        {
            "Buy & Hold BTC": 1.0,
            f"Price > {slow_ema_days}D SMA": simple_signal,
            f"{fast_ema_days}/{slow_ema_days} EMA": dual_signal,
            "Strict Trend": strict_signal,
            "Vol-Targeted Strict Trend": volatility_weight,
        },
        index=btc.index,
    )
    applied_weights = signal_weights.shift(1).fillna(0.0)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    period_index = btc.index[(btc.index >= start) & (btc.index <= end)]
    if len(period_index) < 2:
        raise ValueError("No usable Bitcoin observations exist in the selected period.")

    applied_weights = applied_weights.loc[period_index]
    btc_period_returns = btc_returns.loc[period_index]
    cash_period_returns = cash_returns.loc[period_index]
    cost_fraction = transaction_cost_bps / 10_000.0

    strategy_returns: dict[str, pd.Series] = {}
    for name in applied_weights.columns:
        weight = applied_weights[name]
        gross = weight * btc_period_returns + (1.0 - weight) * cash_period_returns
        turnover = weight.diff().abs()
        turnover.iloc[0] = weight.iloc[0]
        net = gross - turnover * cost_fraction
        # The backtest starts at the first selected close; only initial trading
        # cost is charged on that date, not the preceding day's market move.
        net.iloc[0] = -weight.iloc[0] * cost_fraction
        strategy_returns[name] = net

    daily_returns = pd.DataFrame(strategy_returns)
    equity = (1.0 + daily_returns).cumprod() * starting_capital
    drawdowns = equity.apply(_drawdown)
    annual_returns = daily_returns.resample("YE").apply(
        lambda values: (1.0 + values).prod() - 1.0
    )
    annual_returns.index = annual_returns.index.year
    annual_returns.index.name = "Year"
    metrics = pd.DataFrame(
        {
            name: _metric_row(daily_returns[name], applied_weights[name], starting_capital)
            for name in daily_returns.columns
        }
    ).T

    return TrendComparisonResult(
        equity=equity,
        drawdowns=drawdowns,
        weights=applied_weights,
        daily_returns=daily_returns,
        annual_returns=annual_returns,
        metrics=metrics,
    )


def strict_trend_parameter_stability(
    btc_close: pd.Series,
    cash_close: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    starting_capital: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Test nearby fixed EMA settings to expose parameter fragility."""
    rows: list[dict[str, float]] = []
    for fast in (30, 50, 75):
        for slow in (150, 175, 200, 225, 250):
            if fast >= slow:
                continue
            result = run_trend_comparison(
                btc_close,
                cash_close,
                start_date,
                end_date,
                starting_capital=starting_capital,
                transaction_cost_bps=transaction_cost_bps,
                fast_ema_days=fast,
                slow_ema_days=slow,
            )
            metric = result.metrics.loc["Strict Trend"]
            rows.append(
                {
                    "Fast EMA": float(fast),
                    "Slow EMA": float(slow),
                    "CAGR": float(metric["CAGR"]),
                    "Maximum Drawdown": float(metric["Maximum Drawdown"]),
                    "Losing Complete Years": float(metric["Losing Complete Years"]),
                    "Worst Complete Year": float(metric["Worst Complete Year"]),
                    "BTC Entries": float(metric["BTC Entries"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Losing Complete Years", "Maximum Drawdown", "CAGR"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
