"""
performance_metrics.py
──────────────────────
Standalone performance and risk metric calculations.

All functions are pure — they take a pandas Series and return
numbers or dicts. No MT5 or Telegram dependencies.

Usage:
    from performance_metrics import compute_metrics, format_report
    metrics = compute_metrics(strategy_returns, cumulative_returns)
    print(format_report(metrics, symbol="XAUUSD", ema_short=9, ema_long=21))
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  CORE METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

def total_return(cum_returns: pd.Series) -> float:
    """Total percentage return over the period."""
    return round((cum_returns.iloc[-1] - 1) * 100, 2)


def cagr(cum_returns: pd.Series, bars_per_year: int = 252) -> float:
    """Compound Annual Growth Rate (%)."""
    n_years = len(cum_returns) / bars_per_year
    rate    = (cum_returns.iloc[-1] ** (1 / max(n_years, 1e-6))) - 1
    return round(rate * 100, 2)


def annualised_volatility(returns: pd.Series, bars_per_year: int = 252) -> float:
    """Annualised standard deviation of log returns (%)."""
    return round(returns.std() * np.sqrt(bars_per_year) * 100, 2)


def sharpe_ratio(returns: pd.Series,
                 risk_free_rate: float = 0.02,
                 bars_per_year: int = 252) -> float:
    """
    Annualised Sharpe Ratio.
    Higher is better. Benchmark: > 1.0 acceptable, > 2.0 excellent.
    """
    daily_rf = risk_free_rate / bars_per_year
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return round((excess.mean() / excess.std()) * np.sqrt(bars_per_year), 3)


def sortino_ratio(returns: pd.Series,
                  risk_free_rate: float = 0.02,
                  bars_per_year: int = 252) -> float:
    """
    Annualised Sortino Ratio (only penalises downside volatility).
    Higher than Sharpe = most volatility is upside (good sign).
    """
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std() * np.sqrt(bars_per_year)
    if downside_std == 0:
        return 0.0
    return round((returns.mean() * bars_per_year - risk_free_rate) / downside_std, 3)


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (%).
    Returns a negative number. Closer to 0 is better.
    """
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return round(drawdown.min() * 100, 2)


def calmar_ratio(cum_returns: pd.Series, bars_per_year: int = 252) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.
    Measures return earned per unit of drawdown pain.
    Higher is better. > 1.0 good, > 2.0 excellent.
    """
    _cagr = cagr(cum_returns, bars_per_year)
    _mdd  = max_drawdown(cum_returns)
    if _mdd == 0:
        return 0.0
    return round(_cagr / abs(_mdd), 3)


def win_rate(returns: pd.Series) -> float:
    """Percentage of non-zero return bars that are positive."""
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    wins = trades[trades > 0]
    return round(len(wins) / len(trades) * 100, 2)


def profit_factor(returns: pd.Series) -> float:
    """
    Gross profit / Gross loss.
    > 1.0 = profitable. > 1.5 good. > 2.0 excellent.
    """
    trades = returns[returns != 0]
    wins   = trades[trades > 0]
    losses = trades[trades < 0]
    if len(losses) == 0 or losses.sum() == 0:
        return np.nan
    return round(wins.sum() / abs(losses.sum()), 3)


def drawdown_series(cum_returns: pd.Series) -> pd.Series:
    """Return the full drawdown time series (values are negative %)."""
    roll_max = cum_returns.cummax()
    return (cum_returns - roll_max) / roll_max * 100


# ═══════════════════════════════════════════════════════════════════
#  COMPOSITE METRICS DICT
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(strategy_returns: pd.Series,
                    cum_returns: pd.Series,
                    risk_free_rate: float = 0.02,
                    bars_per_year: int = 252) -> dict:
    """
    Compute the full suite of performance metrics.

    Parameters
    ----------
    strategy_returns : pd.Series  — per-bar log returns of the strategy
    cum_returns      : pd.Series  — cumulative multiplier (starts near 1.0)
    risk_free_rate   : float      — annual risk-free rate (default 2%)
    bars_per_year    : int        — trading days per year (default 252)

    Returns
    -------
    dict — all metrics keyed by display name
    """
    strat  = strategy_returns.dropna()
    trades = strat[strat != 0]

    return {
        "Total Return (%)":     total_return(cum_returns),
        "CAGR (%)":             cagr(cum_returns, bars_per_year),
        "Ann. Volatility (%)":  annualised_volatility(strat, bars_per_year),
        "Sharpe Ratio":         sharpe_ratio(strat, risk_free_rate, bars_per_year),
        "Sortino Ratio":        sortino_ratio(strat, risk_free_rate, bars_per_year),
        "Calmar Ratio":         calmar_ratio(cum_returns, bars_per_year),
        "Max Drawdown (%)":     max_drawdown(cum_returns),
        "Win Rate (%)":         win_rate(strat),
        "Profit Factor":        profit_factor(strat),
        "Total Trades":         len(trades),
        "Avg Trade (log)":      round(trades.mean(), 6) if len(trades) > 0 else 0,
        "Best Trade (log)":     round(trades.max(), 6)  if len(trades) > 0 else 0,
        "Worst Trade (log)":    round(trades.min(), 6)  if len(trades) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  FORMATTED REPORT
# ═══════════════════════════════════════════════════════════════════

def format_report(metrics: dict,
                  symbol: str = "",
                  ema_short: int = 0,
                  ema_long: int = 0) -> str:
    """
    Return a plain-text report suitable for printing or Telegram.

    Parameters
    ----------
    metrics   : dict  — output of compute_metrics()
    symbol    : str   — trading symbol shown in header
    ema_short : int   — short EMA period shown in header
    ema_long  : int   — long EMA period shown in header
    """
    rating = {
        "Sharpe Ratio":    lambda v: "⭐⭐⭐" if v >= 2 else ("⭐⭐" if v >= 1 else "⭐"),
        "Sortino Ratio":   lambda v: "⭐⭐⭐" if v >= 2 else ("⭐⭐" if v >= 1 else "⭐"),
        "Calmar Ratio":    lambda v: "⭐⭐⭐" if v >= 2 else ("⭐⭐" if v >= 1 else "⭐"),
        "Max Drawdown (%)": lambda v: "🟢" if v > -10 else ("🟡" if v > -20 else "🔴"),
        "Profit Factor":   lambda v: "⭐⭐" if v >= 1.5 else ("⭐" if v >= 1.0 else "❌"),
    }

    lines = [
        "=" * 48,
        f"  📊 PERFORMANCE REPORT",
        f"  {symbol}  |  EMA {ema_short}/{ema_long}",
        "=" * 48,
    ]
    for k, v in metrics.items():
        badge = rating.get(k, lambda _: "")(v) if isinstance(v, (int, float)) else ""
        lines.append(f"  {k:<25}  {v}  {badge}")
    lines.append("=" * 48)
    return "\n".join(lines)


def format_telegram_report(metrics: dict,
                            symbol: str = "",
                            ema_short: int = 0,
                            ema_long: int = 0) -> str:
    """
    Return a Markdown-formatted report for Telegram messages.
    Uses backtick code spans for values.
    """
    lines = [
        f"📊 *BACKTEST REPORT*",
        f"Symbol: `{symbol}` | EMA `{ema_short}/{ema_long}`",
        "─" * 32,
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: `{v}`")
    return "\n".join(lines)
