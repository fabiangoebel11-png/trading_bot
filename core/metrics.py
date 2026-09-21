"""Performance metrics for equity/return series produced by the backtester."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualization_factor(periods_per_year: int) -> float:
    return np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * annualization_factor(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * annualization_factor(periods_per_year))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, cumulative_returns: pd.Series, periods_per_year: int) -> float:
    total_years = len(returns) / periods_per_year
    if total_years <= 0:
        return 0.0
    cagr = cumulative_returns.iloc[-1] ** (1 / total_years) - 1
    mdd = abs(max_drawdown(cumulative_returns))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def win_rate(trade_returns: pd.Series) -> float:
    trades = trade_returns[trade_returns != 0]
    if len(trades) == 0:
        return 0.0
    return float((trades > 0).mean())


def profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize_performance(
    returns: pd.Series, periods_per_year: int, trade_pnl: pd.Series | None = None
) -> dict:
    cumulative = (1 + returns).cumprod()
    summary = {
        "total_return_pct": float((cumulative.iloc[-1] - 1) * 100) if len(cumulative) else 0.0,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown_pct": max_drawdown(cumulative) * 100,
        "calmar": calmar_ratio(returns, cumulative, periods_per_year),
        "n_periods": int(len(returns)),
    }
    if trade_pnl is not None:
        summary["win_rate_pct"] = win_rate(trade_pnl) * 100
        summary["profit_factor"] = profit_factor(trade_pnl)
    return summary
