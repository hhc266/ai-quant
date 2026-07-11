"""Double moving-average strategy and backtest utilities for TASK3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    short_window: int = 5
    long_window: int = 15
    initial_capital: float = 100_000.0
    fee_rate: float = 0.0003
    slippage_rate: float = 0.0002
    lot_size: int = 100


def add_moving_averages(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving-average windows must be positive.")
    if short_window >= long_window:
        raise ValueError("The short SMA window must be smaller than the long SMA window.")

    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result = result.sort_values("trade_date").reset_index(drop=True)
    result["sma_short"] = result["close"].rolling(short_window, min_periods=short_window).mean()
    result["sma_long"] = result["close"].rolling(long_window, min_periods=long_window).mean()
    return result


def add_trade_signals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    previous_short = result["sma_short"].shift(1)
    previous_long = result["sma_long"].shift(1)
    current_short = result["sma_short"]
    current_long = result["sma_long"]

    golden_cross = (previous_short <= previous_long) & (current_short > current_long)
    death_cross = (previous_short >= previous_long) & (current_short < current_long)

    result["signal"] = 0
    result.loc[golden_cross, "signal"] = 1
    result.loc[death_cross, "signal"] = -1
    result["signal_label"] = ""
    result.loc[result["signal"] == 1, "signal_label"] = "金叉买入"
    result.loc[result["signal"] == -1, "signal_label"] = "死叉卖出"
    result["execution_signal"] = result["signal"].shift(1).fillna(0).astype(int)
    return result


def prepare_strategy_frame(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    return add_trade_signals(add_moving_averages(frame, short_window, long_window))


def run_backtest(frame: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    data = prepare_strategy_frame(frame, config.short_window, config.long_window)
    cash = float(config.initial_capital)
    shares = 0
    trade_rows = []
    equity_rows = []

    for row in data.itertuples(index=False):
        date = str(row.trade_date)
        execution_signal = int(row.execution_signal)
        open_price = float(row.open)
        close_price = float(row.close)

        if execution_signal == 1 and shares == 0:
            buy_price = open_price * (1 + config.slippage_rate)
            max_shares = math.floor(cash / (buy_price * (1 + config.fee_rate)))
            if config.lot_size > 1:
                max_shares = (max_shares // config.lot_size) * config.lot_size
            if max_shares > 0:
                gross = max_shares * buy_price
                fee = gross * config.fee_rate
                cash -= gross + fee
                shares = max_shares
                trade_rows.append(
                    {
                        "trade_date": date,
                        "action": "BUY",
                        "price": round(buy_price, 4),
                        "shares": shares,
                        "fee": round(fee, 4),
                        "cash_after": round(cash, 4),
                    }
                )

        elif execution_signal == -1 and shares > 0:
            sell_price = open_price * (1 - config.slippage_rate)
            gross = shares * sell_price
            fee = gross * config.fee_rate
            cash += gross - fee
            trade_rows.append(
                {
                    "trade_date": date,
                    "action": "SELL",
                    "price": round(sell_price, 4),
                    "shares": shares,
                    "fee": round(fee, 4),
                    "cash_after": round(cash, 4),
                }
            )
            shares = 0

        market_value = shares * close_price
        total_equity = cash + market_value
        equity_rows.append(
            {
                "trade_date": date,
                "cash": cash,
                "shares": shares,
                "market_value": market_value,
                "equity": total_equity,
                "close": close_price,
                "signal": int(row.signal),
                "execution_signal": execution_signal,
                "sma_short": row.sma_short,
                "sma_long": row.sma_long,
            }
        )

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(equity, trades, config.initial_capital)
    return equity, trades, metrics


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if equity.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trade_count": 0.0,
            "final_equity": initial_capital,
            "win_rate": 0.0,
            "holding_ratio": 0.0,
            "buy_hold_return": 0.0,
        }

    values = equity["equity"].astype(float)
    daily_returns = values.pct_change().fillna(0.0)
    running_max = values.cummax()
    drawdowns = values / running_max - 1
    cumulative_return = values.iloc[-1] / initial_capital - 1
    periods = max(len(equity), 1)
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / periods) - 1 if cumulative_return > -1 else -1
    std = daily_returns.std(ddof=0)
    sharpe = daily_returns.mean() / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    buy_hold_return = equity["close"].iloc[-1] / equity["close"].iloc[0] - 1
    round_trips = _round_trip_returns(trades)

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(drawdowns.min()),
        "sharpe_ratio": float(sharpe),
        "trade_count": float(len(trades)),
        "final_equity": float(values.iloc[-1]),
        "win_rate": float(np.mean([item > 0 for item in round_trips])) if round_trips else 0.0,
        "holding_ratio": float((equity["shares"] > 0).mean()),
        "buy_hold_return": float(buy_hold_return),
    }


def add_drawdown_columns(equity: pd.DataFrame) -> pd.DataFrame:
    result = equity.copy()
    result["strategy_return"] = result["equity"].pct_change().fillna(0.0)
    result["strategy_nav"] = result["equity"] / result["equity"].iloc[0]
    result["benchmark_nav"] = result["close"] / result["close"].iloc[0]
    result["running_max"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["running_max"] - 1
    return result


def _round_trip_returns(trades: pd.DataFrame) -> list[float]:
    returns: list[float] = []
    entry_price = None
    for row in trades.itertuples(index=False):
        if row.action == "BUY":
            entry_price = float(row.price)
        elif row.action == "SELL" and entry_price:
            returns.append(float(row.price) / entry_price - 1)
            entry_price = None
    return returns
