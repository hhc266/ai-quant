"""Turtle trading strategy utilities for TASK4.

The implementation is intentionally long-only and cash-only because the
workshop data set is made of A-share daily bars. It keeps the core Turtle
ideas: Donchian high/low channels, ATR volatility, and an ATR-based stop.
Signals are generated after the close and executed at the next trading day's
open to avoid look-ahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class TurtleConfig:
    entry_window: int = 20
    exit_window: int = 20
    atr_window: int = 20
    stop_atr_multiplier: float = 2.0
    initial_capital: float = 100_000.0
    fee_rate: float = 0.0003
    slippage_rate: float = 0.0002
    lot_size: int = 100


def prepare_turtle_frame(frame: pd.DataFrame, config: TurtleConfig) -> pd.DataFrame:
    """Add Donchian channels, true range, ATR, and stateless raw signals."""

    _validate_config(config)
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result = result.sort_values("trade_date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    previous_close = result["close"].shift(1)
    high_low = result["high"] - result["low"]
    high_prev_close = (result["high"] - previous_close).abs()
    low_prev_close = (result["low"] - previous_close).abs()
    result["true_range"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    result["atr"] = result["true_range"].rolling(config.atr_window, min_periods=config.atr_window).mean()
    result["upper_channel"] = (
        result["high"].rolling(config.entry_window, min_periods=config.entry_window).max().shift(1)
    )
    result["lower_channel"] = (
        result["low"].rolling(config.exit_window, min_periods=config.exit_window).min().shift(1)
    )

    result["raw_buy_signal"] = (result["close"] > result["upper_channel"]) & result["atr"].notna()
    result["raw_channel_exit_signal"] = result["close"] < result["lower_channel"]
    return result


def run_backtest(frame: pd.DataFrame, config: TurtleConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run a next-open execution backtest for the simplified Turtle system."""

    data = prepare_turtle_frame(frame, config)
    data["signal"] = 0
    data["signal_label"] = ""
    data["signal_reason"] = ""
    data["execution_signal"] = 0
    data["shares"] = 0
    data["position"] = 0
    data["entry_price"] = np.nan
    data["stop_price"] = np.nan

    cash = float(config.initial_capital)
    shares = 0
    entry_price: float | None = None
    stop_price: float | None = None
    pending_action: str | None = None
    pending_reason = ""
    pending_signal_date = ""
    pending_atr: float | None = None
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for idx, row in data.iterrows():
        date = str(row["trade_date"])
        open_price = float(row["open"])
        close_price = float(row["close"])
        execution_signal = 0

        if pending_action == "BUY" and shares == 0 and pending_atr and pending_atr > 0:
            buy_price = open_price * (1 + config.slippage_rate)
            max_shares = math.floor(cash / (buy_price * (1 + config.fee_rate)))
            if config.lot_size > 1:
                max_shares = (max_shares // config.lot_size) * config.lot_size
            if max_shares > 0:
                gross = max_shares * buy_price
                fee = gross * config.fee_rate
                cash -= gross + fee
                shares = max_shares
                entry_price = buy_price
                stop_price = buy_price - config.stop_atr_multiplier * float(pending_atr)
                execution_signal = 1
                trade_rows.append(
                    {
                        "signal_date": pending_signal_date,
                        "trade_date": date,
                        "action": "BUY",
                        "reason": pending_reason,
                        "price": round(buy_price, 4),
                        "shares": shares,
                        "fee": round(fee, 4),
                        "cash_after": round(cash, 4),
                        "atr_at_signal": round(float(pending_atr), 4),
                        "stop_price": round(float(stop_price), 4),
                    }
                )

        elif pending_action == "SELL" and shares > 0:
            sell_price = open_price * (1 - config.slippage_rate)
            gross = shares * sell_price
            fee = gross * config.fee_rate
            cash += gross - fee
            execution_signal = -1
            trade_rows.append(
                {
                    "signal_date": pending_signal_date,
                    "trade_date": date,
                    "action": "SELL",
                    "reason": pending_reason,
                    "price": round(sell_price, 4),
                    "shares": shares,
                    "fee": round(fee, 4),
                    "cash_after": round(cash, 4),
                    "atr_at_signal": round(float(row["atr"]), 4) if pd.notna(row["atr"]) else np.nan,
                    "stop_price": round(float(stop_price), 4) if stop_price is not None else np.nan,
                }
            )
            shares = 0
            entry_price = None
            stop_price = None

        pending_action = None
        pending_reason = ""
        pending_signal_date = ""
        pending_atr = None

        data.at[idx, "execution_signal"] = execution_signal
        data.at[idx, "shares"] = shares
        data.at[idx, "position"] = 1 if shares > 0 else 0
        if entry_price is not None:
            data.at[idx, "entry_price"] = entry_price
        if stop_price is not None:
            data.at[idx, "stop_price"] = stop_price

        if shares == 0 and bool(row["raw_buy_signal"]):
            data.at[idx, "signal"] = 1
            data.at[idx, "signal_label"] = "买入"
            data.at[idx, "signal_reason"] = "收盘价突破前期高点通道"
            pending_action = "BUY"
            pending_reason = "breakout"
            pending_signal_date = date
            pending_atr = float(row["atr"])
        elif shares > 0:
            stop_hit = stop_price is not None and close_price <= stop_price
            channel_exit = bool(row["raw_channel_exit_signal"])
            if stop_hit or channel_exit:
                reason = "ATR止损" if stop_hit else "跌破低点通道"
                if stop_hit and channel_exit:
                    reason = "ATR止损且跌破低点通道"
                data.at[idx, "signal"] = -1
                data.at[idx, "signal_label"] = "卖出"
                data.at[idx, "signal_reason"] = reason
                pending_action = "SELL"
                pending_reason = reason
                pending_signal_date = date

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
                "atr": row["atr"],
                "upper_channel": row["upper_channel"],
                "lower_channel": row["lower_channel"],
                "stop_price": stop_price if stop_price is not None else np.nan,
                "signal": int(data.at[idx, "signal"]),
                "execution_signal": execution_signal,
            }
        )

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    equity = add_performance_columns(equity, config.initial_capital)
    metrics = calculate_metrics(equity, trades, config.initial_capital)
    return data, equity, trades, metrics


def add_performance_columns(equity: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    result = equity.copy()
    if result.empty:
        return result
    result["strategy_return"] = result["equity"].pct_change().fillna(0.0)
    result["strategy_nav"] = result["equity"] / float(initial_capital)
    result["benchmark_nav"] = result["close"] / result["close"].iloc[0]
    result["running_max"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["running_max"] - 1
    return result


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if equity.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trade_count": 0.0,
            "round_trip_count": 0.0,
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
        "round_trip_count": float(len(round_trips)),
        "final_equity": float(values.iloc[-1]),
        "win_rate": float(np.mean([item > 0 for item in round_trips])) if round_trips else 0.0,
        "holding_ratio": float((equity["shares"] > 0).mean()),
        "buy_hold_return": float(buy_hold_return),
    }


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


def _validate_config(config: TurtleConfig) -> None:
    if config.entry_window <= 0 or config.exit_window <= 0 or config.atr_window <= 0:
        raise ValueError("Turtle windows must be positive integers.")
    if config.stop_atr_multiplier <= 0:
        raise ValueError("The ATR stop multiplier must be positive.")
    if config.initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
