"""Market data loader for TASK3.

The preferred source is Tushare when a token is available. The
public dashboard is built from cached CSV/JS files so no secret token is exposed
after generation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class StockSpec:
    ts_code: str
    name: str
    eastmoney_secid: str
    yahoo_symbol: str


STOCKS: tuple[StockSpec, ...] = (
    StockSpec("688981.SH", "中芯国际", "1.688981", "688981.SS"),
    StockSpec("002594.SZ", "比亚迪", "0.002594", "002594.SZ"),
    StockSpec("600900.SH", "长江电力", "1.600900", "600900.SS"),
    StockSpec("601318.SH", "中国平安", "1.601318", "601318.SS"),
    StockSpec("600519.SH", "贵州茅台", "1.600519", "600519.SS"),
)


def stock_map() -> dict[str, StockSpec]:
    return {stock.ts_code: stock for stock in STOCKS}


def load_market_data(
    stock: StockSpec,
    start_date: str,
    end_date: str,
    prefer_tushare: bool = True,
) -> pd.DataFrame:
    """Load daily market data and return a Tushare-like dataframe.

    Dates are expected in YYYYMMDD format. If Tushare is not configured, the
    loader falls back to Eastmoney's public historical k-line endpoint and marks
    the source column accordingly.
    """

    if prefer_tushare:
        tushare_frame = _try_load_tushare(stock.ts_code, start_date, end_date)
        if tushare_frame is not None and not tushare_frame.empty:
            return _normalize_frame(tushare_frame, stock.ts_code, "tushare_qfq")

    try:
        eastmoney_frame = _load_eastmoney(stock, start_date, end_date)
        return _normalize_frame(eastmoney_frame, stock.ts_code, "eastmoney")
    except Exception:
        yahoo_frame = _load_yahoo(stock, start_date, end_date)
        return _normalize_frame(yahoo_frame, stock.ts_code, "yahoo")


def load_all_market_data(
    output_dir: Path,
    start_date: str,
    end_date: str,
    stocks: Iterable[StockSpec] = STOCKS,
    prefer_tushare: bool = True,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for stock in stocks:
        frame = load_market_data(stock, start_date, end_date, prefer_tushare)
        frames[stock.ts_code] = frame
        out_path = output_dir / f"{stock.ts_code.replace('.', '_')}_daily.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return frames


def _try_load_tushare(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TS_TOKEN")
    if not token:
        return None

    try:
        params = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
        }
        daily = _request_tushare_api(
            "daily",
            token,
            params,
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
        if daily.empty:
            raise RuntimeError(f"Tushare returned no data for {ts_code}")
        return _forward_adjust_from_daily_returns(daily)
    except Exception:
        if os.environ.get("TUSHARE_STRICT") == "1":
            raise
        return None


def _request_tushare_api(
    api_name: str,
    token: str,
    params: dict[str, str],
    fields: str,
) -> pd.DataFrame:
    payload = {
        "api_name": api_name,
        "token": token,
        "params": params,
        "fields": fields,
    }
    request = Request(
        "https://api.tushare.pro",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "TASK3-double-ma-workshop/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=45) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("code") != 0:
        raise RuntimeError(result.get("msg") or f"Tushare {api_name} request failed")

    data = result.get("data") or {}
    response_fields = data.get("fields") or []
    items = data.get("items") or []
    return pd.DataFrame(items, columns=response_fields)


def _forward_adjust_from_daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a continuous forward-adjusted series from Tushare daily returns.

    Tushare's pct_chg uses the exchange's ex-right reference price. Chaining
    those returns removes split and dividend price gaps without another API
    call, then scales the reconstructed series to the latest raw close.
    """

    adjusted = frame.copy().sort_values("trade_date").reset_index(drop=True)
    raw_close = pd.to_numeric(adjusted["close"], errors="coerce")
    raw_pre_close = pd.to_numeric(adjusted["pre_close"], errors="coerce")
    daily_return = pd.to_numeric(adjusted["pct_chg"], errors="coerce").fillna(0.0) / 100
    wealth = (1 + daily_return).cumprod()
    adjusted_close = wealth / float(wealth.iloc[-1]) * float(raw_close.iloc[-1])
    price_ratio = adjusted_close / raw_close

    for column in ["open", "high", "low", "close"]:
        adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") * price_ratio
    adjusted["pre_close"] = adjusted["close"].shift(1)
    adjusted.loc[0, "pre_close"] = raw_pre_close.iloc[0] * price_ratio.iloc[0]
    adjusted["change"] = adjusted["close"] - adjusted["pre_close"]
    adjusted["pct_chg"] = adjusted["change"] / adjusted["pre_close"] * 100
    return adjusted


def _load_eastmoney(stock: StockSpec, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "secid": stock.eastmoney_secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start_date,
        "end": end_date,
        "lmt": "1000000",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    request = Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://quote.eastmoney.com/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Eastmoney request failed for {stock.ts_code}: {last_error}") from last_error

    if payload.get("rc") != 0 or not payload.get("data") or not payload["data"].get("klines"):
        raise RuntimeError(f"Eastmoney returned no data for {stock.ts_code}")

    rows = []
    for item in payload["data"]["klines"]:
        values = item.split(",")
        rows.append(
            {
                "trade_date": values[0].replace("-", ""),
                "open": float(values[1]),
                "close": float(values[2]),
                "high": float(values[3]),
                "low": float(values[4]),
                "vol": float(values[5]),
                "amount": float(values[6]),
                "pct_chg": float(values[8]) if values[8] else 0.0,
                "change": float(values[9]) if values[9] else 0.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame["pre_close"] = frame["close"].shift(1)
    if not frame.empty:
        frame.loc[frame.index[0], "pre_close"] = frame.loc[frame.index[0], "open"]
    return frame


def _load_yahoo(stock: StockSpec, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = _date_to_unix(start_date)
    end_ts = _date_to_unix(end_date) + 24 * 60 * 60
    params = {
        "period1": str(start_ts),
        "period2": str(end_ts),
        "interval": "1d",
        "events": "history",
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock.yahoo_symbol}?" + urlencode(params)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {stock.ts_code}")

    timestamps = result[0].get("timestamp") or []
    quote = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
    rows = []
    for idx, stamp in enumerate(timestamps):
        open_price = _optional_float(quote.get("open", [None])[idx])
        high = _optional_float(quote.get("high", [None])[idx])
        low = _optional_float(quote.get("low", [None])[idx])
        close = _optional_float(quote.get("close", [None])[idx])
        volume = _optional_float(quote.get("volume", [None])[idx], default=0.0)
        if open_price is None or high is None or low is None or close is None:
            continue
        rows.append(
            {
                "trade_date": datetime.fromtimestamp(int(stamp), tz=timezone.utc).strftime("%Y%m%d"),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "vol": volume,
                "amount": volume * close,
            }
        )

    frame = pd.DataFrame(rows)
    frame["pre_close"] = frame["close"].shift(1)
    if not frame.empty:
        frame.loc[frame.index[0], "pre_close"] = frame.loc[frame.index[0], "open"]
    frame["change"] = frame["close"] - frame["pre_close"]
    frame["pct_chg"] = frame["change"] / frame["pre_close"] * 100
    return frame


def _normalize_frame(frame: pd.DataFrame, ts_code: str, source: str) -> pd.DataFrame:
    required = ["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    normalized = frame.copy()
    normalized["ts_code"] = ts_code
    normalized["source"] = source
    normalized["trade_date"] = normalized["trade_date"].astype(str)
    for col in required[1:]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized[["ts_code", *required, "source"]]
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized.sort_values("trade_date").reset_index(drop=True)
    return normalized


def _date_to_unix(value: str) -> int:
    return int(datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
