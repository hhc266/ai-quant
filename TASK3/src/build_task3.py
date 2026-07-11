"""Build all TASK3 data artifacts and dashboard payloads."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_loader import STOCKS, StockSpec, load_all_market_data
from double_ma_strategy import BacktestConfig, add_drawdown_columns, prepare_strategy_frame, run_backtest


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "outputs" / "metrics"
TRADES_DIR = ROOT / "outputs" / "trades"
FIGURES_DIR = ROOT / "outputs" / "figures"
PAGES_DIR = ROOT

START_DATE = "20250101"
END_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_CONFIG = BacktestConfig(short_window=5, long_window=15, initial_capital=100_000, fee_rate=0.0003, slippage_rate=0.0002)


def main() -> None:
    _ensure_dirs()
    frames = load_all_market_data(RAW_DIR, START_DATE, END_DATE, STOCKS, prefer_tushare=True)
    dashboard_payload = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dateRange": {"start": _format_date(START_DATE), "end": _format_date(END_DATE)},
        "defaultConfig": {
            "shortWindow": DEFAULT_CONFIG.short_window,
            "longWindow": DEFAULT_CONFIG.long_window,
            "initialCapital": DEFAULT_CONFIG.initial_capital,
            "feeRate": DEFAULT_CONFIG.fee_rate,
            "slippageRate": DEFAULT_CONFIG.slippage_rate,
        },
        "stocks": [],
    }

    summary_rows = []
    first_stock_outputs = None

    for stock in STOCKS:
        source = str(frames[stock.ts_code]["source"].iloc[0])
        processed = prepare_strategy_frame(frames[stock.ts_code], DEFAULT_CONFIG.short_window, DEFAULT_CONFIG.long_window)
        equity, trades, metrics = run_backtest(frames[stock.ts_code], DEFAULT_CONFIG)
        equity = add_drawdown_columns(equity)

        processed_path = PROCESSED_DIR / f"{stock.ts_code.replace('.', '_')}_sma_signals.csv"
        equity_path = PROCESSED_DIR / f"{stock.ts_code.replace('.', '_')}_equity_curve.csv"
        trades_path = TRADES_DIR / f"{stock.ts_code.replace('.', '_')}_trades.csv"
        metrics_path = METRICS_DIR / f"{stock.ts_code.replace('.', '_')}_metrics.json"

        processed.to_csv(processed_path, index=False, encoding="utf-8-sig")
        equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        _write_json(metrics_path, {"stock": stock.__dict__, "dataSource": source, "defaultConfig": DEFAULT_CONFIG.__dict__, "metrics": metrics})

        summary_rows.append(
            {
                "ts_code": stock.ts_code,
                "name": stock.name,
                "source": source,
                **metrics,
            }
        )
        dashboard_payload["stocks"].append(_dashboard_stock_payload(stock, frames[stock.ts_code], source, metrics))

        if first_stock_outputs is None:
            first_stock_outputs = (stock, processed, equity, trades, metrics)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(METRICS_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    _write_json(METRICS_DIR / "metrics_summary.json", summary_rows)
    _write_dashboard_payload(dashboard_payload)

    if first_stock_outputs:
        _write_default_figures(*first_stock_outputs)


def _ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, METRICS_DIR, TRADES_DIR, FIGURES_DIR, PAGES_DIR / "assets"]:
        path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_dashboard_payload(payload: dict) -> None:
    data_text = "window.TASK3_STOCK_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    target = PAGES_DIR / "assets" / "dashboard-data.js"
    target.write_text(data_text, encoding="utf-8")


def _dashboard_stock_payload(stock: StockSpec, frame: pd.DataFrame, source: str, metrics: dict[str, float]) -> dict:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "date": _format_date(str(row.trade_date)),
                "tradeDate": str(row.trade_date),
                "open": _round(float(row.open)),
                "high": _round(float(row.high)),
                "low": _round(float(row.low)),
                "close": _round(float(row.close)),
                "volume": _round(float(row.vol), 2),
                "amount": _round(float(row.amount), 2),
            }
        )
    return {
        "tsCode": stock.ts_code,
        "name": stock.name,
        "source": source,
        "defaultMetrics": {key: _round(value, 6) for key, value in metrics.items()},
        "rows": rows,
    }


def _write_default_figures(
    stock: StockSpec,
    processed: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    compact_processed = processed.dropna(subset=["sma_short", "sma_long"]).reset_index(drop=True)
    _write_price_signal_svg(
        FIGURES_DIR / "fig1_price_sma_signals.svg",
        f"图 1：{stock.name} 股价与双均线交易信号",
        compact_processed,
    )
    _write_line_svg(
        FIGURES_DIR / "fig2_strategy_nav.svg",
        f"图 2：{stock.name} 策略净值曲线",
        equity,
        [("strategy_nav", "策略净值"), ("benchmark_nav", "买入持有")],
        "净值",
    )
    _write_line_svg(
        FIGURES_DIR / "fig3_drawdown.svg",
        f"图 3：{stock.name} 策略回撤曲线",
        equity,
        [("drawdown", "回撤")],
        "回撤",
        percent=True,
    )
    _write_metrics_svg(FIGURES_DIR / "fig4_metrics_cards.svg", f"图 4：{stock.name} 核心指标汇总", metrics)


def _write_price_signal_svg(path: Path, title: str, frame: pd.DataFrame) -> None:
    width, height = 1000, 520
    margin = {"left": 70, "right": 30, "top": 70, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    values = pd.concat([frame["close"], frame["sma_short"], frame["sma_long"]]).dropna()
    y_min, y_max = values.min(), values.max()
    x = _x_scale(len(frame), margin["left"], plot_w)
    y = _y_scale(y_min, y_max, margin["top"], plot_h)
    close_points = _polyline(frame["close"], x, y)
    short_points = _polyline(frame["sma_short"], x, y)
    long_points = _polyline(frame["sma_long"], x, y)
    marks = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        if row["signal"] == 1:
            marks.append(f'<circle cx="{x(idx):.2f}" cy="{y(row["close"]):.2f}" r="5" fill="#2f9e44"><title>买入 {row["trade_date"]}</title></circle>')
        elif row["signal"] == -1:
            marks.append(f'<path d="M {x(idx)-5:.2f} {y(row["close"])-5:.2f} L {x(idx)+5:.2f} {y(row["close"])+5:.2f} M {x(idx)+5:.2f} {y(row["close"])-5:.2f} L {x(idx)-5:.2f} {y(row["close"])+5:.2f}" stroke="#c92a2a" stroke-width="2"><title>卖出 {row["trade_date"]}</title></path>')
    svg = _svg_shell(
        width,
        height,
        title,
        f"""
        {_axes(margin, plot_w, plot_h, y_min, y_max)}
        <polyline points="{close_points}" fill="none" stroke="#343a40" stroke-width="2"/>
        <polyline points="{short_points}" fill="none" stroke="#1971c2" stroke-width="2"/>
        <polyline points="{long_points}" fill="none" stroke="#e67700" stroke-width="2"/>
        {''.join(marks)}
        {_legend([("收盘价", "#343a40"), ("短周期 SMA", "#1971c2"), ("长周期 SMA", "#e67700"), ("买入/卖出", "#2f9e44")], 720, 35)}
        <text x="70" y="500" font-size="15" fill="#495057">解读：短均线上穿长均线形成金叉，作为买入信号；短均线下穿长均线形成死叉，作为卖出信号。</text>
        """,
    )
    path.write_text(svg, encoding="utf-8")


def _write_line_svg(
    path: Path,
    title: str,
    frame: pd.DataFrame,
    columns: Iterable[tuple[str, str]],
    y_label: str,
    percent: bool = False,
) -> None:
    width, height = 1000, 520
    margin = {"left": 70, "right": 30, "top": 70, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    series = [(col, label, frame[col].astype(float)) for col, label in columns]
    values = pd.concat([item[2] for item in series]).dropna()
    y_min, y_max = values.min(), values.max()
    x = _x_scale(len(frame), margin["left"], plot_w)
    y = _y_scale(y_min, y_max, margin["top"], plot_h)
    colors = ["#1971c2", "#868e96", "#c92a2a"]
    lines = []
    legend_items = []
    for idx, (col, label, values_series) in enumerate(series):
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{_polyline(values_series, x, y)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_items.append((label, color))
    unit_text = "百分比" if percent else y_label
    interpretation = "解读：净值曲线越向上说明累计收益越高，可与买入持有基准比较策略择时效果。"
    if percent:
        interpretation = "解读：回撤越深代表从阶段高点下跌越多，最低点对应策略承受的最大资金压力。"
    svg = _svg_shell(
        width,
        height,
        title,
        f"""
        {_axes(margin, plot_w, plot_h, y_min, y_max, unit_text, percent)}
        {''.join(lines)}
        {_legend(legend_items, 760, 35)}
        <text x="70" y="500" font-size="15" fill="#495057">{interpretation}</text>
        """,
    )
    path.write_text(svg, encoding="utf-8")


def _write_metrics_svg(path: Path, title: str, metrics: dict[str, float]) -> None:
    width, height = 1000, 360
    items = [
        ("累计回报", _format_pct(metrics["cumulative_return"])),
        ("最大回撤", _format_pct(metrics["max_drawdown"])),
        ("夏普比率", f"{metrics['sharpe_ratio']:.2f}"),
        ("交易次数", f"{int(metrics['trade_count'])}"),
        ("期末资产", f"{metrics['final_equity']:,.0f} 元"),
    ]
    cards = []
    for idx, (label, value) in enumerate(items):
        x = 55 + idx * 185
        cards.append(
            f"""
            <rect x="{x}" y="110" width="160" height="100" rx="8" fill="#f8f9fa" stroke="#dee2e6"/>
            <text x="{x + 16}" y="145" font-size="15" fill="#495057">{label}</text>
            <text x="{x + 16}" y="185" font-size="24" font-weight="700" fill="#212529">{value}</text>
            """
        )
    svg = _svg_shell(
        width,
        height,
        title,
        f"""
        {''.join(cards)}
        <text x="55" y="285" font-size="15" fill="#495057">解读：指标卡片用于同时观察收益、风险和交易活跃度，需结合净值曲线与回撤曲线综合判断策略表现。</text>
        """,
    )
    path.write_text(svg, encoding="utf-8")


def _svg_shell(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{title}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="50" y="42" font-size="24" font-weight="700" fill="#212529">{title}</text>
  {body}
</svg>
"""


def _axes(margin: dict[str, int], plot_w: int, plot_h: int, y_min: float, y_max: float, unit: str = "价格", percent: bool = False) -> str:
    lines = [f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#dee2e6"/>']
    y = _y_scale(y_min, y_max, margin["top"], plot_h)
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        label = _format_pct(value) if percent else f"{value:.2f}"
        yy = y(value)
        lines.append(f'<line x1="{margin["left"]}" x2="{margin["left"] + plot_w}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#edf2f7"/>')
        lines.append(f'<text x="{margin["left"] - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" fill="#868e96">{label}</text>')
    lines.append(f'<text x="{margin["left"]}" y="{margin["top"] - 15}" font-size="12" fill="#868e96">{unit}</text>')
    return "\n".join(lines)


def _legend(items: Iterable[tuple[str, str]], x: int, y: int) -> str:
    pieces = []
    for idx, (label, color) in enumerate(items):
        xx = x + idx * 92
        pieces.append(f'<line x1="{xx}" y1="{y}" x2="{xx + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        pieces.append(f'<text x="{xx + 28}" y="{y + 5}" font-size="13" fill="#495057">{label}</text>')
    return "".join(pieces)


def _polyline(values: pd.Series, x, y) -> str:
    points = []
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        points.append(f"{x(idx):.2f},{y(float(value)):.2f}")
    return " ".join(points)


def _x_scale(count: int, left: int, width: int):
    span = max(count - 1, 1)
    return lambda idx: left + width * idx / span


def _y_scale(y_min: float, y_max: float, top: int, height: int):
    if y_max == y_min:
        y_max = y_min + 1
    pad = (y_max - y_min) * 0.08
    low = y_min - pad
    high = y_max + pad
    return lambda value: top + height - (value - low) / (high - low) * height


def _format_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


if __name__ == "__main__":
    main()
