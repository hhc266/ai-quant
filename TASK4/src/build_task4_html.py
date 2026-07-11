# -*- coding: utf-8 -*-
"""Build the TASK4 Turtle strategy HTML dashboard.

The page is standalone: all computed series and metrics are embedded so it can
be opened directly from disk without a local web server.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from turtle_strategy import TurtleConfig, run_backtest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RAW_DIR = ROOT / "data" / "raw"
SOURCE_RAW_DIR = REPO_ROOT / "TASK3" / "data" / "raw"
METRICS_DIR = ROOT / "outputs" / "metrics"
HTML_PATH = ROOT / "index.html"

STOCKS: tuple[tuple[str, str], ...] = (
    ("688981.SH", "中芯国际"),
    ("002594.SZ", "比亚迪"),
    ("600900.SH", "长江电力"),
    ("601318.SH", "中国平安"),
    ("600519.SH", "贵州茅台"),
)

WINDOWS = (10, 20, 55)
DEFAULT_WINDOW = 20
DEFAULT_STOCK = "688981.SH"
INITIAL_CAPITAL = 100_000.0


def main() -> None:
    payload = build_payload()
    HTML_PATH.write_text(build_html(payload), encoding="utf-8")


def build_payload() -> dict[str, object]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stocks_payload: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    combos: dict[str, dict[str, object]] = {}
    start_date = ""
    end_date = ""

    for code, name in STOCKS:
        frame = _load_frame(code)
        if not start_date or str(frame["trade_date"].iloc[0]) < start_date:
            start_date = str(frame["trade_date"].iloc[0])
        if not end_date or str(frame["trade_date"].iloc[-1]) > end_date:
            end_date = str(frame["trade_date"].iloc[-1])

        stock_combos: dict[str, object] = {}
        for window in WINDOWS:
            config = TurtleConfig(
                entry_window=window,
                exit_window=window,
                atr_window=20,
                stop_atr_multiplier=2.0,
                initial_capital=INITIAL_CAPITAL,
                fee_rate=0.0003,
                slippage_rate=0.0002,
                lot_size=100,
            )
            signals, equity, trades, metrics = run_backtest(frame, config)
            rows = _combine_rows(signals, equity)
            stock_combos[str(window)] = {
                "metrics": _round_metrics(metrics),
                "rows": rows,
                "trades": _compact_trades(trades),
            }
            summary_rows.append(
                {
                    "code": code,
                    "name": name,
                    "window": window,
                    **_round_metrics(metrics),
                }
            )

        combos[code] = stock_combos
        stocks_payload.append({"code": code, "name": name})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(METRICS_DIR / "html_parameter_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "dateRange": {"start": _format_date(start_date), "end": _format_date(end_date)},
        "stocks": stocks_payload,
        "windows": list(WINDOWS),
        "defaultStock": DEFAULT_STOCK,
        "defaultWindow": DEFAULT_WINDOW,
        "initialCapital": INITIAL_CAPITAL,
        "combos": combos,
        "summary": summary_rows,
    }


def _load_frame(code: str) -> pd.DataFrame:
    path = RAW_DIR / f"{code.replace('.', '_')}_daily.csv"
    if not path.exists():
        path = SOURCE_RAW_DIR / f"{code.replace('.', '_')}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved price data for {code}: {path}")
    frame = pd.read_csv(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.sort_values("trade_date").reset_index(drop=True)


def _combine_rows(signals: pd.DataFrame, equity: pd.DataFrame) -> list[dict[str, object]]:
    merged = signals.merge(
        equity[
            [
                "trade_date",
                "equity",
                "strategy_nav",
                "benchmark_nav",
                "drawdown",
            ]
        ],
        on="trade_date",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "date": _format_date(str(row.trade_date)),
                "close": _round(row.close),
                "upper": _round(row.upper_channel),
                "lower": _round(row.lower_channel),
                "atr": _round(row.atr),
                "stop": _round(row.stop_price),
                "signal": int(row.signal),
                "nav": _round(row.strategy_nav, 5),
                "benchmark": _round(row.benchmark_nav, 5),
                "drawdown": _round(row.drawdown, 5),
            }
        )
    return rows


def _compact_trades(trades: pd.DataFrame) -> list[dict[str, object]]:
    if trades.empty:
        return []
    rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        rows.append(
            {
                "signalDate": _format_date(str(row.signal_date)),
                "tradeDate": _format_date(str(row.trade_date)),
                "action": str(row.action),
                "reason": "通道突破" if row.reason == "breakout" else str(row.reason),
                "price": _round(row.price),
                "shares": int(row.shares),
            }
        )
    return rows


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: _round(value, 6) for key, value in metrics.items()}


def _round(value: object, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _format_date(value: str) -> str:
    value = str(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK4 海龟交易法则实战演练</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d7dee8;
      --soft: #edf2f7;
      --blue: #2563eb;
      --orange: #ea580c;
      --green: #16a34a;
      --red: #dc2626;
      --purple: #7c3aed;
      --shadow: 0 14px 36px rgba(15, 23, 42, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.6;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      display: grid;
      gap: 12px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(24px, 4vw, 36px);
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    .subtitle {{ color: var(--muted); max-width: 860px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-top: 16px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 14px;
    }}
    select {{
      min-width: 180px;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfdff;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .metric strong {{
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .chart-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .chart-title small {{ color: var(--muted); }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      overflow: visible;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 22px;
      height: 3px;
      vertical-align: middle;
      margin-right: 6px;
      background: var(--swatch);
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .insight {{
      border-left: 4px solid var(--blue);
      background: #fbfdff;
      padding: 10px 12px;
      border-radius: 6px;
    }}
    .insight strong {{ display: block; margin-bottom: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 650; background: #fbfdff; }}
    .table-wrap {{ overflow-x: auto; }}
    .note {{ color: var(--muted); font-size: 14px; margin-top: 10px; }}
    .code-steps {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfdff;
    }}
    .step code {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      white-space: normal;
    }}
    @media (max-width: 820px) {{
      main {{ width: min(100% - 20px, 1180px); }}
      .metric-grid, .insight-grid, .code-steps {{ grid-template-columns: 1fr; }}
      .chart-title {{ display: block; }}
      th, td {{ padding: 8px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>TASK4 海龟交易法则实战演练</h1>
    <p class="subtitle">基于已保存的日线行情数据，计算高低点通道、ATR、买卖信号，并用回测指标和参数敏感性观察策略表现。数据区间：<span id="date-range"></span>。</p>
  </header>

  <section class="panel" aria-labelledby="control-title">
    <h2 id="control-title">策略参数</h2>
    <div class="controls">
      <label>股票类型
        <select id="stock-select"></select>
      </label>
      <label>高低通道周期
        <select id="window-select"></select>
      </label>
    </div>
    <div class="metric-grid" id="metrics"></div>
  </section>

  <section class="panel" aria-labelledby="price-title">
    <div class="chart-title">
      <h2 id="price-title">图1：股价、高低价格通道与交易信号</h2>
      <small>买入为向上三角，卖出为向下三角</small>
    </div>
    <svg id="price-chart" viewBox="0 0 1100 430" role="img" aria-label="股价、高低价格通道与交易信号"></svg>
    <div class="legend">
      <span style="--swatch: var(--ink)">收盘价</span>
      <span style="--swatch: var(--blue)">上轨</span>
      <span style="--swatch: var(--orange)">下轨</span>
      <span style="--swatch: var(--green)">买入信号</span>
      <span style="--swatch: var(--red)">卖出信号</span>
    </div>
    <p class="note">解读：价格突破上轨时产生买入信号；价格跌破下轨或触发 ATR 止损时产生卖出信号。通道越短越灵敏，越长越偏向过滤噪声。</p>
  </section>

  <section class="panel" aria-labelledby="risk-title">
    <div class="chart-title">
      <h2 id="risk-title">图2：策略净值、买入持有基准与最大回撤</h2>
      <small>上半部分为净值，下半部分为回撤</small>
    </div>
    <svg id="nav-chart" viewBox="0 0 1100 430" role="img" aria-label="策略净值、买入持有基准与最大回撤"></svg>
    <div class="legend">
      <span style="--swatch: var(--blue)">策略净值</span>
      <span style="--swatch: var(--ink)">买入持有</span>
      <span style="--swatch: var(--red)">回撤</span>
    </div>
    <p class="note">解读：净值曲线衡量收益累积，回撤曲线衡量资金压力。趋势越连续，海龟策略越容易持仓吃到主要行情。</p>
  </section>

  <section class="panel" aria-labelledby="sensitivity-title">
    <h2 id="sensitivity-title">图3：股票类型与通道周期参数敏感性</h2>
    <div class="table-wrap">
      <table id="summary-table"></table>
    </div>
    <p class="note" id="best-note"></p>
  </section>

  <section class="panel" aria-labelledby="steps-title">
    <h2 id="steps-title">Python实现流程</h2>
    <div class="code-steps">
      <div class="step">1. 加载已保存股价数据<code>pd.read_csv("TASK3/data/raw/*_daily.csv")</code></div>
      <div class="step">2. 计算高低通道<code>high.rolling(N).max().shift(1)</code></div>
      <div class="step">3. 计算ATR<code>TR=max(high-low, |high-pre_close|, |low-pre_close|)</code></div>
      <div class="step">4. 生成买卖信号<code>close &gt; upper 买入；close &lt; lower 或 close &lt; stop 卖出</code></div>
      <div class="step">5. 下一交易日开盘成交<code>execution_signal = signal.shift(1)</code></div>
      <div class="step">6. 回测绩效指标<code>MDD、Sharpe、Cumulative Return</code></div>
    </div>
  </section>

  <section class="panel" aria-labelledby="scene-title">
    <h2 id="scene-title">适应场景与使用心得</h2>
    <div class="insight-grid">
      <div class="insight"><strong>适合场景</strong><p>趋势明确、突破后延续性强、流动性较好的标的。此时策略能用较少交易捕捉主要波段。</p></div>
      <div class="insight"><strong>主要风险</strong><p>震荡市容易出现假突破，短周期通道尤其可能频繁进出并放大交易成本。</p></div>
      <div class="insight"><strong>参数心得</strong><p>短通道更灵敏，长通道更稳健。实际应用应结合标的波动率、交易成本和更长样本做稳健性检验。</p></div>
    </div>
  </section>
</main>

<script>
const DATA = {payload_json};
const stockSelect = document.getElementById("stock-select");
const windowSelect = document.getElementById("window-select");
const metricsEl = document.getElementById("metrics");
const priceSvg = document.getElementById("price-chart");
const navSvg = document.getElementById("nav-chart");
const summaryTable = document.getElementById("summary-table");
const bestNote = document.getElementById("best-note");
document.getElementById("date-range").textContent = `${{DATA.dateRange.start}} 至 ${{DATA.dateRange.end}}`;

DATA.stocks.forEach(stock => {{
  const option = document.createElement("option");
  option.value = stock.code;
  option.textContent = `${{stock.name}}（${{stock.code}}）`;
  option.selected = stock.code === DATA.defaultStock;
  stockSelect.appendChild(option);
}});

DATA.windows.forEach(windowValue => {{
  const option = document.createElement("option");
  option.value = String(windowValue);
  option.textContent = `${{windowValue}}日`;
  option.selected = windowValue === DATA.defaultWindow;
  windowSelect.appendChild(option);
}});

stockSelect.addEventListener("change", update);
windowSelect.addEventListener("change", update);

function update() {{
  const stockCode = stockSelect.value;
  const windowValue = windowSelect.value;
  const combo = DATA.combos[stockCode][windowValue];
  renderMetrics(combo.metrics);
  renderPriceChart(combo.rows);
  renderNavChart(combo.rows);
  renderSummaryTable();
}}

function renderMetrics(metrics) {{
  const items = [
    ["累计回报", pct(metrics.cumulative_return)],
    ["最大回撤", pct(metrics.max_drawdown)],
    ["Sharpe", num(metrics.sharpe_ratio, 2)],
    ["交易次数", String(Math.round(metrics.trade_count))],
    ["期末资金", money(metrics.final_equity)]
  ];
  metricsEl.innerHTML = items.map(([label, value]) => `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
}}

function renderPriceChart(rows) {{
  const box = {{left: 68, top: 28, right: 1064, bottom: 360}};
  const values = rows.flatMap(row => [row.close, row.upper, row.lower]).filter(isFiniteNumber);
  const scale = makeScale(rows, values, box);
  let svg = axes(rows, scale, box, "价格");
  svg += path(rows, "close", scale, "var(--ink)", 2.2);
  svg += path(rows, "upper", scale, "var(--blue)", 2.2);
  svg += path(rows, "lower", scale, "var(--orange)", 2.2);
  rows.forEach((row, i) => {{
    if (!isFiniteNumber(row.close) || row.signal === 0) return;
    const x = scale.x(i);
    const y = scale.y(row.close);
    if (row.signal === 1) {{
      svg += `<polygon points="${{x}},${{y - 10}} ${{x - 8}},${{y + 8}} ${{x + 8}},${{y + 8}}" fill="var(--green)"><title>${{row.date}} 买入</title></polygon>`;
    }} else {{
      svg += `<polygon points="${{x}},${{y + 10}} ${{x - 8}},${{y - 8}} ${{x + 8}},${{y - 8}}" fill="var(--red)"><title>${{row.date}} 卖出</title></polygon>`;
    }}
  }});
  priceSvg.innerHTML = svg;
}}

function renderNavChart(rows) {{
  const navBox = {{left: 68, top: 28, right: 1064, bottom: 210}};
  const ddBox = {{left: 68, top: 265, right: 1064, bottom: 380}};
  const navValues = rows.flatMap(row => [row.nav, row.benchmark]).filter(isFiniteNumber);
  const ddValues = rows.map(row => row.drawdown).filter(isFiniteNumber);
  const navScale = makeScale(rows, navValues, navBox);
  const ddScale = makeScale(rows, ddValues.concat([0]), ddBox);
  let svg = axes(rows, navScale, navBox, "净值");
  svg += axes(rows, ddScale, ddBox, "回撤", true);
  svg += path(rows, "nav", navScale, "var(--blue)", 2.2);
  svg += path(rows, "benchmark", navScale, "var(--ink)", 2.2);
  svg += path(rows, "drawdown", ddScale, "var(--red)", 2.2);
  navSvg.innerHTML = svg;
}}

function renderSummaryTable() {{
  const windows = DATA.windows;
  const header = `<thead><tr><th>标的</th>${{windows.map(w => `<th>${{w}}日累计回报</th>`).join("")}}</tr></thead>`;
  const body = DATA.stocks.map(stock => {{
    const cells = windows.map(w => {{
      const metrics = DATA.combos[stock.code][String(w)].metrics;
      const value = metrics.cumulative_return;
      const color = value >= 0 ? "var(--green)" : "var(--red)";
      return `<td style="color:${{color}}">${{pct(value)}}</td>`;
    }}).join("");
    return `<tr><td>${{stock.name}}</td>${{cells}}</tr>`;
  }}).join("");
  summaryTable.innerHTML = header + `<tbody>${{body}}</tbody>`;
  const best = DATA.summary.reduce((acc, row) => row.cumulative_return > acc.cumulative_return ? row : acc, DATA.summary[0]);
  bestNote.textContent = `本次测试中累计回报最高的组合是 ${{best.name}} 的 ${{best.window}} 日通道，累计回报 ${{pct(best.cumulative_return)}}。`;
}}

function makeScale(rows, values, box) {{
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (!isFinite(min) || !isFinite(max)) {{ min = 0; max = 1; }}
  if (min === max) max = min + 1;
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  return {{
    min, max,
    x: i => box.left + (box.right - box.left) * i / Math.max(rows.length - 1, 1),
    y: v => box.bottom - (v - min) / (max - min) * (box.bottom - box.top)
  }};
}}

function axes(rows, scale, box, label, percent = false) {{
  let out = `<rect x="${{box.left}}" y="${{box.top}}" width="${{box.right - box.left}}" height="${{box.bottom - box.top}}" fill="none" stroke="var(--line)"/>`;
  for (let i = 0; i <= 4; i++) {{
    const value = scale.min + (scale.max - scale.min) * i / 4;
    const y = scale.y(value);
    out += `<line x1="${{box.left}}" x2="${{box.right}}" y1="${{y}}" y2="${{y}}" stroke="var(--soft)"/>`;
    out += `<text x="${{box.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="var(--muted)">${{percent ? pct(value) : num(value, 2)}}</text>`;
  }}
  const tickCount = 5;
  for (let i = 0; i <= tickCount; i++) {{
    const idx = Math.round((rows.length - 1) * i / tickCount);
    const x = scale.x(idx);
    out += `<text x="${{x}}" y="${{box.bottom + 22}}" text-anchor="middle" font-size="12" fill="var(--muted)">${{rows[idx].date.slice(0, 7)}}</text>`;
  }}
  out += `<text x="${{box.left}}" y="${{box.top - 10}}" font-size="12" fill="var(--muted)">${{label}}</text>`;
  return out;
}}

function path(rows, key, scale, color, width) {{
  let d = "";
  let started = false;
  rows.forEach((row, i) => {{
    const value = row[key];
    if (!isFiniteNumber(value)) {{
      started = false;
      return;
    }}
    const command = started ? "L" : "M";
    d += `${{command}} ${{scale.x(i).toFixed(2)}} ${{scale.y(value).toFixed(2)}} `;
    started = true;
  }});
  return `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="${{width}}" stroke-linejoin="round" stroke-linecap="round"/>`;
}}

function isFiniteNumber(value) {{
  return typeof value === "number" && Number.isFinite(value);
}}

function pct(value) {{
  return `${{(value * 100).toFixed(2)}}%`;
}}

function num(value, digits = 2) {{
  return Number(value).toFixed(digits);
}}

function money(value) {{
  return Number(value).toLocaleString("zh-CN", {{maximumFractionDigits: 0}});
}}

update();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
