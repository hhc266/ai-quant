# -*- coding: utf-8 -*-
"""Build TASK2 technical indicators HTML from existing SVG charts + JSON summary.

Reads the 4 pre-generated SVG charts (MACD, RSI, KDJ, Bollinger Bands) and the
indicator summary JSON, then produces a self-contained index.html.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "technical_indicators"
HTML_PATH = ROOT / "index.html"

CHARTS = [
    ("smic_688981_macd.svg",            "MACD (12,26,9)",      "趋势动能与转折"),
    ("smic_688981_rsi_14.svg",          "RSI (14)",            "超买超卖区间"),
    ("smic_688981_kdj.svg",             "KDJ (9,3,3)",         "短期反转信号"),
    ("smic_688981_bollinger_bands.svg", "Bollinger Bands (20,2)", "波动率与突破"),
]

SIGNAL_MAP = {
    "rsi_zone": {"neutral": "中性", "overbought": "超买", "oversold": "超卖"},
    "macd_position": {"above_signal": "MACD > 信号线（多头）", "below_signal": "MACD < 信号线（空头）"},
    "bollinger_position": {"inside_band": "通道内", "above_upper": "突破上轨", "below_lower": "跌破下轨"},
    "kdj_zone": {"neutral": "中性", "overbought": "超买", "oversold": "超卖"},
    "kdj_position": {"k_above_d": "K > D（金叉）", "k_below_d": "K < D（死叉）"},
}


def main() -> None:
    summary = json.loads(
        (OUTPUT_DIR / "smic_688981_SH_indicator_summary.json").read_text(encoding="utf-8")
    )
    svgs = {}
    for fname, _title, _desc in CHARTS:
        svgs[fname] = (OUTPUT_DIR / fname).read_text(encoding="utf-8")
    html = build_html(summary, svgs)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Generated {HTML_PATH}")


def build_html(summary: dict, svgs: dict[str, str]) -> str:
    latest = summary.get("latest", {})
    signals = summary.get("signals_latest", {})
    params = summary.get("parameters", {})
    date_end = summary.get("date_end", "")
    rows = summary.get("rows", 0)

    metric_cards = _build_metric_cards(latest, signals)
    chart_sections = _build_chart_sections(svgs)
    param_section = _build_param_section(params)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK2 技术指标分析看板</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #e2e8f0;
      --up: #c5332b;
      --down: #0b8063;
      --accent: #2563eb;
      --shadow: 0 10px 30px rgba(15,23,42,.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", -apple-system, sans-serif;
      line-height: 1.6;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(24px, 4vw, 34px);
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .metric .signal {{
      margin-top: 4px;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 12px;
      display: inline-block;
      background: #eef2ff;
      color: var(--accent);
    }}
    .signal.bullish {{ background: #fef2f2; color: var(--up); }}
    .signal.bearish {{ background: #f0fdf4; color: var(--down); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 16px;
      overflow-x: auto;
    }}
    .panel h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .panel .desc {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .panel svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .params {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
    }}
    .param-card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fbfdff;
    }}
    .param-card .pname {{
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 4px;
    }}
    .param-card .pval {{
      color: var(--muted);
      font-size: 13px;
    }}
    footer {{
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 540px) {{
      main {{ width: min(100% - 16px, 1180px); }}
      .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>TASK2 技术指标分析看板</h1>
    <p class="subtitle">中芯国际 688981.SH · 数据区间 {summary.get("date_start", "")} 至 {date_end} · 共 {rows} 个交易日</p>
  </header>

  <section class="metrics-grid">
{metric_cards}
  </section>

{chart_sections}

  <section class="panel">
    <h2>指标参数说明</h2>
    <p class="desc">各技术指标的计算参数</p>
    <div class="params">
{param_section}
    </div>
  </section>

  <footer>数据来源：Tushare Pro 日线数据 · 仅供学习研究，不构成投资建议</footer>
</main>
</body>
</html>"""


def _build_metric_cards(latest: dict, signals: dict) -> str:
    cards = []

    def _signal_class(zone: str) -> str:
        if "overbought" in zone or "above" in zone:
            return "signal bullish"
        if "oversold" in zone or "below" in zone:
            return "signal bearish"
        return "signal"

    def _signal_text(key: str, default: str = "") -> str:
        val = signals.get(key, default)
        return SIGNAL_MAP.get(key, {}).get(val, val)

    cards.append(f"""    <div class="metric">
      <div class="label">最新收盘价</div>
      <div class="value">{latest.get("close", "--")}</div>
      <div class="signal">{latest.get("trade_date", "")}</div>
    </div>""")

    rsi_val = latest.get("rsi_14", "--")
    rsi_zone = signals.get("rsi_zone", "")
    cards.append(f"""    <div class="metric">
      <div class="label">RSI (14)</div>
      <div class="value">{_fmt(rsi_val)}</div>
      <span class="{_signal_class(rsi_zone)}">{_signal_text("rsi_zone")}</span>
    </div>""")

    macd_val = latest.get("macd", "--")
    macd_pos = signals.get("macd_position", "")
    cards.append(f"""    <div class="metric">
      <div class="label">MACD</div>
      <div class="value">{_fmt(macd_val)}</div>
      <span class="{_signal_class(macd_pos)}">{_signal_text("macd_position")}</span>
    </div>""")

    kdj_j = latest.get("kdj_j", "--")
    kdj_zone = signals.get("kdj_zone", "")
    cards.append(f"""    <div class="metric">
      <div class="label">KDJ-J</div>
      <div class="value">{_fmt(kdj_j)}</div>
      <span class="{_signal_class(kdj_zone)}">{_signal_text("kdj_zone")}</span>
    </div>""")

    bb_pct = latest.get("bb_percent_b", "--")
    bb_pos = signals.get("bollinger_position", "")
    cards.append(f"""    <div class="metric">
      <div class="label">BB %B</div>
      <div class="value">{_fmt_pct(bb_pct)}</div>
      <span class="{_signal_class(bb_pos)}">{_signal_text("bollinger_position")}</span>
    </div>""")

    return "\n".join(cards)


def _build_chart_sections(svgs: dict[str, str]) -> str:
    sections = []
    for fname, title, desc in CHARTS:
        svg_content = svgs.get(fname, "")
        sections.append(f"""  <section class="panel">
    <h2>{title}</h2>
    <p class="desc">{desc}</p>
    {svg_content}
  </section>""")
    return "\n\n".join(sections)


def _build_param_section(params: dict) -> str:
    cards = []

    rsi_p = params.get("rsi_period", 14)
    cards.append(f"""      <div class="param-card">
        <div class="pname">RSI</div>
        <div class="pval">周期 = {rsi_p}</div>
      </div>""")

    macd_p = params.get("macd", {})
    cards.append(f"""      <div class="param-card">
        <div class="pname">MACD</div>
        <div class="pval">fast={macd_p.get("fast",12)}, slow={macd_p.get("slow",26)}, signal={macd_p.get("signal",9)}</div>
      </div>""")

    bb_p = params.get("bollinger", {})
    cards.append(f"""      <div class="param-card">
        <div class="pname">Bollinger Bands</div>
        <div class="pval">周期={bb_p.get("period",20)}, 标准差倍数={bb_p.get("std_multiplier",2)}</div>
      </div>""")

    kdj_p = params.get("kdj", {})
    cards.append(f"""      <div class="param-card">
        <div class="pname">KDJ</div>
        <div class="pval">周期={kdj_p.get("period",9)}, 平滑={kdj_p.get("smooth",3)}, 初始K/D={kdj_p.get("initial_kd",50)}</div>
      </div>""")

    return "\n".join(cards)


def _fmt(value) -> str:
    if value is None or value == "--":
        return "--"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value) -> str:
    if value is None or value == "--":
        return "--"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
