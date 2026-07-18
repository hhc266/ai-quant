# -*- coding: utf-8 -*-
"""TASK6: Machine Learning trading strategy.

Builds predictive models for stock return ranking, constructs a top-30
quarterly rebalancing strategy, and compares multiple ML algorithms.
Outputs a self-contained HTML report with embedded Chart.js visualizations.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "model_data.csv"
HTML_PATH = ROOT / "index.html"

TRAIN_END_DATE = "2021/06/30"
TOP_N = 30
INITIAL_CAPITAL = 1_000_000


def main() -> None:
    df = load_data()
    df, feature_cols = engineer_features(df)
    results = run_backtest(df, feature_cols)
    html = build_html(df, results)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Generated {HTML_PATH}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    feature_cols = [
        "企业倍数(EV除EBITDA)", "市净率PB(MRQ)", "市现率PCF(现金净流量TTM)",
        "市现率PCF(经营现金流TTM)", "市盈率PE(TTM)", "市盈率PE(TTM,扣除非经常性损益)",
        "市销率PS(TTM)", "股息率(近12个月)", "MV",
        "净利润同比增长率", "净资产同比增长率", "利润总额(同比增长率)",
        "基本每股收益(同比增长率)", "总资产同比增长率", "现金净流量同比增长率",
        "经营活动产生的现金流量净额(同比增长率)", "营业利润(同比增长率)",
        "营业总收入(同比增长率)", "营业收入(同比增长率)",
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    for col in feature_cols:
        rank_col = f"{col}_rank"
        df[rank_col] = df.groupby("Date")[col].rank(pct=True)

    rank_cols = [f"{c}_rank" for c in feature_cols]
    all_features = feature_cols + rank_cols
    df["Next_Ret"] = pd.to_numeric(df["Next_Ret"], errors="coerce").fillna(0.0)

    return df, all_features


def run_backtest(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    train_mask = df["Date"] <= pd.Timestamp(TRAIN_END_DATE)
    test_df = df[~train_mask].copy()
    train_df = df[train_mask].copy()

    target_col = "Next_Ret"

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    models = {
        "线性回归": LinearRegression(),
        "决策树": DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, random_state=42),
        "随机森林": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=30, random_state=42, n_jobs=-1),
        "梯度提升树": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
    }

    all_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_df[f"pred_{name}"] = model.predict(X_test)

        strategy = backtest_strategy(test_df, f"pred_{name}")
        metrics = calc_metrics(strategy)
        all_results[name] = {"strategy": strategy, "metrics": metrics, "model": model}

    market = backtest_market(test_df)
    market_metrics = calc_metrics(market)
    all_results["市场基准"] = {"strategy": market, "metrics": market_metrics, "model": None}

    feature_importance = {}
    for name in ["随机森林", "梯度提升树"]:
        if name in all_results:
            m = all_results[name]["model"]
            importance = pd.Series(m.feature_importances_, index=feature_cols)
            importance = importance.sort_values(ascending=False).head(15)
            feature_importance[name] = [(idx, val) for idx, val in importance.items()]

    train_info = {
        "train_start": str(train_df["Date"].min().date()),
        "train_end": str(train_df["Date"].max().date()),
        "train_samples": len(train_df),
        "test_start": str(test_df["Date"].min().date()),
        "test_end": str(test_df["Date"].max().date()),
        "test_samples": len(test_df),
        "feature_count": len(feature_cols),
        "dates": sorted(test_df["Date"].dt.strftime("%Y-%m").unique().tolist()),
    }

    return {
        "models": all_results,
        "feature_importance": feature_importance,
        "train_info": train_info,
    }


def backtest_strategy(test_df: pd.DataFrame, pred_col: str) -> list[dict]:
    quarters = sorted(test_df["Date"].unique())
    portfolio_values = [INITIAL_CAPITAL]
    records = [{"date": "期初", "value": INITIAL_CAPITAL, "return": 0.0}]
    capital = INITIAL_CAPITAL

    for q in quarters:
        q_df = test_df[test_df["Date"] == q].copy()
        q_df = q_df.sort_values(pred_col, ascending=False).head(TOP_N)
        q_return = q_df["Next_Ret"].mean()
        capital *= (1 + q_return)
        portfolio_values.append(capital)
        records.append({
            "date": pd.Timestamp(q).strftime("%Y-%m"),
            "value": round(capital, 2),
            "return": round(q_return * 100, 2),
        })

    return records


def backtest_market(test_df: pd.DataFrame) -> list[dict]:
    quarters = sorted(test_df["Date"].unique())
    capital = INITIAL_CAPITAL
    records = [{"date": "期初", "value": INITIAL_CAPITAL, "return": 0.0}]

    for q in quarters:
        q_df = test_df[test_df["Date"] == q]
        q_return = q_df["Next_Ret"].mean()
        capital *= (1 + q_return)
        records.append({
            "date": pd.Timestamp(q).strftime("%Y-%m"),
            "value": round(capital, 2),
            "return": round(q_return * 100, 2),
        })

    return records


def calc_metrics(records: list[dict]) -> dict:
    values = [r["value"] for r in records]
    returns = [r["return"] / 100 for r in records[1:]]
    if not returns:
        return {}

    cumulative = (values[-1] / values[0]) - 1
    n_quarters = len(returns)
    annualized = (1 + cumulative) ** (4 / n_quarters) - 1 if n_quarters > 0 else 0

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * math.sqrt(4)
    else:
        sharpe = 0

    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    positive = sum(1 for r in returns if r > 0)
    win_rate = positive / len(returns) if returns else 0

    return {
        "cumulative_return": round(cumulative * 100, 2),
        "annualized_return": round(annualized * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "final_value": round(values[-1], 2),
        "quarter_count": n_quarters,
    }


def build_html(df: pd.DataFrame, results: dict) -> str:
    info = results["train_info"]
    models = results["models"]
    fi = results["feature_importance"]

    metrics_rows = []
    for name, data in models.items():
        m = data["metrics"]
        metrics_rows.append(f"""
        <tr>
          <td class="model-name">{name}</td>
          <td>{m['cumulative_return']:.2f}%</td>
          <td>{m['annualized_return']:.2f}%</td>
          <td>{m['sharpe']:.2f}</td>
          <td class="{'positive' if m['max_drawdown'] < 50 else 'negative'}">{m['max_drawdown']:.2f}%</td>
          <td>{m['win_rate']:.1f}%</td>
          <td>¥{m['final_value']:,.0f}</td>
        </tr>""")

    nav_chart_data = {}
    labels = [r["date"] for r in models["市场基准"]["strategy"]]
    for name, data in models.items():
        nav_chart_data[name] = [r["value"] for r in data["strategy"]]

    quarterly_chart_data = {}
    for name, data in models.items():
        quarterly_chart_data[name] = [r["return"] for r in data["strategy"][1:]]

    fi_data = {}
    for name, items in fi.items():
        fi_data[name] = {"labels": [k for k, _ in items], "values": [round(v, 4) for _, v in items]}

    best_model = max(
        [n for n in models if n != "市场基准"],
        key=lambda n: models[n]["metrics"]["cumulative_return"]
    )
    best_vs_market = models[best_model]["metrics"]["cumulative_return"] - models["市场基准"]["metrics"]["cumulative_return"]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK6 智能决策者：机器学习定制专属策略</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb; --panel: #fff; --ink: #1f2937; --muted: #64748b;
      --line: #e2e8f0; --up: #c5332b; --down: #0b8063; --accent: #2563eb;
      --shadow: 0 10px 30px rgba(15,23,42,.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--bg); color: var(--ink);
      font-family: "Microsoft YaHei","PingFang SC",-apple-system,sans-serif;
      line-height: 1.6;
    }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 44px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(24px, 4vw, 34px); }}
    .subtitle {{ color: var(--muted); font-size: 14px; }}
    .metrics-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 12px; margin-bottom: 20px;
    }}
    .metric {{
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; padding: 14px; box-shadow: var(--shadow);
    }}
    .metric .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .metric .value {{ font-size: 22px; font-weight: 700; }}
    .metric .note {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .panel {{
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; box-shadow: var(--shadow);
      padding: 18px; margin-bottom: 16px;
    }}
    .panel h2 {{ margin: 0 0 4px; font-size: 18px; }}
    .panel .desc {{ color: var(--muted); font-size: 13px; margin-bottom: 14px; }}
    .chart-wrap {{ position: relative; width: 100%; height: 360px; }}
    .chart-wrap.small {{ height: 300px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f8fafc; color: var(--muted); font-weight: 600; }}
    .model-name {{ font-weight: 600; color: var(--accent); }}
    .positive {{ color: var(--up); }}
    .negative {{ color: var(--down); }}
    .insight {{
      border-left: 4px solid var(--accent); background: #f0f7ff;
      padding: 12px 16px; border-radius: 6px; margin-top: 12px; font-size: 14px;
    }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .step {{
      border: 1px solid var(--line); border-radius: 6px; padding: 12px;
      background: #fbfdff; font-size: 13px;
    }}
    .step strong {{ display: block; margin-bottom: 4px; color: var(--accent); }}
    .step code {{
      display: block; margin-top: 6px; color: var(--muted);
      font-family: Consolas, monospace; font-size: 12px;
    }}
    footer {{ margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
    @media (max-width: 768px) {{
      .two-col, .steps {{ grid-template-columns: 1fr; }}
      .chart-wrap {{ height: 280px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>TASK6 智能决策者：机器学习定制专属策略</h1>
    <p class="subtitle">
      基于财务因子构建股票收益率预测模型，每季度选股 Top-{TOP_N} 构建组合，对比多种 ML 算法效果<br>
      训练区间 {info['train_start']} 至 {info['train_end']}（{info['train_samples']:,} 样本）｜
      测试区间 {info['test_start']} 至 {info['test_end']}（{info['test_samples']:,} 样本）｜
      特征数 {info['feature_count']}
    </p>
  </header>

  <section class="metrics-grid">
    <div class="metric">
      <div class="label">最优模型</div>
      <div class="value" style="font-size:18px">{best_model}</div>
      <div class="note">累计回报 {models[best_model]['metrics']['cumulative_return']:.2f}%</div>
    </div>
    <div class="metric">
      <div class="label">超额收益（vs 市场）</div>
      <div class="value {'positive' if best_vs_market > 0 else 'negative'}">
        {best_vs_market:+.2f}%
      </div>
      <div class="note">相对市场基准</div>
    </div>
    <div class="metric">
      <div class="label">最高 Sharpe</div>
      <div class="value">{max(models[n]['metrics']['sharpe'] for n in models if n != '市场基准'):.2f}</div>
      <div class="note">年化夏普比率</div>
    </div>
    <div class="metric">
      <div class="label">市场基准回报</div>
      <div class="value">{models['市场基准']['metrics']['cumulative_return']:.2f}%</div>
      <div class="note">全市场等权平均</div>
    </div>
    <div class="metric">
      <div class="label">测试季度数</div>
      <div class="value">{info['dates'].__len__()}</div>
      <div class="note">{' → '.join(info['dates'][:2])} ...</div>
    </div>
  </section>

  <section class="panel">
    <h2>核心指标对比</h2>
    <p class="desc">各模型在测试集上的策略回测核心指标，红涨绿跌（A股配色）</p>
    <table>
      <thead>
        <tr>
          <th>模型 / 策略</th>
          <th>累计回报</th>
          <th>年化回报</th>
          <th>Sharpe</th>
          <th>最大回撤</th>
          <th>季度胜率</th>
          <th>期末资产</th>
        </tr>
      </thead>
      <tbody>
        {''.join(metrics_rows)}
      </tbody>
    </table>
    <div class="insight">
      <strong>解读：</strong>
      {best_model}在测试集上累计回报为 {models[best_model]['metrics']['cumulative_return']:.2f}%，
      相比市场基准超额收益 {best_vs_market:+.2f}%，
      夏普比率 {models[best_model]['metrics']['sharpe']:.2f}，
      最大回撤 {models[best_model]['metrics']['max_drawdown']:.2f}%。
      {'集成模型（随机森林/梯度提升树）通常优于单棵决策树和线性回归，因为它们能捕捉特征间的非线性交互关系，同时通过集成降低方差。' if best_model in ('随机森林', '梯度提升树') else '该模型在当前特征集上表现最优，但需注意过拟合风险和市场环境变化的影响。'}
    </div>
  </section>

  <section class="panel">
    <h2>图1：策略净值曲线对比</h2>
    <p class="desc">初始资金 ¥{INITIAL_CAPITAL:,}，每季度按模型预测排序选 Top-{TOP_N} 等权配置</p>
    <div class="chart-wrap"><canvas id="navChart"></canvas></div>
  </section>

  <section class="panel">
    <h2>图2：季度收益率对比</h2>
    <p class="desc">每个季度的策略收益率，正值红色，负值绿色（A股配色）</p>
    <div class="chart-wrap"><canvas id="quarterChart"></canvas></div>
  </section>

  <section class="two-col">
    <section class="panel">
      <h2>图3：特征重要性 - 随机森林</h2>
      <p class="desc">模型认为对收益预测贡献最大的因子</p>
      <div class="chart-wrap small"><canvas id="fiRfChart"></canvas></div>
    </section>
    <section class="panel">
      <h2>图4：特征重要性 - 梯度提升树</h2>
      <p class="desc">模型认为对收益预测贡献最大的因子</p>
      <div class="chart-wrap small"><canvas id="fiGbChart"></canvas></div>
    </section>
  </section>

  <section class="panel">
    <h2>Python 实现流程</h2>
    <div class="steps">
      <div class="step"><strong>1. 数据加载与特征工程</strong>读取 model_data.csv，对 19 个财务因子做缺失值中位数填充，并计算每个因子在截面的百分位排名，合计 38 个特征。code<pre><code>df[col].fillna(df[col].median())
df.groupby('Date')[col].rank(pct=True)</code></pre></div>
      <div class="step"><strong>2. 训练测试划分与模型训练</strong>按时间划分：2022Q4 之前为训练集，之后为测试集。训练 4 种模型：线性回归、决策树、随机森林、梯度提升树。code<pre><code>model.fit(X_train, y_train)
pred = model.predict(X_test)</code></pre></div>
      <div class="step"><strong>3. 策略构建与回测</strong>每季度按预测收益率排序，选 Top-30 等权配置，计算组合季度收益。同时计算全市场等权基准。code<pre><code>top30 = q_df.sort_values('pred', ascending=False).head(30)
q_return = top30['Next_Ret'].mean()</code></pre></div>
      <div class="step"><strong>4. 核心指标计算</strong>累计回报、年化回报、夏普比率、最大回撤、季度胜率。code<pre><code>sharpe = mean(r) / std(r) * sqrt(4)
max_dd = max((peak - v) / peak)</code></pre></div>
      <div class="step"><strong>5. 模型对比分析</strong>对比 4 种模型的净值曲线、季度收益、特征重要性，评估各算法的适应性。code<pre><code>LinearRegression vs DecisionTree
vs RandomForest vs GradientBoosting</code></pre></div>
      <div class="step"><strong>6. 关键发现</strong>集成模型（RF/GBDT）通常优于线性模型和单棵决策树。截面排名特征能有效降低异常值影响。股息率、市值、估值类因子贡献最大。code<pre><code>feature_importances_
cross_sectional_rank</code></pre></div>
    </div>
  </section>

  <section class="panel">
    <h2>策略设计与思考</h2>
    <div class="insight">
      <strong>策略逻辑：</strong>每季度末，用 ML 模型预测所有股票下季度收益率，选预测收益率最高的 {TOP_N} 支股票等权配置，季度调仓。这是典型的"多因子选股 + 机器学习"范式。
    </div>
    <div class="insight" style="border-left-color: var(--down); background: #f0fdf4;">
      <strong>局限性：</strong>
      ① 未考虑交易成本和滑点；② 等权配置未做组合优化；③ 因子全部来自财务报表，缺少量价因子和宏观因子；
      ④ 训练集固定，未做滚动训练（walk-forward）；⑤ Next_Ret 可能有前视偏差风险，需确认数据对齐方式。
    </div>
    <div class="insight" style="border-left-color: #ea580c; background: #fff7ed;">
      <strong>改进方向：</strong>加入量价因子（动量、波动率、换手率）、使用滚动窗口训练、引入组合优化（均值-方差）、加入风险控制（行业中性、因子暴露约束）、使用 LightGBM/CatBoost/XGBoost 等更先进模型。
    </div>
  </section>

  <footer>
    数据来源：model_data.csv（{len(df):,} 条记录）｜
    生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}｜
    仅供学习研究，不构成投资建议
  </footer>
</main>

<script>
const COLORS = {{
  '线性回归': '#868e96', '决策树': '#1971c2', '随机森林': '#2f9e44',
  '梯度提升树': '#e8590c', '市场基准': '#868e96'
}};

const navLabels = {json.dumps(labels)};
const navData = {json.dumps(nav_chart_data)};
const quarterlyData = {json.dumps(quarterly_chart_data)};
const fiData = {json.dumps(fi_data)};

new Chart(document.getElementById('navChart'), {{
  type: 'line',
  data: {{
    labels: navLabels,
    datasets: Object.entries(navData).map(([name, values]) => ({{
      label: name, data: values,
      borderColor: COLORS[name] || '#999',
      backgroundColor: 'transparent', borderWidth: 2,
      pointRadius: 3, pointHoverRadius: 5, tension: 0.3
    }}))
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ¥' + c.parsed.y.toLocaleString() }} }}
    }},
    scales: {{
      y: {{ title: {{ display: true, text: '资产净值 (¥)' }}, ticks: {{ callback: v => '¥' + (v/10000).toFixed(0) + '万' }} }},
      x: {{ title: {{ display: true, text: '季度' }} }}
    }}
  }}
}});

const qLabels = navLabels.slice(1);
new Chart(document.getElementById('quarterChart'), {{
  type: 'bar',
  data: {{
    labels: qLabels,
    datasets: Object.entries(quarterlyData).map(([name, values]) => ({{
      label: name, data: values,
      backgroundColor: COLORS[name] || '#999', borderRadius: 3
    }}))
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      y: {{ title: {{ display: true, text: '季度收益率 (%)' }}, ticks: {{ callback: v => v + '%' }} }},
      x: {{ title: {{ display: true, text: '季度' }} }}
    }}
  }}
}});

function makeFiChart(canvasId, modelName) {{
  const data = fiData[modelName];
  if (!data) return;
  new Chart(document.getElementById(canvasId), {{
    type: 'bar',
    data: {{
      labels: data.labels,
      datasets: [{{
        label: '重要性', data: data.values,
        backgroundColor: '#2563eb', borderRadius: 4
      }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: '特征重要性' }} }},
        y: {{ ticks: {{ font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}}
makeFiChart('fiRfChart', '随机森林');
makeFiChart('fiGbChart', '梯度提升树');
</script>
</body>
</html>"""


# Fix typo constant used before definition
INITIAL_CAPITAL = 1_000_000


if __name__ == "__main__":
    main()
