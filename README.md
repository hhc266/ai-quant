# AIQuant - A股量化策略看板

个人 AI 量化交易学习项目，包含多个策略可视化看板。

## 在线访问

🔗 [https://hhc266.github.io/ai-quant/](https://hhc266.github.io/ai-quant/)

## 子站点

| TASK | 名称 | 说明 |
|------|------|------|
| TASK1 | K线行情看板 | 中芯国际(688981)近一年日K线与成交量 |
| TASK2 | 技术指标分析 | MACD、RSI、KDJ、布林带四大技术指标 |
| TASK3 | 双均线策略回测 | 5只A股标的双均线交叉策略交互式回测 |
| TASK4 | 海龟交易法则 | 通道突破+ATR止损策略回测 |

## 技术栈

- 前端：纯 HTML/CSS/JavaScript（无框架依赖）
- 数据源：Tushare Pro API
- CI/CD：GitHub Actions（每日工作日 16:00 北京时间自动更新）
- 部署：GitHub Pages

## 本地运行

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 设置 Tushare Token
export TUSHARE_TOKEN=your_token_here

# 生成 TASK3 看板（抓取数据 + 构建）
python TASK3/src/build_task3.py

# 生成 TASK4 看板（依赖 TASK3 的数据）
python TASK4/src/build_task4_html.py

# 生成 TASK2 看板（读取已有 SVG + JSON）
python TASK2/src/build_task2_html.py
```

## 新增子站点

1. 创建 `TASKX/` 目录，内含 `index.html`
2. 编辑根目录 `index.html`，在 `sites` 数组中追加一条记录
3. （可选）在 `.github/workflows/update-data.yml` 中添加构建步骤
4. 提交代码，GitHub Pages 自动部署

## 声明

本项目仅供学习研究，不构成任何投资建议。
