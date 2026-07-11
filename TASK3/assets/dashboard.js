(function () {
  "use strict";

  const payload = window.TASK3_STOCK_DATA;
  if (!payload || !Array.isArray(payload.stocks) || payload.stocks.length === 0) {
    document.body.innerHTML = "<p>看板数据加载失败，请确认 assets/dashboard-data.js 存在。</p>";
    return;
  }

  const form = document.getElementById("strategy-form");
  const stockSelect = document.getElementById("stock-select");
  const startDateInput = document.getElementById("start-date");
  const endDateInput = document.getElementById("end-date");
  const shortWindowInput = document.getElementById("short-window");
  const longWindowInput = document.getElementById("long-window");
  const feeRateInput = document.getElementById("fee-rate");
  const slippageRateInput = document.getElementById("slippage-rate");
  const initialCapitalInput = document.getElementById("initial-capital");
  const validationMessage = document.getElementById("validation-message");
  const sourceStatus = document.getElementById("source-status");
  const parameterStatus = document.getElementById("parameter-status");
  const dataStamp = document.getElementById("data-stamp");
  const resultSummary = document.getElementById("result-summary");
  const priceChart = document.getElementById("price-chart");
  const navChart = document.getElementById("nav-chart");
  const drawdownChart = document.getElementById("drawdown-chart");

  const state = {
    result: null,
    resizeTimer: null,
  };

  const metricBindings = {
    cumulativeReturn: document.getElementById("metric-return"),
    maxDrawdown: document.getElementById("metric-mdd"),
    sharpeRatio: document.getElementById("metric-sharpe"),
    tradeCount: document.getElementById("metric-trades"),
    finalEquity: document.getElementById("metric-equity"),
    buyHoldReturn: document.getElementById("metric-benchmark"),
  };

  function initialize() {
    for (const stock of payload.stocks) {
      const option = document.createElement("option");
      option.value = stock.tsCode;
      option.textContent = `${stock.name}（${stock.tsCode}）`;
      stockSelect.appendChild(option);
    }

    shortWindowInput.value = payload.defaultConfig.shortWindow;
    longWindowInput.value = payload.defaultConfig.longWindow;
    feeRateInput.value = (payload.defaultConfig.feeRate * 100).toFixed(2);
    slippageRateInput.value = (payload.defaultConfig.slippageRate * 100).toFixed(2);
    initialCapitalInput.value = payload.defaultConfig.initialCapital;
    updateDateRange(true);

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      calculateAndRender();
    });
    stockSelect.addEventListener("change", function () {
      updateDateRange(true);
      calculateAndRender();
    });

    const resizeObserver = new ResizeObserver(function () {
      window.clearTimeout(state.resizeTimer);
      state.resizeTimer = window.setTimeout(function () {
        if (state.result) renderCharts(state.result);
      }, 100);
    });
    resizeObserver.observe(priceChart);
    resizeObserver.observe(navChart);
    resizeObserver.observe(drawdownChart);

    calculateAndRender();
  }

  function selectedStock() {
    return payload.stocks.find((stock) => stock.tsCode === stockSelect.value) || payload.stocks[0];
  }

  function updateDateRange(resetValues) {
    const stock = selectedStock();
    const firstDate = stock.rows[0].date;
    const lastDate = stock.rows[stock.rows.length - 1].date;
    startDateInput.min = firstDate;
    startDateInput.max = lastDate;
    endDateInput.min = firstDate;
    endDateInput.max = lastDate;
    if (resetValues || !startDateInput.value) startDateInput.value = firstDate;
    if (resetValues || !endDateInput.value) endDateInput.value = lastDate;
    dataStamp.innerHTML = `<strong>Tushare 前复权缓存</strong>${firstDate} 至 ${lastDate}`;
  }

  function readConfig() {
    return {
      stock: selectedStock(),
      startDate: startDateInput.value,
      endDate: endDateInput.value,
      shortWindow: Number(shortWindowInput.value),
      longWindow: Number(longWindowInput.value),
      feeRate: Number(feeRateInput.value) / 100,
      slippageRate: Number(slippageRateInput.value) / 100,
      initialCapital: Number(initialCapitalInput.value),
      lotSize: 100,
    };
  }

  function validateConfig(config) {
    if (!config.startDate || !config.endDate || config.startDate > config.endDate) {
      return "开始日期不能晚于结束日期。";
    }
    if (!Number.isInteger(config.shortWindow) || !Number.isInteger(config.longWindow)) {
      return "均线周期必须是整数。";
    }
    if (config.shortWindow < 2 || config.longWindow < 3 || config.shortWindow >= config.longWindow) {
      return "短周期必须小于长周期，且周期至少分别为 2 和 3。";
    }
    if (config.feeRate < 0 || config.slippageRate < 0) {
      return "手续费率和滑点率不能为负数。";
    }
    if (!Number.isFinite(config.initialCapital) || config.initialCapital <= 0) {
      return "初始资金必须大于 0。";
    }
    const rows = config.stock.rows.filter((row) => row.date >= config.startDate && row.date <= config.endDate);
    if (rows.length < config.longWindow + 2) {
      return `当前时间窗口只有 ${rows.length} 个交易日，至少需要 ${config.longWindow + 2} 个交易日。`;
    }
    return "";
  }

  function calculateAndRender() {
    const config = readConfig();
    const validation = validateConfig(config);
    validationMessage.textContent = validation;
    if (validation) return;

    state.result = runBacktest(config);
    updateStatus(state.result);
    updateMetrics(state.result);
    renderCharts(state.result);
    renderTrades(state.result);
    updateInterpretations(state.result);
  }

  function runBacktest(config) {
    const rows = config.stock.rows
      .filter((row) => row.date >= config.startDate && row.date <= config.endDate)
      .map((row) => ({ ...row }));
    const closes = rows.map((row) => Number(row.close));
    const shortSma = rollingAverage(closes, config.shortWindow);
    const longSma = rollingAverage(closes, config.longWindow);
    const signals = new Array(rows.length).fill(0);

    for (let index = 1; index < rows.length; index += 1) {
      const previousShort = shortSma[index - 1];
      const previousLong = longSma[index - 1];
      const currentShort = shortSma[index];
      const currentLong = longSma[index];
      if ([previousShort, previousLong, currentShort, currentLong].some((value) => value === null)) continue;
      if (previousShort <= previousLong && currentShort > currentLong) signals[index] = 1;
      if (previousShort >= previousLong && currentShort < currentLong) signals[index] = -1;
    }

    let cash = config.initialCapital;
    let shares = 0;
    const trades = [];
    const points = [];

    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      const executionSignal = index > 0 ? signals[index - 1] : 0;
      const openPrice = Number(row.open);
      const closePrice = Number(row.close);

      if (executionSignal === 1 && shares === 0) {
        const buyPrice = openPrice * (1 + config.slippageRate);
        let maxShares = Math.floor(cash / (buyPrice * (1 + config.feeRate)));
        maxShares = Math.floor(maxShares / config.lotSize) * config.lotSize;
        if (maxShares > 0) {
          const gross = maxShares * buyPrice;
          const fee = gross * config.feeRate;
          cash -= gross + fee;
          shares = maxShares;
          trades.push({
            date: row.date,
            action: "BUY",
            price: buyPrice,
            shares,
            fee,
            cashAfter: cash,
          });
        }
      } else if (executionSignal === -1 && shares > 0) {
        const sellPrice = openPrice * (1 - config.slippageRate);
        const gross = shares * sellPrice;
        const fee = gross * config.feeRate;
        cash += gross - fee;
        trades.push({
          date: row.date,
          action: "SELL",
          price: sellPrice,
          shares,
          fee,
          cashAfter: cash,
        });
        shares = 0;
      }

      const marketValue = shares * closePrice;
      points.push({
        ...row,
        smaShort: shortSma[index],
        smaLong: longSma[index],
        signal: signals[index],
        executionSignal,
        cash,
        shares,
        marketValue,
        equity: cash + marketValue,
      });
    }

    let runningMax = points[0].equity;
    const initialClose = Number(points[0].close);
    for (const point of points) {
      runningMax = Math.max(runningMax, point.equity);
      point.strategyNav = point.equity / config.initialCapital;
      point.benchmarkNav = Number(point.close) / initialClose;
      point.drawdown = point.equity / runningMax - 1;
    }

    const metrics = calculateMetrics(points, trades, config.initialCapital);
    return { config, points, trades, metrics };
  }

  function rollingAverage(values, windowSize) {
    const averages = new Array(values.length).fill(null);
    let sum = 0;
    for (let index = 0; index < values.length; index += 1) {
      sum += values[index];
      if (index >= windowSize) sum -= values[index - windowSize];
      if (index >= windowSize - 1) averages[index] = sum / windowSize;
    }
    return averages;
  }

  function calculateMetrics(points, trades, initialCapital) {
    const equities = points.map((point) => point.equity);
    const dailyReturns = equities.map((value, index) => (index === 0 ? 0 : value / equities[index - 1] - 1));
    const meanReturn = average(dailyReturns);
    const variance = average(dailyReturns.map((value) => (value - meanReturn) ** 2));
    const standardDeviation = Math.sqrt(variance);
    const cumulativeReturn = equities[equities.length - 1] / initialCapital - 1;
    const annualizedReturn = cumulativeReturn > -1
      ? (1 + cumulativeReturn) ** (252 / points.length) - 1
      : -1;
    const maxDrawdown = Math.min(...points.map((point) => point.drawdown));
    const sharpeRatio = standardDeviation > 0 ? (meanReturn / standardDeviation) * Math.sqrt(252) : 0;
    const buyHoldReturn = Number(points[points.length - 1].close) / Number(points[0].close) - 1;
    const roundTripReturns = [];
    let entryPrice = null;
    for (const trade of trades) {
      if (trade.action === "BUY") entryPrice = trade.price;
      if (trade.action === "SELL" && entryPrice !== null) {
        roundTripReturns.push(trade.price / entryPrice - 1);
        entryPrice = null;
      }
    }
    const winRate = roundTripReturns.length
      ? roundTripReturns.filter((value) => value > 0).length / roundTripReturns.length
      : 0;
    const holdingRatio = points.filter((point) => point.shares > 0).length / points.length;
    const maxDrawdownIndex = points.findIndex((point) => point.drawdown === maxDrawdown);

    return {
      cumulativeReturn,
      annualizedReturn,
      maxDrawdown,
      maxDrawdownIndex,
      sharpeRatio,
      tradeCount: trades.length,
      finalEquity: equities[equities.length - 1],
      winRate,
      holdingRatio,
      buyHoldReturn,
    };
  }

  function updateStatus(result) {
    const { config, points } = result;
    sourceStatus.innerHTML = `<strong>${config.stock.name}</strong> · ${config.stock.source} · ${points.length} 个交易日`;
    parameterStatus.textContent = `SMA ${config.shortWindow}/${config.longWindow} · 手续费 ${formatPercent(config.feeRate, 2)} · 滑点 ${formatPercent(config.slippageRate, 2)} · 100 股/手`;
  }

  function updateMetrics(result) {
    const metrics = result.metrics;
    setMetric(metricBindings.cumulativeReturn, formatPercent(metrics.cumulativeReturn), metrics.cumulativeReturn);
    setMetric(metricBindings.maxDrawdown, formatPercent(metrics.maxDrawdown), metrics.maxDrawdown);
    setMetric(metricBindings.sharpeRatio, metrics.sharpeRatio.toFixed(2), metrics.sharpeRatio);
    setMetric(metricBindings.tradeCount, String(metrics.tradeCount), null);
    setMetric(metricBindings.finalEquity, formatCurrency(metrics.finalEquity, 0), metrics.finalEquity - result.config.initialCapital);
    setMetric(metricBindings.buyHoldReturn, formatPercent(metrics.buyHoldReturn), metrics.buyHoldReturn);
    document.getElementById("metric-annualized").textContent = `年化 ${formatPercent(metrics.annualizedReturn)}`;
    document.getElementById("metric-win-rate").textContent = `完成交易胜率 ${formatPercent(metrics.winRate, 1)}`;
    document.getElementById("metric-holding").textContent = `持仓占比 ${formatPercent(metrics.holdingRatio, 1)}`;
  }

  function setMetric(element, text, signValue) {
    element.textContent = text;
    element.classList.remove("positive", "negative");
    if (signValue > 0) element.classList.add("positive");
    if (signValue < 0) element.classList.add("negative");
  }

  function updateInterpretations(result) {
    const { config, metrics, points, trades } = result;
    const buySignals = points.filter((point) => point.signal === 1).length;
    const sellSignals = points.filter((point) => point.signal === -1).length;
    const lastPoint = points[points.length - 1];
    const alignment = lastPoint.smaShort !== null && lastPoint.smaLong !== null
      ? (lastPoint.smaShort > lastPoint.smaLong ? "短均线位于长均线上方，趋势信号偏强" : "短均线位于长均线下方，趋势信号偏弱")
      : "当前窗口末端均线尚未形成完整比较";

    document.getElementById("price-interpretation").innerHTML = `<strong>解读：</strong>窗口内出现 ${buySignals} 次金叉、${sellSignals} 次死叉；${alignment}。信号于收盘确认，并在下一交易日开盘执行。`;

    const excess = metrics.cumulativeReturn - metrics.buyHoldReturn;
    const comparison = excess >= 0
      ? `策略较买入持有高 ${formatPercent(Math.abs(excess))}`
      : `策略较买入持有低 ${formatPercent(Math.abs(excess))}`;
    document.getElementById("nav-interpretation").innerHTML = `<strong>解读：</strong>策略累计回报 ${formatPercent(metrics.cumulativeReturn)}，买入持有为 ${formatPercent(metrics.buyHoldReturn)}，${comparison}。`;

    const mddPoint = points[metrics.maxDrawdownIndex];
    document.getElementById("drawdown-interpretation").innerHTML = `<strong>解读：</strong>最大回撤为 ${formatPercent(metrics.maxDrawdown)}，低点出现在 ${mddPoint.date}；该值表示策略从此前净值高点承受的最大资金回落。`;

    let summary;
    if (trades.length === 0 && buySignals > 0) {
      const minimumLotCost = Math.min(...points.map((point) => Number(point.open))) * config.lotSize;
      summary = `当前参数产生了金叉，但没有实际成交。按 A 股 100 股一手估算，最低一手约需 ${formatCurrency(minimumLotCost, 0)}，可提高初始资金或更换标的后比较。`;
    } else if (trades.length === 0) {
      summary = "当前窗口没有形成可执行的完整交易，指标保持在初始状态。可调整时间窗口或均线周期观察信号变化。";
    } else {
      const returnText = metrics.cumulativeReturn >= 0 ? "取得正累计回报" : "录得累计亏损";
      const sharpeText = metrics.sharpeRatio >= 1
        ? "风险调整后表现较好"
        : metrics.sharpeRatio > 0
          ? "风险调整后收益为正，但稳定性仍有限"
          : "单位波动未带来正收益";
      summary = `${config.stock.name}在当前参数下${returnText} ${formatPercent(metrics.cumulativeReturn)}，最大回撤 ${formatPercent(metrics.maxDrawdown)}，夏普比率 ${metrics.sharpeRatio.toFixed(2)}，${sharpeText}。`;
    }
    resultSummary.textContent = summary;
  }

  function renderCharts(result) {
    renderPriceChart(result);
    renderNavChart(result);
    renderDrawdownChart(result);
  }

  function renderPriceChart(result) {
    const width = chartWidth(priceChart);
    const height = width < 620 ? 340 : 390;
    const margin = { top: 26, right: 22, bottom: 42, left: 62 };
    const values = [];
    for (const point of result.points) {
      values.push(Number(point.close));
      if (point.smaShort !== null) values.push(point.smaShort);
      if (point.smaLong !== null) values.push(point.smaLong);
    }
    const scale = createScale(result.points.length, values, width, height, margin, 0.06);
    const colors = chartColors();
    const pricePath = linePath(result.points, (point) => Number(point.close), scale);
    const shortPath = linePath(result.points, (point) => point.smaShort, scale);
    const longPath = linePath(result.points, (point) => point.smaLong, scale);
    const marks = result.points.map((point, index) => {
      const x = scale.x(index);
      const y = scale.y(Number(point.close));
      if (point.signal === 1) {
        return `<path d="M ${x} ${y - 9} L ${x - 6} ${y + 3} L ${x + 6} ${y + 3} Z" fill="${colors.positive}" stroke="${colors.surface}" stroke-width="1.5"/>`;
      }
      if (point.signal === -1) {
        return `<path d="M ${x - 5} ${y - 5} L ${x + 5} ${y + 5} M ${x + 5} ${y - 5} L ${x - 5} ${y + 5}" fill="none" stroke="${colors.negative}" stroke-width="2.2"/>`;
      }
      return "";
    }).join("");

    priceChart.innerHTML = `${svgShell({
      id: "price",
      width,
      height,
      margin,
      scale,
      points: result.points,
      colors,
      title: "股价、短周期均线、长周期均线及交易信号",
      description: "收盘价与双均线随日期变化，三角形表示金叉买入信号，叉号表示死叉卖出信号。",
      body: `
        <path d="${pricePath}" fill="none" stroke="${colors.price}" stroke-width="1.8" vector-effect="non-scaling-stroke"/>
        <path d="${shortPath}" fill="none" stroke="${colors.short}" stroke-width="2" vector-effect="non-scaling-stroke"/>
        <path d="${longPath}" fill="none" stroke="${colors.long}" stroke-width="2" vector-effect="non-scaling-stroke"/>
        ${marks}
      `,
      yFormatter: (value) => value.toFixed(2),
    })}<div class="chart-tooltip" aria-hidden="true"></div>`;
    bindTooltip(priceChart, result, "price", { width, height, margin, scale });
  }

  function renderNavChart(result) {
    const width = chartWidth(navChart);
    const height = width < 620 ? 300 : 310;
    const margin = { top: 24, right: 22, bottom: 42, left: 62 };
    const values = result.points.flatMap((point) => [point.strategyNav, point.benchmarkNav]);
    const scale = createScale(result.points.length, values, width, height, margin, 0.06);
    const colors = chartColors();
    const strategyPath = linePath(result.points, (point) => point.strategyNav, scale);
    const benchmarkPath = linePath(result.points, (point) => point.benchmarkNav, scale);

    navChart.innerHTML = `${svgShell({
      id: "nav",
      width,
      height,
      margin,
      scale,
      points: result.points,
      colors,
      title: "策略净值与买入持有基准",
      description: "策略净值和买入持有基准净值的时间序列对比。",
      body: `
        <path d="${strategyPath}" fill="none" stroke="${colors.short}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>
        <path d="${benchmarkPath}" fill="none" stroke="${colors.benchmark}" stroke-width="1.8" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>
      `,
      yFormatter: (value) => value.toFixed(2),
    })}<div class="chart-tooltip" aria-hidden="true"></div>`;
    bindTooltip(navChart, result, "nav", { width, height, margin, scale });
  }

  function renderDrawdownChart(result) {
    const width = chartWidth(drawdownChart);
    const height = width < 620 ? 280 : 290;
    const margin = { top: 24, right: 22, bottom: 42, left: 62 };
    const values = result.points.map((point) => point.drawdown).concat([0]);
    const scale = createScale(result.points.length, values, width, height, margin, 0.04);
    const colors = chartColors();
    const drawdownPath = linePath(result.points, (point) => point.drawdown, scale);
    const firstX = scale.x(0);
    const lastX = scale.x(result.points.length - 1);
    const zeroY = scale.y(0);
    const areaPath = `${drawdownPath} L ${lastX} ${zeroY} L ${firstX} ${zeroY} Z`;
    const mddPoint = result.points[result.metrics.maxDrawdownIndex];
    const mddX = scale.x(result.metrics.maxDrawdownIndex);
    const mddY = scale.y(mddPoint.drawdown);

    drawdownChart.innerHTML = `${svgShell({
      id: "drawdown",
      width,
      height,
      margin,
      scale,
      points: result.points,
      colors,
      title: "策略每日回撤",
      description: "策略净值相对历史最高点的回撤序列，最低点为最大回撤。",
      body: `
        <path d="${areaPath}" fill="${colors.negativeSoft}" stroke="none"/>
        <path d="${drawdownPath}" fill="none" stroke="${colors.drawdown}" stroke-width="2" vector-effect="non-scaling-stroke"/>
        <circle cx="${mddX}" cy="${mddY}" r="4.5" fill="${colors.drawdown}" stroke="${colors.surface}" stroke-width="1.5"/>
      `,
      yFormatter: (value) => formatPercent(value, 0),
    })}<div class="chart-tooltip" aria-hidden="true"></div>`;
    bindTooltip(drawdownChart, result, "drawdown", { width, height, margin, scale });
  }

  function svgShell(options) {
    const { id, width, height, margin, scale, points, colors, title, description, body, yFormatter } = options;
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const yTicks = 5;
    let grid = "";
    for (let tick = 0; tick < yTicks; tick += 1) {
      const value = scale.minY + ((scale.maxY - scale.minY) * tick) / (yTicks - 1);
      const y = scale.y(value);
      grid += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}" stroke="${colors.grid}" stroke-width="1"/>`;
      grid += `<text x="${margin.left - 9}" y="${y + 4}" text-anchor="end" fill="${colors.muted}" font-size="11">${yFormatter(value)}</text>`;
    }
    const tickIndexes = uniqueIndexes(points.length, 5);
    let xAxis = "";
    for (const index of tickIndexes) {
      const x = scale.x(index);
      xAxis += `<line x1="${x}" x2="${x}" y1="${height - margin.bottom}" y2="${height - margin.bottom + 5}" stroke="${colors.border}"/>`;
      xAxis += `<text x="${x}" y="${height - 14}" text-anchor="middle" fill="${colors.muted}" font-size="11">${shortDate(points[index].date)}</text>`;
    }
    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc">
        <title id="${id}-title">${title}</title>
        <desc id="${id}-desc">${description}</desc>
        ${grid}
        <line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" stroke="${colors.border}"/>
        ${xAxis}
        ${body}
        <line class="chart-crosshair" x1="0" x2="0" y1="${margin.top}" y2="${height - margin.bottom}" stroke="${colors.muted}" stroke-width="1" stroke-dasharray="3 3" visibility="hidden"/>
        <rect class="chart-overlay" x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" fill="transparent"/>
      </svg>`;
  }

  function createScale(count, rawValues, width, height, margin, paddingRatio) {
    const values = rawValues.filter((value) => Number.isFinite(value));
    let minY = Math.min(...values);
    let maxY = Math.max(...values);
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const padding = (maxY - minY) * paddingRatio;
    minY -= padding;
    maxY += padding;
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    return {
      minY,
      maxY,
      x: (index) => margin.left + (plotWidth * index) / Math.max(count - 1, 1),
      y: (value) => margin.top + plotHeight - ((value - minY) / (maxY - minY)) * plotHeight,
    };
  }

  function linePath(points, accessor, scale) {
    let path = "";
    let drawing = false;
    for (let index = 0; index < points.length; index += 1) {
      const value = accessor(points[index]);
      if (value === null || !Number.isFinite(value)) {
        drawing = false;
        continue;
      }
      path += `${drawing ? " L" : "M"} ${scale.x(index)} ${scale.y(value)}`;
      drawing = true;
    }
    return path;
  }

  function bindTooltip(container, result, type, dimensions) {
    const svg = container.querySelector("svg");
    const overlay = container.querySelector(".chart-overlay");
    const crosshair = container.querySelector(".chart-crosshair");
    const tooltip = container.querySelector(".chart-tooltip");
    if (!svg || !overlay || !tooltip || !crosshair) return;

    overlay.addEventListener("pointermove", function (event) {
      const svgRect = svg.getBoundingClientRect();
      const svgX = ((event.clientX - svgRect.left) / svgRect.width) * dimensions.width;
      const plotWidth = dimensions.width - dimensions.margin.left - dimensions.margin.right;
      const ratio = clamp((svgX - dimensions.margin.left) / plotWidth, 0, 1);
      const index = Math.round(ratio * (result.points.length - 1));
      const point = result.points[index];
      const x = dimensions.scale.x(index);
      crosshair.setAttribute("x1", x);
      crosshair.setAttribute("x2", x);
      crosshair.setAttribute("visibility", "visible");
      tooltip.innerHTML = tooltipContent(point, type);
      tooltip.classList.add("visible");

      const containerRect = container.getBoundingClientRect();
      const preferredLeft = event.clientX - containerRect.left + 12;
      const preferredTop = event.clientY - containerRect.top - 10;
      const maxLeft = container.clientWidth - tooltip.offsetWidth - 8;
      const maxTop = container.clientHeight - tooltip.offsetHeight - 8;
      tooltip.style.left = `${clamp(preferredLeft, 8, Math.max(8, maxLeft))}px`;
      tooltip.style.top = `${clamp(preferredTop, 8, Math.max(8, maxTop))}px`;
    });

    overlay.addEventListener("pointerleave", function () {
      crosshair.setAttribute("visibility", "hidden");
      tooltip.classList.remove("visible");
    });
  }

  function tooltipContent(point, type) {
    if (type === "price") {
      const shortValue = point.smaShort === null ? "--" : point.smaShort.toFixed(2);
      const longValue = point.smaLong === null ? "--" : point.smaLong.toFixed(2);
      return `<strong>${point.date}</strong><br>收盘 ${Number(point.close).toFixed(2)}<br>短 SMA ${shortValue}<br>长 SMA ${longValue}`;
    }
    if (type === "nav") {
      return `<strong>${point.date}</strong><br>策略净值 ${point.strategyNav.toFixed(3)}<br>买入持有 ${point.benchmarkNav.toFixed(3)}`;
    }
    return `<strong>${point.date}</strong><br>回撤 ${formatPercent(point.drawdown)}`;
  }

  function renderTrades(result) {
    const body = document.getElementById("trades-body");
    const recentTrades = result.trades.slice().reverse().slice(0, 12);
    if (recentTrades.length === 0) {
      body.innerHTML = '<tr><td class="empty-row" colspan="6">当前参数下暂无实际成交记录</td></tr>';
      return;
    }
    body.innerHTML = recentTrades.map((trade) => `
      <tr>
        <td>${trade.date}</td>
        <td><span class="trade-action ${trade.action === "BUY" ? "buy" : "sell"}">${trade.action === "BUY" ? "买入" : "卖出"}</span></td>
        <td>${trade.price.toFixed(2)}</td>
        <td>${formatNumber(trade.shares, 0)}</td>
        <td>${formatCurrency(trade.fee, 2)}</td>
        <td>${formatCurrency(trade.cashAfter, 2)}</td>
      </tr>`).join("");
  }

  function chartColors() {
    const styles = getComputedStyle(document.documentElement);
    return {
      surface: styles.getPropertyValue("--surface").trim(),
      muted: styles.getPropertyValue("--muted").trim(),
      border: styles.getPropertyValue("--border").trim(),
      grid: styles.getPropertyValue("--grid").trim(),
      price: styles.getPropertyValue("--series-price").trim(),
      short: styles.getPropertyValue("--series-short").trim(),
      long: styles.getPropertyValue("--series-long").trim(),
      benchmark: styles.getPropertyValue("--series-benchmark").trim(),
      drawdown: styles.getPropertyValue("--series-drawdown").trim(),
      positive: styles.getPropertyValue("--positive").trim(),
      negative: styles.getPropertyValue("--negative").trim(),
      negativeSoft: styles.getPropertyValue("--negative-soft").trim(),
    };
  }

  function chartWidth(container) {
    return Math.max(320, Math.floor(container.clientWidth || 900));
  }

  function uniqueIndexes(length, count) {
    const indexes = [];
    for (let index = 0; index < count; index += 1) {
      indexes.push(Math.round(((length - 1) * index) / Math.max(count - 1, 1)));
    }
    return [...new Set(indexes)];
  }

  function shortDate(value) {
    return value.slice(2, 4) + "-" + value.slice(5, 7) + "-" + value.slice(8, 10);
  }

  function average(values) {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function formatPercent(value, digits = 2) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatCurrency(value, digits = 0) {
    return `¥${formatNumber(value, digits)}`;
  }

  function formatNumber(value, digits = 0) {
    return Number(value).toLocaleString("zh-CN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  initialize();
})();
