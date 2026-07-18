"""Benchmark six classifiers and generate a complete HTML evaluation report."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from string import Template
from time import perf_counter

# Keep Matplotlib's cache inside TASK5 so the script works with a read-only home.
TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"
MATPLOTLIB_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "target"

MODEL_LABELS = {
    "Logistic Regression": "逻辑回归",
    "Decision Tree": "决策树",
    "Random Forest": "随机森林",
    "SVM (RBF)": "支持向量机（RBF）",
    "KNN": "K近邻",
    "Gaussian Naive Bayes": "高斯朴素贝叶斯",
}

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "Decision Tree": "#7c3aed",
    "Random Forest": "#f59e0b",
    "SVM (RBF)": "#0f766e",
    "KNN": "#db2777",
    "Gaussian Naive Bayes": "#64748b",
}

MODEL_DETAILS = {
    "Logistic Regression": {
        "short": "可解释的线性概率基准",
        "principle": "计算特征的线性组合，再通过 Sigmoid 函数映射为 0～1 的概率。",
        "strength": "训练快、输出概率、参数含义清晰，适合作为分类任务的基准模型。",
        "weakness": "主要学习线性决策边界，复杂非线性关系通常需要特征工程。",
        "setup": "StandardScaler + class_weight='balanced'",
    },
    "Decision Tree": {
        "short": "直观的非线性规则模型",
        "principle": "按能最大程度降低节点不纯度的特征和切分点递归划分样本。",
        "strength": "规则直观、可解释，能够学习非线性关系，且无须特征标准化。",
        "weakness": "单棵树容易过拟合，对训练样本变化较敏感。",
        "setup": "max_depth=4, min_samples_leaf=5",
    },
    "Random Forest": {
        "short": "多棵随机决策树集成",
        "principle": "通过 Bootstrap 抽样训练多棵树，并在随机特征子集上分裂，最终投票。",
        "strength": "准确稳定、非线性能力强、抗异常值，并可评估特征重要性。",
        "weakness": "模型体积和计算成本高于单棵树，整体规则不易直接解释。",
        "setup": "300 trees + class_weight='balanced'",
    },
    "SVM (RBF)": {
        "short": "最大间隔非线性分类器",
        "principle": "寻找分类间隔最大的边界，RBF 核将样本映射到可处理非线性关系的空间。",
        "strength": "适合中小规模和高维数据，复杂边界下通常表现良好。",
        "weakness": "大样本训练较慢，参数和预测结果不如线性模型容易解释。",
        "setup": "StandardScaler + RBF + probability=True",
    },
    "KNN": {
        "short": "邻近样本多数投票",
        "principle": "寻找距离新样本最近的 K 个训练样本，根据距离加权投票确定类别。",
        "strength": "原理简单，不预设线性形式，能够拟合非线性分类边界。",
        "weakness": "预测较慢，对特征尺度、无关特征和 K 值敏感。",
        "setup": "StandardScaler + K=7 + distance weights",
    },
    "Gaussian Naive Bayes": {
        "short": "基于条件概率的快速模型",
        "principle": "利用贝叶斯公式计算后验概率，并假设给定类别后各特征条件独立且服从高斯分布。",
        "strength": "训练和预测很快，在小样本、高维任务中可作为有效基准。",
        "weakness": "特征独立与高斯分布假设较强，相关特征较多时可能限制效果。",
        "setup": "StandardScaler + GaussianNB",
    },
}


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and validate numeric features and a binary target."""
    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The target column '{TARGET_COLUMN}' was not found.")

    target = data[TARGET_COLUMN]
    features = data.drop(columns=TARGET_COLUMN)
    if target.isna().any() or features.isna().any().any():
        raise ValueError("The data contains missing values. Please handle them first.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("The target column must contain exactly labels 0 and 1.")

    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    return features, target.astype(int)


def scaled_pipeline(model: object) -> Pipeline:
    """Build a leakage-safe standardization and classification pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_models() -> dict[str, object]:
    """Create six commonly used classification models."""
    return {
        "Logistic Regression": scaled_pipeline(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": scaled_pipeline(
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )
        ),
        "KNN": scaled_pipeline(
            KNeighborsClassifier(n_neighbors=7, weights="distance")
        ),
        "Gaussian Naive Bayes": scaled_pipeline(GaussianNB()),
    }


def plot_roc_curves(
    roc_points: dict[str, tuple], output_path: Path
) -> None:
    plt.figure(figsize=(9.5, 7.2))
    for model_name, (fpr, tpr, auc_value) in roc_points.items():
        plt.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model_name],
            linewidth=2.2,
            label=f"{model_name} (AUC={auc_value:.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.3, label="Random classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("ROC Curves: Six Classification Models")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.22)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curves(
    pr_points: dict[str, tuple], positive_rate: float, output_path: Path
) -> None:
    plt.figure(figsize=(9.5, 7.2))
    for model_name, (recall, precision, ap_value) in pr_points.items():
        plt.plot(
            recall,
            precision,
            color=MODEL_COLORS[model_name],
            linewidth=2.2,
            label=f"{model_name} (AP={ap_value:.4f})",
        )
    plt.axhline(
        positive_rate,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Positive rate ({positive_rate:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: Six Classification Models")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.22)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics.sort_values("auc", ascending=False)
    columns = ["accuracy", "f1", "auc", "average_precision"]
    labels = ["Accuracy", "F1", "ROC-AUC", "AP"]
    colors = ["#60a5fa", "#a78bfa", "#f59e0b", "#14b8a6"]
    x_positions = list(range(len(ordered)))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for index, (column, label, color) in enumerate(zip(columns, labels, colors)):
        offsets = [x + (index - 1.5) * width for x in x_positions]
        ax.bar(offsets, ordered[column], width=width, label=label, color=color)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered["model"], rotation=18, ha="right")
    ax.set_ylim(0.75, 1.01)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Classification Performance Comparison")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(
    confusion_matrices: dict[str, list[list[int]]], output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    for ax, (model_name, matrix) in zip(axes.ravel(), confusion_matrices.items()):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        threshold = max(max(row) for row in matrix) / 2
        for row_index in range(2):
            for column_index in range(2):
                value = matrix[row_index][column_index]
                ax.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "#102a43",
                    fontsize=14,
                    fontweight="bold",
                )
        ax.set_title(model_name, fontsize=11)
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        image.set_clim(0, max(max(row) for row in matrix))
    fig.suptitle("Confusion Matrices on the Test Set", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_metrics_table(metrics: pd.DataFrame) -> str:
    rows = []
    for _, row in metrics.iterrows():
        best_class = ' class="best-row"' if int(row["auc_rank"]) == 1 else ""
        rows.append(
            f"""<tr{best_class}>
              <td><span class="rank">{int(row['auc_rank'])}</span></td>
              <td class="model-name">{MODEL_LABELS[str(row['model'])]}</td>
              <td>{row['accuracy']:.4f}</td><td>{row['precision']:.4f}</td>
              <td>{row['recall']:.4f}</td><td>{row['specificity']:.4f}</td>
              <td>{row['f1']:.4f}</td><td><strong>{row['auc']:.4f}</strong></td>
              <td>{row['average_precision']:.4f}</td><td>{row['log_loss']:.4f}</td>
              <td>{row['fit_time_ms']:.1f}</td>
            </tr>"""
        )
    return "\n".join(rows)


def build_confusion_table(metrics: pd.DataFrame) -> str:
    return "\n".join(
        f"""<tr>
          <td class="model-name">{MODEL_LABELS[str(row['model'])]}</td>
          <td>{int(row['tn'])}</td><td>{int(row['fp'])}</td>
          <td>{int(row['fn'])}</td><td>{int(row['tp'])}</td>
        </tr>"""
        for _, row in metrics.iterrows()
    )


def build_algorithm_cards(metrics: pd.DataFrame) -> str:
    by_model = metrics.set_index("model")
    cards = []
    for index, (model_name, details) in enumerate(MODEL_DETAILS.items(), start=1):
        result = by_model.loc[model_name]
        cards.append(
            f"""<article class="algorithm-card" style="--model-color:{MODEL_COLORS[model_name]}">
              <div class="algorithm-top"><span class="algorithm-number">0{index}</span>
                <div><h3>{MODEL_LABELS[model_name]}</h3><p>{details['short']}</p></div></div>
              <div class="score-strip"><span>AUC <strong>{result['auc']:.4f}</strong></span>
                <span>F1 <strong>{result['f1']:.4f}</strong></span>
                <span>AP <strong>{result['average_precision']:.4f}</strong></span></div>
              <p><b>原理：</b>{details['principle']}</p>
              <p><b>优点：</b>{details['strength']}</p>
              <p><b>局限：</b>{details['weakness']}</p>
              <p class="setup"><b>本次配置：</b>{details['setup']}</p>
            </article>"""
        )
    return "\n".join(cards)


def write_html_report(
    output_path: Path,
    metrics: pd.DataFrame,
    sample_count: int,
    feature_count: int,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Create a complete Chinese learning and model-comparison report."""
    best_auc = metrics.loc[metrics["auc"].idxmax()]
    best_f1 = metrics.loc[metrics["f1"].idxmax()]
    best_ap = metrics.loc[metrics["average_precision"].idxmax()]
    best_log_loss = metrics.loc[metrics["log_loss"].idxmin()]
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    template = Template(r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TASK5｜分类机器学习模型对比报告</title>
  <style>
    :root {
      --ink:#172033; --muted:#64748b; --line:#dce4ee; --paper:#ffffff;
      --canvas:#f4f7fb; --navy:#0d2742; --blue:#2563eb; --cyan:#0e7490;
      --amber:#f59e0b; --green:#0f766e; --soft-blue:#eaf2ff;
    }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; color:var(--ink); background:var(--canvas); font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif; line-height:1.72; }
    .topbar { position:sticky; top:0; z-index:10; background:rgba(13,39,66,.96); backdrop-filter:blur(10px); }
    .topbar-inner { max-width:1180px; margin:auto; padding:11px 24px; display:flex; align-items:center; justify-content:space-between; gap:22px; }
    .brand { color:white; font-weight:800; letter-spacing:.04em; white-space:nowrap; }
    nav { display:flex; gap:20px; overflow:auto; }
    nav a { color:#d7e7f7; text-decoration:none; font-size:14px; white-space:nowrap; }
    nav a:hover { color:white; }
    main { max-width:1180px; margin:auto; padding:28px 24px 64px; }
    .hero { color:white; overflow:hidden; position:relative; padding:56px 58px; border-radius:22px; background:linear-gradient(125deg,#0b2340 0%,#114b72 58%,#0e7490 100%); box-shadow:0 18px 42px rgba(18,53,82,.18); }
    .hero::after { content:""; position:absolute; width:320px; height:320px; right:-90px; top:-140px; border:70px solid rgba(255,255,255,.08); border-radius:50%; }
    .eyebrow { display:inline-block; padding:5px 10px; border:1px solid rgba(255,255,255,.35); border-radius:99px; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }
    h1 { max-width:820px; margin:18px 0 12px; font-size:clamp(32px,5vw,53px); line-height:1.15; letter-spacing:-.025em; }
    .hero>p { max-width:790px; margin:0; color:#d8e8f5; font-size:17px; }
    .hero-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:32px; }
    .hero-stat { padding:14px 16px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.14); border-radius:12px; }
    .hero-stat span { display:block; color:#c7deef; font-size:12px; }
    .hero-stat strong { display:block; margin-top:2px; font-size:21px; }
    section { margin-top:24px; padding:34px 38px; background:var(--paper); border:1px solid var(--line); border-radius:18px; box-shadow:0 7px 24px rgba(32,56,85,.045); }
    .section-kicker { margin:0 0 4px; color:var(--cyan); font-size:12px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    h2 { margin:0 0 10px; color:var(--navy); font-size:27px; line-height:1.3; }
    h3 { margin:0; color:var(--navy); }
    .section-intro { max-width:850px; margin:0 0 24px; color:var(--muted); }
    .dataset-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
    .data-card { padding:19px; background:#f7faff; border:1px solid #e3eaf3; border-radius:13px; }
    .data-card span { display:block; color:var(--muted); font-size:13px; }
    .data-card strong { display:block; margin-top:4px; color:var(--navy); font-size:25px; }
    .method-flow { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:18px; }
    .method-step { padding:16px; border-top:3px solid var(--blue); background:var(--soft-blue); border-radius:8px 8px 12px 12px; }
    .method-step b { display:block; margin-bottom:5px; color:var(--blue); }
    .method-step span { color:#40516a; font-size:13px; }
    .algorithm-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
    .algorithm-card { padding:22px; border:1px solid var(--line); border-top:4px solid var(--model-color); border-radius:13px; background:#fff; }
    .algorithm-top { display:flex; align-items:center; gap:13px; }
    .algorithm-number { color:var(--model-color); font-size:24px; font-weight:900; }
    .algorithm-top p { margin:1px 0 0; color:var(--muted); font-size:13px; }
    .algorithm-card>p { margin:10px 0; font-size:14px; color:#40516a; }
    .score-strip { display:flex; gap:9px; margin:15px 0 13px; flex-wrap:wrap; }
    .score-strip span { padding:5px 9px; color:var(--model-color); background:#f4f7fb; border-radius:6px; font-size:12px; }
    .setup { padding-top:10px; border-top:1px dashed var(--line); }
    .result-banner { display:grid; grid-template-columns:1.4fr repeat(3,1fr); gap:12px; margin:18px 0 24px; }
    .result-card { padding:17px; border:1px solid var(--line); border-radius:12px; background:#f8fafc; }
    .result-card.primary { color:white; background:linear-gradient(125deg,#1d4ed8,#0e7490); border:0; }
    .result-card span { display:block; font-size:12px; opacity:.75; }
    .result-card strong { display:block; margin-top:4px; font-size:19px; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; }
    table { width:100%; min-width:930px; border-collapse:collapse; text-align:center; font-variant-numeric:tabular-nums; }
    th { padding:11px 10px; color:#dfeefa; background:var(--navy); font-size:12px; white-space:nowrap; }
    td { padding:11px 10px; border-bottom:1px solid var(--line); font-size:13px; }
    tbody tr:last-child td { border-bottom:0; }
    tbody tr:hover td { background:#f5f9ff; }
    .best-row td { background:#fff8e8; }
    .model-name { font-weight:700; text-align:left; white-space:nowrap; }
    .rank { display:inline-grid; place-items:center; width:25px; height:25px; color:white; background:#64748b; border-radius:50%; font-size:12px; }
    .best-row .rank { background:var(--amber); }
    .note { margin:12px 0 0; color:var(--muted); font-size:13px; }
    .visual-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .figure-card { margin:0; padding:14px; border:1px solid var(--line); border-radius:13px; background:#fbfdff; }
    .figure-card.wide { grid-column:1/-1; }
    .figure-card img { display:block; width:100%; height:auto; border-radius:7px; }
    figcaption { padding:9px 8px 2px; color:var(--muted); font-size:13px; }
    .metric-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:13px; }
    .metric-card { padding:18px; background:#f8fafc; border:1px solid var(--line); border-radius:11px; }
    .metric-card h3 { font-size:16px; }
    .formula { display:block; width:max-content; max-width:100%; margin:8px 0; padding:5px 9px; color:#174a75; background:#e8f2fb; border-radius:6px; font-family:Consolas,monospace; font-size:12px; }
    .metric-card p { margin:6px 0 0; color:#4f6074; font-size:13px; }
    .matrix-demo { display:grid; grid-template-columns:1.1fr 1fr; gap:18px; align-items:start; margin-bottom:18px; }
    .matrix-table { min-width:0; }
    .matrix-table th,.matrix-table td { padding:10px; }
    .matrix-table .tp,.matrix-table .tn { background:#e6f6ef; color:#096446; }
    .matrix-table .fp,.matrix-table .fn { background:#fff0ed; color:#a33a24; }
    .callout { padding:17px 19px; color:#4b3b0b; background:#fff7dd; border-left:4px solid var(--amber); border-radius:8px; }
    .conclusion-list { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin:18px 0 0; padding:0; list-style:none; }
    .conclusion-list li { padding:16px; background:#f7faff; border:1px solid var(--line); border-radius:10px; }
    footer { padding:26px; color:var(--muted); text-align:center; font-size:12px; }
    @media (max-width:900px) {
      .hero-stats,.dataset-grid,.method-flow { grid-template-columns:repeat(2,1fr); }
      .result-banner { grid-template-columns:repeat(2,1fr); }
      .visual-grid,.matrix-demo { grid-template-columns:1fr; }
    }
    @media (max-width:650px) {
      .topbar-inner { align-items:flex-start; flex-direction:column; gap:6px; }
      main { padding:16px 12px 44px; }
      .hero { padding:34px 24px; }
      section { padding:26px 19px; }
      .algorithm-grid,.metric-grid,.conclusion-list { grid-template-columns:1fr; }
      .hero-stats,.dataset-grid,.method-flow,.result-banner { grid-template-columns:1fr 1fr; }
    }
    @media print { .topbar { display:none; } body { background:white; } section,.hero { break-inside:avoid; box-shadow:none; } }
  </style>
</head>
<body>
  <div class="topbar"><div class="topbar-inner"><div class="brand">AIQuant · TASK5</div>
    <nav><a href="#data">实验设计</a><a href="#models">算法说明</a><a href="#results">性能对比</a><a href="#curves">可视化</a><a href="#metrics">指标解释</a><a href="#conclusion">结论</a></nav>
  </div></div>
  <main>
    <header class="hero">
      <span class="eyebrow">Classification Benchmark · 6 Models</span>
      <h1>分类机器学习模型<br>训练与性能对比报告</h1>
      <p>在同一乳腺癌数据集与测试集上，对逻辑回归、决策树、随机森林、SVM、KNN 和朴素贝叶斯进行统一评估。</p>
      <div class="hero-stats">
        <div class="hero-stat"><span>模型数量</span><strong>6</strong></div>
        <div class="hero-stat"><span>最佳模型（AUC）</span><strong>$best_auc_model</strong></div>
        <div class="hero-stat"><span>最佳 AUC</span><strong>$best_auc</strong></div>
        <div class="hero-stat"><span>测试样本</span><strong>$test_count</strong></div>
      </div>
    </header>

    <section id="data">
      <p class="section-kicker">01 · Experiment</p><h2>数据与实验设计</h2>
      <p class="section-intro">分类学习根据带标签样本学习特征与类别之间的关系。本实验将 <code>target=1</code> 视为正类，所有模型共享完全相同的分层训练集和测试集。</p>
      <div class="dataset-grid">
        <div class="data-card"><span>总样本数</span><strong>$sample_count</strong></div>
        <div class="data-card"><span>输入特征</span><strong>$feature_count</strong></div>
        <div class="data-card"><span>训练集</span><strong>$train_count</strong></div>
        <div class="data-card"><span>测试集</span><strong>$test_count</strong></div>
      </div>
      <div class="method-flow">
        <div class="method-step"><b>① 加载与检查</b><span>无缺失值、因变量严格为 0/1、全部特征为数值型。</span></div>
        <div class="method-step"><b>② 分层划分</b><span>80% 训练、20% 测试；random_state=42，保持类别比例。</span></div>
        <div class="method-step"><b>③ 预处理</b><span>LR、SVM、KNN、NB 在流水线内标准化；树模型不缩放。</span></div>
        <div class="method-step"><b>④ 统一测试</b><span>同一测试集计算分类、排序和概率质量指标。</span></div>
      </div>
      <p class="note">类别分布：训练集 0 类 $train_zero 个、1 类 $train_one 个；测试集 0 类 $test_zero 个、1 类 $test_one 个。</p>
    </section>

    <section id="models">
      <p class="section-kicker">02 · Algorithms</p><h2>六种分类算法：原理、优点与局限</h2>
      <p class="section-intro">每张卡片同时展示算法知识与本次真实测试结果，方便把理论特征和实际性能联系起来。</p>
      <div class="algorithm-grid">$algorithm_cards</div>
    </section>

    <section id="results">
      <p class="section-kicker">03 · Benchmark</p><h2>模型性能总表</h2>
      <p class="section-intro">表格按 ROC-AUC 从高到低排序。Accuracy、Precision、Recall、Specificity、F1、AUC、AP 均为越高越好；Log Loss 越低越好。</p>
      <div class="result-banner">
        <div class="result-card primary"><span>ROC-AUC 最佳</span><strong>$best_auc_model · $best_auc</strong></div>
        <div class="result-card"><span>F1 最佳</span><strong>$best_f1_model · $best_f1</strong></div>
        <div class="result-card"><span>AP 最佳</span><strong>$best_ap_model · $best_ap</strong></div>
        <div class="result-card"><span>Log Loss 最低</span><strong>$best_ll_model · $best_ll</strong></div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>AUC排名</th><th>模型</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>AUC</th><th>AP</th><th>Log Loss↓</th><th>训练ms*</th></tr></thead>
        <tbody>$metrics_rows</tbody>
      </table></div>
      <p class="note">* 训练耗时受本机状态影响，只用于粗略观察计算成本，不作为模型质量结论。所有其他指标来自 114 个独立测试样本。</p>
    </section>

    <section id="curves">
      <p class="section-kicker">04 · Visual Evidence</p><h2>多维度性能可视化</h2>
      <p class="section-intro">ROC 观察全阈值下 TPR 与 FPR 的权衡；PR 曲线更聚焦正类识别；混淆矩阵则展示固定阈值下具体的误报与漏报。</p>
      <div class="visual-grid">
        <figure class="figure-card wide"><img src="model_metrics_comparison.png" alt="六种模型综合指标柱状图"><figcaption>综合指标：横向比较 Accuracy、F1、ROC-AUC 和 AP。</figcaption></figure>
        <figure class="figure-card"><img src="roc_curve.png" alt="六种模型 ROC 曲线"><figcaption>ROC 曲线：越靠近左上角越好；虚线代表随机分类器。</figcaption></figure>
        <figure class="figure-card"><img src="pr_curve.png" alt="六种模型 PR 曲线"><figcaption>PR 曲线：越靠近右上区域越好；虚线表示测试集正类比例。</figcaption></figure>
        <figure class="figure-card wide"><img src="confusion_matrices.png" alt="六种模型混淆矩阵"><figcaption>混淆矩阵：逐模型查看 TN、FP、FN、TP，区分误报与漏报。</figcaption></figure>
      </div>
    </section>

    <section id="confusion">
      <p class="section-kicker">05 · Error Analysis</p><h2>混淆矩阵与错误类型</h2>
      <div class="matrix-demo">
        <table class="matrix-table"><thead><tr><th>真实情况 / 预测</th><th>预测为正</th><th>预测为负</th></tr></thead>
          <tbody><tr><td>真实为正</td><td class="tp"><b>TP</b><br>正确检出</td><td class="fn"><b>FN</b><br>漏报</td></tr>
          <tr><td>真实为负</td><td class="fp"><b>FP</b><br>误报</td><td class="tn"><b>TN</b><br>正确排除</td></tr></tbody></table>
        <div class="callout"><b>如何理解：</b><br>TP 和 TN 是正确预测；FP 是把负类误判为正类；FN 是把正类漏判为负类。疾病筛查通常更在意 FN，因此会重点关注 Recall。</div>
      </div>
      <div class="table-wrap"><table><thead><tr><th>模型</th><th>TN</th><th>FP（误报）</th><th>FN（漏报）</th><th>TP</th></tr></thead><tbody>$confusion_rows</tbody></table></div>
    </section>

    <section id="metrics">
      <p class="section-kicker">06 · Metrics</p><h2>评价指标完整解释</h2>
      <p class="section-intro">分类模型不存在对所有任务都最好的单一指标。应根据类别分布、误报/漏报成本及是否关注概率质量进行组合判断。</p>
      <div class="metric-grid">
        <article class="metric-card"><h3>1. Accuracy 准确率</h3><span class="formula">(TP + TN) / (TP + TN + FP + FN)</span><p>全部预测中正确的比例，直观但在类别严重不平衡时可能产生误导。</p></article>
        <article class="metric-card"><h3>2. Precision 精确率</h3><span class="formula">TP / (TP + FP)</span><p>预测为正的样本中实际为正的比例。越高表示误报越少。</p></article>
        <article class="metric-card"><h3>3. Recall 召回率 / TPR</h3><span class="formula">TP / (TP + FN)</span><p>全部真实正类中被成功找出的比例。越高表示漏报越少。</p></article>
        <article class="metric-card"><h3>4. Specificity 特异度</h3><span class="formula">TN / (TN + FP) = 1 − FPR</span><p>全部真实负类中被正确识别为负类的比例，反映排除负类的能力。</p></article>
        <article class="metric-card"><h3>5. F1 分数</h3><span class="formula">2 × Precision × Recall / (Precision + Recall)</span><p>Precision 与 Recall 的调和平均，适合同等重视误报和漏报的任务，但不考虑 TN。</p></article>
        <article class="metric-card"><h3>6. ROC 曲线</h3><span class="formula">横轴 FPR = FP/(FP+TN)；纵轴 TPR = TP/(TP+FN)</span><p>改变分类阈值得到一系列 FPR/TPR 点。曲线越靠近左上角，分类能力越好。</p></article>
        <article class="metric-card"><h3>7. ROC-AUC</h3><span class="formula">Area Under the ROC Curve ∈ [0, 1]</span><p>衡量跨阈值的整体排序能力；可理解为随机正样本得分高于随机负样本的概率。它不等于概率校准质量。</p></article>
        <article class="metric-card"><h3>8. PR 曲线与 AP</h3><span class="formula">横轴 Recall；纵轴 Precision</span><p>类别极不平衡时通常比 ROC 更敏感。AP 汇总不同召回率下的 Precision，越高越好。</p></article>
        <article class="metric-card"><h3>9. Log Loss 交叉熵</h3><span class="formula">−mean[y·log(p) + (1−y)·log(1−p)]</span><p>衡量预测概率质量，越低越好；对“非常自信但错误”的预测惩罚很大。</p></article>
        <article class="metric-card"><h3>10. 阈值与业务成本</h3><span class="formula">默认分类阈值 = 0.5</span><p>降低阈值通常提高 Recall 但增加 FP；提高阈值可能提高 Precision 但增加 FN。最终阈值应由业务成本决定。</p></article>
      </div>
    </section>

    <section id="conclusion">
      <p class="section-kicker">07 · Conclusion</p><h2>结论与选择建议</h2>
      <div class="callout"><b>本次结果：</b>$best_auc_model 获得最高 ROC-AUC（$best_auc）。但模型选择不应只看 AUC，还应结合 F1、AP、Log Loss、误报/漏报数量、解释需求和计算成本。</div>
      <ul class="conclusion-list">
        <li><b>需要概率与解释：</b><br>优先观察逻辑回归，它是稳定且容易解释的基准。</li>
        <li><b>重视综合预测性能：</b><br>重点比较随机森林和 SVM 的 AUC、AP 与错误类型。</li>
        <li><b>需要可读规则：</b><br>决策树最直观，但要限制深度并警惕过拟合。</li>
        <li><b>正类非常稀少：</b><br>重点查看 PR 曲线、AP、Precision 和 Recall，而非只看 Accuracy。</li>
      </ul>
      <p class="note">方法限制：本报告采用固定的单次分层划分，适合教学演示和初步比较。若用于正式建模，应进一步采用分层交叉验证、超参数搜索、概率校准和独立外部测试集。</p>
    </section>
    <footer>数据文件：model_data_cancer.csv · 正类标签：1 · 生成时间：$generated_at</footer>
  </main>
</body>
</html>""")

    report = template.substitute(
        sample_count=sample_count,
        feature_count=feature_count,
        train_count=len(y_train),
        test_count=len(y_test),
        train_zero=int(train_counts.get(0, 0)),
        train_one=int(train_counts.get(1, 0)),
        test_zero=int(test_counts.get(0, 0)),
        test_one=int(test_counts.get(1, 0)),
        best_auc_model=MODEL_LABELS[str(best_auc["model"])],
        best_auc=f"{best_auc['auc']:.4f}",
        best_f1_model=MODEL_LABELS[str(best_f1["model"])],
        best_f1=f"{best_f1['f1']:.4f}",
        best_ap_model=MODEL_LABELS[str(best_ap["model"])],
        best_ap=f"{best_ap['average_precision']:.4f}",
        best_ll_model=MODEL_LABELS[str(best_log_loss["model"])],
        best_ll=f"{best_log_loss['log_loss']:.4f}",
        algorithm_cards=build_algorithm_cards(metrics),
        metrics_rows=build_metrics_table(metrics),
        confusion_rows=build_confusion_table(metrics),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    csv_path = TASK_DIR / "model_data_cancer.csv"
    OUTPUT_DIR.mkdir(exist_ok=True)
    features, target = load_data(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print(f"Data shape: {features.shape[0]} rows, {features.shape[1]} features")
    print(f"Training set: {len(x_train)} rows; test set: {len(x_test)} rows")
    print(f"Training target distribution: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().sort_index().to_dict()}\n")

    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_data = pd.DataFrame({"actual": y_test}, index=y_test.index)
    roc_points: dict[str, tuple] = {}
    pr_points: dict[str, tuple] = {}
    matrix_values: dict[str, list[list[int]]] = {}

    for model_name, model in build_models().items():
        fit_start = perf_counter()
        model.fit(x_train, y_train)
        fit_time_ms = (perf_counter() - fit_start) * 1000

        predict_start = perf_counter()
        predicted_class = model.predict(x_test)
        predicted_probability = model.predict_proba(x_test)[:, 1]
        predict_time_ms = (perf_counter() - predict_start) * 1000

        matrix = confusion_matrix(y_test, predicted_class, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        auc_value = roc_auc_score(y_test, predicted_probability)
        ap_value = average_precision_score(y_test, predicted_probability)

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predicted_class),
                "precision": precision_score(y_test, predicted_class, zero_division=0),
                "recall": recall_score(y_test, predicted_class, zero_division=0),
                "specificity": specificity,
                "f1": f1_score(y_test, predicted_class, zero_division=0),
                "auc": auc_value,
                "average_precision": ap_value,
                "log_loss": log_loss(y_test, predicted_probability, labels=[0, 1]),
                "fit_time_ms": fit_time_ms,
                "predict_time_ms": predict_time_ms,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        prediction_data[f"{safe_name}_prediction"] = predicted_class
        prediction_data[f"{safe_name}_probability_1"] = predicted_probability

        fpr, tpr, _ = roc_curve(y_test, predicted_probability)
        precision_values, recall_values, _ = precision_recall_curve(
            y_test, predicted_probability
        )
        roc_points[model_name] = (fpr, tpr, auc_value)
        pr_points[model_name] = (recall_values, precision_values, ap_value)
        matrix_values[model_name] = matrix.tolist()

    metrics = pd.DataFrame(metrics_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    metrics.insert(0, "auc_rank", range(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False, encoding="utf-8-sig")
    prediction_data.sort_index().to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index_label="sample_index",
        encoding="utf-8-sig",
    )

    plot_roc_curves(roc_points, OUTPUT_DIR / "roc_curve.png")
    plot_pr_curves(pr_points, float(y_test.mean()), OUTPUT_DIR / "pr_curve.png")
    plot_metric_comparison(metrics, OUTPUT_DIR / "model_metrics_comparison.png")
    plot_confusion_matrices(matrix_values, OUTPUT_DIR / "confusion_matrices.png")
    write_html_report(
        output_path=OUTPUT_DIR / "classification_report.html",
        metrics=metrics,
        sample_count=len(features),
        feature_count=features.shape[1],
        y_train=y_train,
        y_test=y_test,
    )

    print("Model evaluation results (sorted by AUC):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nHTML report: {OUTPUT_DIR / 'classification_report.html'}")


if __name__ == "__main__":
    main()
