from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "唐智科技湖南发展 | 诊断数据分析数据挖掘展示"
DATA_PATH = Path("price_input_foreign.csv")
GITHUB_URL = os.getenv(
    "PROJECT_GITHUB_URL",
    "https://github.com/ZHXwudi/yuce",
)
STREAMLIT_URL = os.getenv(
    "PROJECT_STREAMLIT_URL",
    "",
).strip()


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #182026;
        --muted: #5a6872;
        --line: #d8dee4;
        --panel: #ffffff;
        --accent: #0f766e;
        --warn: #b45309;
        --danger: #be123c;
    }
    .main .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 700;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.85rem;
        margin: 0.9rem 0 1.15rem 0;
    }
    .metric-card {
        min-height: 96px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.25;
    }
    .metric-value {
        color: var(--ink);
        font-size: 1.75rem;
        line-height: 1.05;
        font-weight: 760;
        white-space: nowrap;
    }
    .metric-delta {
        color: #15803d;
        font-size: 0.82rem;
        line-height: 1.25;
        white-space: normal;
    }
    .section-note {
        border-left: 4px solid var(--accent);
        padding: 0.15rem 0 0.15rem 0.85rem;
        color: var(--muted);
        font-size: 0.94rem;
        margin: 0.35rem 0 1rem 0;
    }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        color: var(--ink);
        background: #f8fafc;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .risk-high { color: var(--danger); font-weight: 700; }
    .risk-mid { color: var(--warn); font-weight: 700; }
    .small-muted { color: var(--muted); font-size: 0.88rem; }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        std = values.std(ddof=0)
        scale = std if np.isfinite(std) and std > 1e-9 else 1.0
    return (values - median) / scale


@st.cache_data(show_spinner=False)
def load_price_data(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.shape[1] < 5:
        raise ValueError("CSV 至少需要 5 列：日、月、日期、时间、电价。")

    df = raw.iloc[:, :5].copy()
    df.columns = ["day", "month", "date_text", "time_text", "price"]
    df = df.dropna(how="all")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date_text", "time_text", "price"]).copy()

    timestamp = pd.to_datetime(
        df["date_text"].astype(str).str.strip() + " " + df["time_text"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    if timestamp.isna().any():
        fallback = pd.to_datetime(
            df["date_text"].astype(str).str.strip() + " " + df["time_text"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        timestamp = timestamp.fillna(fallback)

    df["timestamp"] = timestamp
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    return df.reset_index(drop=True)


def enrich_diagnostics(df: pd.DataFrame, anomaly_percentile: int) -> tuple[pd.DataFrame, float]:
    data = df.copy()
    data["price_diff_1h"] = data["price"].diff()
    data["rolling_mean_24h"] = data["price"].rolling(24, min_periods=12).mean()
    data["rolling_std_24h"] = data["price"].rolling(24, min_periods=12).std()
    data["residual_24h"] = data["price"] - data["rolling_mean_24h"]
    data["pattern_median"] = data.groupby(["month", "hour"])["price"].transform("median")
    data["pattern_residual"] = data["price"] - data["pattern_median"]

    data["z_price"] = robust_zscore(data["price"]).abs()
    data["z_change"] = robust_zscore(data["price_diff_1h"].fillna(0)).abs()
    data["z_residual"] = robust_zscore(data["residual_24h"].fillna(0)).abs()
    data["z_pattern"] = robust_zscore(data["pattern_residual"].fillna(0)).abs()
    data["diagnostic_score"] = (
        0.35 * data["z_price"]
        + 0.25 * data["z_change"]
        + 0.25 * data["z_residual"]
        + 0.15 * data["z_pattern"]
    )
    threshold = float(np.nanpercentile(data["diagnostic_score"], anomaly_percentile))
    data["is_anomaly"] = (data["diagnostic_score"] >= threshold) | (data["price"] < 0)

    high_cut = float(np.nanpercentile(data["diagnostic_score"], 99.4))
    mid_cut = float(np.nanpercentile(data["diagnostic_score"], 98.2))
    data["severity"] = np.select(
        [data["diagnostic_score"] >= high_cut, data["diagnostic_score"] >= mid_cut, data["price"] < 0],
        ["高风险", "中风险", "中风险"],
        default="观察",
    )
    data["driver"] = np.select(
        [
            data["price"] < 0,
            data["z_change"] >= data[["z_price", "z_residual", "z_pattern"]].max(axis=1),
            data["z_residual"] >= data[["z_price", "z_change", "z_pattern"]].max(axis=1),
            data["z_pattern"] >= data[["z_price", "z_change", "z_residual"]].max(axis=1),
        ],
        ["负值/倒挂", "小时突变", "短窗残差", "季节模式偏离"],
        default="幅值越限",
    )
    return data, threshold


@st.cache_data(show_spinner=False)
def compute_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df.set_index("timestamp")["price"]
        .resample("h")
        .mean()
        .interpolate(method="time", limit_direction="both")
    )
    y = hourly.to_numpy(dtype=float)
    y = y - np.nanmean(y)
    if len(y) < 48:
        return pd.DataFrame(columns=["period_hours", "power", "power_db"])

    window = np.hanning(len(y))
    spectrum = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0)
    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = spectrum[mask]
    periods = 1 / freqs
    keep = (periods >= 2) & (periods <= 24 * 45)
    out = pd.DataFrame(
        {
            "period_hours": periods[keep],
            "power": spectrum[keep],
        }
    )
    out["power_db"] = 10 * np.log10(out["power"] + 1e-9)
    return out.sort_values("period_hours").reset_index(drop=True)


def multiscale_energy(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    price = df["price"].reset_index(drop=True)
    for window in [3, 6, 12, 24, 72, 168]:
        baseline = price.rolling(window, min_periods=max(2, window // 3)).mean()
        residual = price - baseline
        energy = float(np.sqrt(np.nanmean(np.square(residual))))
        rows.append(
            {
                "尺度": f"{window}小时",
                "残差能量": energy,
                "诊断含义": {
                    3: "瞬时扰动",
                    6: "班次波动",
                    12: "半日模式",
                    24: "日周期偏离",
                    72: "短期工况漂移",
                    168: "周周期偏离",
                }[window],
            }
        )
    return pd.DataFrame(rows)


def kmeans_numpy(features: pd.DataFrame, k: int, seed: int = 42) -> np.ndarray:
    x = features.to_numpy(dtype=float)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    x = np.nan_to_num((x - means) / stds)

    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    first = int(np.argmin(np.linalg.norm(x - np.nanmean(x, axis=0), axis=1)))
    centers = [x[first]]
    while len(centers) < k:
        dist = np.min(
            np.stack([np.linalg.norm(x - center, axis=1) ** 2 for center in centers]),
            axis=0,
        )
        probabilities = dist / dist.sum() if dist.sum() > 0 else np.ones(len(x)) / len(x)
        centers.append(x[rng.choice(len(x), p=probabilities)])
    centers = np.vstack(centers)

    labels = np.zeros(len(x), dtype=int)
    for _ in range(80):
        distances = np.stack([np.linalg.norm(x - center, axis=1) for center in centers], axis=1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            member = x[labels == idx]
            if len(member):
                centers[idx] = member.mean(axis=0)
    return labels


def daily_regimes(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        df.set_index("timestamp")
        .resample("D")
        .agg(
            avg_price=("price", "mean"),
            max_price=("price", "max"),
            min_price=("price", "min"),
            volatility=("price", "std"),
            anomaly_hours=("is_anomaly", "sum"),
            avg_score=("diagnostic_score", "mean"),
            negative_hours=("price", lambda s: int((s < 0).sum())),
        )
        .dropna(subset=["avg_price", "volatility"])
        .reset_index()
    )
    if daily.empty:
        return daily, pd.DataFrame()

    daily["peak_to_valley"] = daily["max_price"] - daily["min_price"]
    feature_cols = ["avg_price", "volatility", "peak_to_valley", "anomaly_hours", "avg_score", "negative_hours"]
    labels = kmeans_numpy(daily[feature_cols], k=k)
    daily["cluster"] = labels

    summary = (
        daily.groupby("cluster")
        .agg(
            days=("timestamp", "count"),
            avg_price=("avg_price", "mean"),
            volatility=("volatility", "mean"),
            anomaly_hours=("anomaly_hours", "mean"),
            avg_score=("avg_score", "mean"),
        )
        .reset_index()
        .sort_values(["avg_score", "volatility", "avg_price"], ascending=[True, True, True])
    )
    names = ["低波动基线", "常规运行", "高负荷波动", "异常冲击", "极端扰动"]
    name_map = {int(row.cluster): names[min(i, len(names) - 1)] for i, row in enumerate(summary.itertuples())}
    daily["regime"] = daily["cluster"].map(name_map)
    summary["regime"] = summary["cluster"].map(name_map)
    return daily, summary


def cyclical(values: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * values.to_numpy(dtype=float) / period
    return np.sin(angle), np.cos(angle)


def ridge_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = df.copy()
    model["lag_1h"] = model["price"].shift(1)
    model["lag_24h"] = model["price"].shift(24)
    model["lag_168h"] = model["price"].shift(168)
    model["rolling_mean_24h"] = model["price"].shift(1).rolling(24, min_periods=12).mean()
    model["rolling_std_24h"] = model["price"].shift(1).rolling(24, min_periods=12).std()
    model["target"] = model["price"]
    hsin, hcos = cyclical(model["hour"], 24)
    wsin, wcos = cyclical(model["weekday"], 7)
    msin, mcos = cyclical(model["month"], 12)
    model["hour_sin"] = hsin
    model["hour_cos"] = hcos
    model["week_sin"] = wsin
    model["week_cos"] = wcos
    model["month_sin"] = msin
    model["month_cos"] = mcos

    feature_cols = [
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "rolling_std_24h",
        "hour_sin",
        "hour_cos",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ]
    model = model.dropna(subset=feature_cols + ["target"]).copy()
    if len(model) < 500:
        return pd.DataFrame(), pd.DataFrame()

    split = int(len(model) * 0.78)
    train = model.iloc[:split]
    test = model.iloc[split:]
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    x_train_std = (x_train - mu) / sigma
    x_test_std = (x_test - mu) / sigma
    x_train_aug = np.c_[np.ones(len(x_train_std)), x_train_std]
    x_test_aug = np.c_[np.ones(len(x_test_std)), x_test_std]

    alpha = 8.0
    regularizer = alpha * np.eye(x_train_aug.shape[1])
    regularizer[0, 0] = 0
    beta = np.linalg.pinv(x_train_aug.T @ x_train_aug + regularizer) @ x_train_aug.T @ y_train
    prediction = x_test_aug @ beta

    residual = y_test - prediction
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1 - np.sum(residual**2) / np.sum((y_test - y_test.mean()) ** 2))
    metrics = pd.DataFrame(
        [
            {"模型": "Ridge 时序基线", "MAE": mae, "RMSE": rmse, "R2": r2, "训练样本": len(train), "测试样本": len(test)}
        ]
    )
    forecast = test[["timestamp", "target"]].copy()
    forecast["prediction"] = prediction
    forecast["absolute_error"] = np.abs(residual)
    return metrics, forecast


def model_availability() -> pd.DataFrame:
    rows = [
        ("SVM / SVR", "sklearn", "异常边界识别、非线性回归"),
        ("随机森林", "sklearn", "特征贡献评估、稳健分类/回归"),
        ("XGBoost", "xgboost", "高性能表格特征建模"),
        ("FFT / 多尺度残差", "numpy", "信号周期、冲击、漂移诊断"),
    ]
    return pd.DataFrame(
        [
            {
                "技术模块": name,
                "依赖": dep,
                "当前环境": "可运行" if importlib.util.find_spec(dep) else "待安装",
                "岗位映射": desc,
            }
            for name, dep, desc in rows
        ]
    )


def format_period(hours: float) -> str:
    if hours >= 48:
        return f"{hours / 24:.1f} 天"
    return f"{hours:.1f} 小时"


def severity_html(severity: str) -> str:
    if severity == "高风险":
        return '<span class="risk-high">高风险</span>'
    if severity == "中风险":
        return '<span class="risk-mid">中风险</span>'
    return "观察"


def metric_card(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else '<div class="metric-delta">&nbsp;</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


if not DATA_PATH.exists():
    st.error(f"未找到数据文件：{DATA_PATH.resolve()}")
    st.stop()


df_raw = load_price_data(str(DATA_PATH))

with st.sidebar:
    st.subheader("分析控制")
    years = sorted(df_raw["year"].dropna().unique().tolist())
    selected_years = st.multiselect("年份", years, default=years)
    anomaly_percentile = st.slider("异常评分分位阈值", min_value=95, max_value=99, value=98, step=1)
    regime_count = st.slider("工况簇数量", min_value=3, max_value=5, value=4, step=1)
    st.divider()
    st.subheader("项目链接")
    st.link_button("GitHub 仓库", GITHUB_URL, width="stretch")
    if STREAMLIT_URL:
        st.link_button("Streamlit 展示", STREAMLIT_URL, width="stretch")
    else:
        st.button("Streamlit 展示待部署", disabled=True, width="stretch")
    st.caption("收到 Streamlit Cloud 链接后，可替换顶部常量或环境变量。")


df_diag, anomaly_threshold = enrich_diagnostics(df_raw, anomaly_percentile)
if selected_years:
    df_view = df_diag[df_diag["year"].isin(selected_years)].copy()
else:
    df_view = df_diag.copy()

if df_view.empty:
    st.warning("当前筛选条件下没有可分析数据。")
    st.stop()

daily_view, regime_summary = daily_regimes(df_view, regime_count)
spectrum = compute_spectrum(df_view)
energy = multiscale_energy(df_view)
model_metrics, forecast = ridge_forecast(df_view)

date_min = df_view["timestamp"].min()
date_max = df_view["timestamp"].max()
anomaly_count = int(df_view["is_anomaly"].sum())
negative_count = int((df_view["price"] < 0).sum())
peak_price = float(df_view["price"].max())
valley_price = float(df_view["price"].min())
mean_price = float(df_view["price"].mean())


st.title(APP_TITLE)
st.markdown(
    """
    <span class="status-pill">工业故障诊断</span>
    <span class="status-pill">信号分析</span>
    <span class="status-pill">传统机器学习</span>
    <span class="status-pill">Python / Streamlit</span>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-note">
    当前数据为国外逐小时电价时序，用作工业信号诊断样例：将价格波动视作可替换的设备状态信号，
    展示清洗、周期识别、异常评分、工况聚类与预测基线。后续可直接替换为振动、电流、温度、牵引系统或轨旁监测数据。
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metric-grid">
    """
    + metric_card("有效样本", f"{len(df_view):,}", f"{date_min:%Y-%m-%d} 至 {date_max:%Y-%m-%d}")
    + metric_card("均值", f"{mean_price:,.2f}")
    + metric_card("峰值", f"{peak_price:,.2f}")
    + metric_card("谷值", f"{valley_price:,.2f}")
    + metric_card("异常小时", f"{anomaly_count:,}", f"阈值 {anomaly_threshold:.2f}")
    + metric_card("负值小时", f"{negative_count:,}")
    + """
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_signal, tab_model, tab_project = st.tabs(["诊断总览", "信号分析", "模型评估", "项目展示"])

with tab_overview:
    st.subheader("时序诊断")
    plot_df = df_view.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=plot_df["timestamp"],
            y=plot_df["price"],
            mode="lines",
            name="信号值",
            line=dict(color="#2563eb", width=1.3),
        )
    )
    anomaly_df = plot_df[plot_df["is_anomaly"]]
    fig.add_trace(
        go.Scattergl(
            x=anomaly_df["timestamp"],
            y=anomaly_df["price"],
            mode="markers",
            name="异常点",
            marker=dict(color="#be123c", size=6, symbol="circle"),
            hovertemplate="%{x}<br>值=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        xaxis_title=None,
        yaxis_title="电价 / 信号幅值",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("小时-月份工况热力")
        heatmap = (
            df_view.pivot_table(index="hour", columns="month", values="price", aggfunc="mean")
            .sort_index()
            .reindex(columns=range(1, 13))
        )
        heat_fig = go.Figure(
            data=go.Heatmap(
                z=heatmap.to_numpy(),
                x=[f"{m}月" for m in heatmap.columns],
                y=[f"{h:02d}:00" for h in heatmap.index],
                colorscale=[[0, "#f8fafc"], [0.5, "#14b8a6"], [1, "#be123c"]],
                colorbar=dict(title="均值"),
            )
        )
        heat_fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=None)
        st.plotly_chart(heat_fig, width="stretch")
    with right:
        st.subheader("高风险样本")
        table = (
            anomaly_df.sort_values("diagnostic_score", ascending=False)
            .head(18)
            .assign(
                时间=lambda x: x["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
                诊断等级=lambda x: x["severity"],
                主因=lambda x: x["driver"],
                信号值=lambda x: x["price"].round(2),
                评分=lambda x: x["diagnostic_score"].round(2),
            )[["时间", "诊断等级", "主因", "信号值", "评分"]]
        )
        st.dataframe(table, width="stretch", hide_index=True)

with tab_signal:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("傅里叶频谱")
        if spectrum.empty:
            st.info("数据长度不足，暂不能计算频谱。")
        else:
            spec_plot = spectrum[(spectrum["period_hours"] <= 24 * 30)].copy()
            spec_fig = px.line(
                spec_plot,
                x="period_hours",
                y="power_db",
                labels={"period_hours": "周期（小时）", "power_db": "功率（dB）"},
            )
            spec_fig.update_traces(line=dict(color="#0f766e", width=1.8))
            spec_fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
            st.plotly_chart(spec_fig, width="stretch")
    with right:
        st.subheader("主周期识别")
        if not spectrum.empty:
            top_periods = (
                spectrum.sort_values("power", ascending=False)
                .head(8)
                .assign(
                    主周期=lambda x: x["period_hours"].map(format_period),
                    周期小时=lambda x: x["period_hours"].round(2),
                    功率dB=lambda x: x["power_db"].round(2),
                )[["主周期", "周期小时", "功率dB"]]
            )
            st.dataframe(top_periods, width="stretch", hide_index=True)
        st.subheader("多尺度残差能量")
        st.dataframe(energy.round({"残差能量": 2}), width="stretch", hide_index=True)

    st.subheader("日级工况聚类")
    if daily_view.empty:
        st.info("当前筛选数据不足，无法形成日级聚类。")
    else:
        scatter = px.scatter(
            daily_view,
            x="avg_price",
            y="volatility",
            color="regime",
            size="anomaly_hours",
            hover_data={
                "timestamp": "|%Y-%m-%d",
                "avg_price": ":.2f",
                "volatility": ":.2f",
                "peak_to_valley": ":.2f",
                "anomaly_hours": True,
                "regime": True,
            },
            labels={"avg_price": "日均值", "volatility": "日内波动", "regime": "工况"},
            color_discrete_sequence=["#0f766e", "#2563eb", "#b45309", "#be123c", "#7c3aed"],
        )
        scatter.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(scatter, width="stretch")
        if not regime_summary.empty:
            st.dataframe(
                regime_summary.assign(
                    avg_price=lambda x: x["avg_price"].round(2),
                    volatility=lambda x: x["volatility"].round(2),
                    anomaly_hours=lambda x: x["anomaly_hours"].round(2),
                    avg_score=lambda x: x["avg_score"].round(2),
                )[["regime", "days", "avg_price", "volatility", "anomaly_hours", "avg_score"]],
                width="stretch",
                hide_index=True,
            )

with tab_model:
    st.subheader("预测基线与异常残差")
    if model_metrics.empty or forecast.empty:
        st.info("当前筛选数据不足，无法训练预测基线。")
    else:
        mcols = st.columns(3)
        row = model_metrics.iloc[0]
        mcols[0].metric("MAE", f"{row['MAE']:.2f}")
        mcols[1].metric("RMSE", f"{row['RMSE']:.2f}")
        mcols[2].metric("R2", f"{row['R2']:.3f}")

        forecast_plot = forecast.tail(min(1200, len(forecast)))
        ffig = go.Figure()
        ffig.add_trace(
            go.Scattergl(
                x=forecast_plot["timestamp"],
                y=forecast_plot["target"],
                name="实际",
                mode="lines",
                line=dict(color="#2563eb", width=1.4),
            )
        )
        ffig.add_trace(
            go.Scattergl(
                x=forecast_plot["timestamp"],
                y=forecast_plot["prediction"],
                name="预测",
                mode="lines",
                line=dict(color="#0f766e", width=1.4),
            )
        )
        ffig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.08),
            yaxis_title="信号值",
            xaxis_title=None,
            hovermode="x unified",
        )
        st.plotly_chart(ffig, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("传统机器学习模块状态")
        st.dataframe(model_availability(), width="stretch", hide_index=True)
    with right:
        st.subheader("特征工程口径")
        feature_table = pd.DataFrame(
            [
                ["时域", "1小时差分、24小时滚动均值/标准差、短窗残差", "冲击、漂移、突变"],
                ["频域", "FFT 主周期、功率谱、多尺度残差能量", "周期异常、运行节律改变"],
                ["模式基线", "月份-小时中位数、季节模式残差", "季节工况偏离"],
                ["工况识别", "日均值、波动、峰谷差、异常小时数聚类", "稳定/波动/冲击工况分层"],
                ["预测残差", "滞后项、周期编码、滚动统计的 Ridge 基线", "可扩展为 SVM/RF/XGBoost"],
            ],
            columns=["类别", "当前实现", "诊断价值"],
        )
        st.dataframe(feature_table, width="stretch", hide_index=True)

with tab_project:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("岗位匹配说明")
        mapping = pd.DataFrame(
            [
                [
                    "核心技术栈",
                    "信号处理（傅里叶变换、多尺度残差，可扩展小波/包络解调）+ 传统机器学习（SVM、随机森林、XGBoost）+ Python/Matlab。",
                ],
                [
                    "技术特点",
                    "深度绑定工业故障诊断与信号分析，以传统算法和机理结合为主，深度学习作为后续增强项。",
                ],
                [
                    "当前页面",
                    "基于国外逐小时时序数据完成数据清洗、异常诊断、工况聚类、频域分析和预测基线。",
                ],
                [
                    "后续接入",
                    "可替换为设备振动、电流、温度、轨旁监测、牵引系统运行数据，保留同一套诊断流程。",
                ],
            ],
            columns=["维度", "唐智科技 数据挖掘岗"],
        )
        st.dataframe(mapping, width="stretch", hide_index=True)
    with right:
        st.subheader("展示链接")
        st.link_button("打开 GitHub 仓库", GITHUB_URL, width="stretch")
        if STREAMLIT_URL:
            st.link_button("打开 Streamlit 在线页面", STREAMLIT_URL, width="stretch")
        else:
            st.button("Streamlit 在线页面待部署", disabled=True, width="stretch")
        st.markdown(
            """
            <div class="section-note">
            部署时建议在 Streamlit Cloud 设置环境变量 PROJECT_GITHUB_URL 和 PROJECT_STREAMLIT_URL；
            若使用真实设备数据，可保留字段映射层并替换特征工程输入。
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("数据质量审计")
    quality = pd.DataFrame(
        [
            ["有效记录", f"{len(df_raw):,}"],
            ["时间范围", f"{df_raw['timestamp'].min():%Y-%m-%d %H:%M} 至 {df_raw['timestamp'].max():%Y-%m-%d %H:%M}"],
            ["重复时间戳", f"{int(df_raw['timestamp'].duplicated().sum()):,}"],
            ["缺失清洗", "已剔除空模板行和缺失电价行"],
            ["日期格式", "按日/月/年解析，避免 01/02/2025 的月日歧义"],
        ],
        columns=["检查项", "结果"],
    )
    st.dataframe(quality, width="stretch", hide_index=True)
