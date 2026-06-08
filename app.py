import random, os, warnings, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e1;
}
.stApp { background-color: #0d0f14; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

.hero {
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid #2a2d36;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0c040 0%, #f07840 60%, #d040b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero p {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.04em;
}

.metric-card {
    background: #161820;
    border: 1px solid #2a2d36;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #f0c040;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 0.2rem;
}

.section-header {
    font-size: 1rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f07840;
    font-family: 'DM Mono', monospace;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2d36;
}

.leaderboard-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: #161820;
    border: 1px solid #2a2d36;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}
.leaderboard-row.best {
    border-color: #f0c040;
    background: #1a1c10;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
}
.badge-gold  { background: #2a2400; color: #f0c040; border: 1px solid #f0c040; }
.badge-green { background: #0a2010; color: #40d080; border: 1px solid #40d080; }
.badge-gray  { background: #1e2030; color: #888;    border: 1px solid #444; }

.upload-zone {
    border: 2px dashed #2a2d36;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    background: #161820;
    transition: border-color 0.2s;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101218 !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stMarkdown { color: #888; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f0c040, #f07840);
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #f0c040, #f07840); }

/* Plots */
section[data-testid="stPlotlyChart"], .stImage { border-radius: 12px; overflow: hidden; }

div[data-testid="stFileUploader"] label { color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 42
TARGET_COL = "PowerConsumption_Zone1"
LOOKBACK   = 48
TEST_RATIO = 0.20

FEATURES = [
    "PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3",
    "Temperature", "Humidity", "WindSpeed",
    "GeneralDiffuseFlows", "DiffuseFlows",
    "Hour", "DayOfWeek", "Month", "IsWeekend",
]
TARGET_IDX = FEATURES.index(TARGET_COL)

PLOT_STYLE = {
    "figure.facecolor": "#0d0f14",  # Changed from "facecolor"
    "text.color": "#e8e6e1",
    "axes.facecolor": "#161820",
    "axes.edgecolor": "#2a2d36",
    "axes.labelcolor": "#e8e6e1",
    "xtick.color": "#666",
    "ytick.color": "#666",
    "grid.color": "#2a2d36",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
}

ACCENT_COLORS = ["#f0c040", "#f07840", "#d040b0", "#40a0f0", "#40d080", "#a040f0"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

def styled_fig(figsize=(14, 5)):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize,
                               facecolor=PLOT_STYLE["figure.facecolor"])
        ax.set_facecolor(PLOT_STYLE["axes.facecolor"])
        return fig, ax

def styled_subplots(rows, cols, figsize=(16, 6)):
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 facecolor=PLOT_STYLE["figure.facecolor"])
        for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
            ax.set_facecolor(PLOT_STYLE["axes.facecolor"])
        return fig, axes

def calc_metrics(y_true, y_pred):
    return {
        "R²":   round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
    }

def create_windows(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback, TARGET_IDX])
    return np.array(X), np.array(y)

def inverse_target(arr, scaler):
    dummy = np.zeros((len(arr), len(FEATURES)))
    dummy[:, TARGET_IDX] = np.array(arr).ravel()
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ Energy Consumption<br>Predictor</h1>
  <p>ML pipeline · Linear Regression · Random Forest · XGBoost · ANN · CNN · CNN+LSTM</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Config")
    lookback   = st.slider("Lookback window (steps)", 12, 96, LOOKBACK, 12,
                           help="Each step = 10 min. Default 48 = 8 hours.")
    test_ratio = st.slider("Test split %", 10, 40, int(TEST_RATIO * 100), 5) / 100
    run_dl     = st.checkbox("Include CNN & CNN+LSTM", value=True,
                             help="Deep learning models take significantly longer.")
    st.divider()
    st.markdown("""
    **Dataset requirements**

    Upload the [Tetuan City Power Consumption](https://archive.ics.uci.edu/dataset/849) CSV  
    from UCI ML Repository.

    Required columns:
    - `Datetime`
    - `PowerConsumption_Zone1/2/3`
    - `Temperature`, `Humidity`, `WindSpeed`
    - `GeneralDiffuseFlows`, `DiffuseFlows`
    """)

# ── File upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">01 · Upload Dataset</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload powerconsumption.csv", type=["csv"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem; margin-bottom:0.8rem">📂</div>
        <div style="font-size:1rem; color:#888; font-family:'DM Mono',monospace;">
            Drop your <strong style="color:#f0c040">powerconsumption.csv</strong> above<br>
            <span style="font-size:0.8rem; color:#555;">Tetuan City Power Consumption · UCI ML Repository</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & clean data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().dropna()
    df["Hour"]      = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month
    df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)
    return df

with st.spinner("Loading & cleaning data…"):
    df_raw = load_data(uploaded.read())

# ── Data overview ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">02 · Dataset Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, "TOTAL ROWS",    f"{len(df_raw):,}",
         f"{df_raw.index.min().date()}"),
    (c2, "FEATURES",      str(len(FEATURES)),
         "incl. temporal"),
    (c3, "DATE RANGE",    str((df_raw.index.max() - df_raw.index.min()).days),
         "days of data"),
    (c4, "INTERVAL",      "10 min",
         "sampling freq"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── EDA ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">03 · Exploratory Data Analysis</div>', unsafe_allow_html=True)

with plt.rc_context(PLOT_STYLE):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             facecolor=PLOT_STYLE["facecolor"])
    for ax in axes:
        ax.set_facecolor(PLOT_STYLE["axes.facecolor"])

    # Time series
    df_raw[TARGET_COL].plot(ax=axes[0], linewidth=0.5, color="#f0c040", alpha=0.9)
    axes[0].set_title(f"{TARGET_COL} — Full Time Series",
                      fontsize=13, color="#e8e6e1", pad=10)
    axes[0].set_ylabel("Power (kW)", color="#888")
    axes[0].set_xlabel("")

    # Daily mean/std
    daily_mean = df_raw[TARGET_COL].resample("1D").mean()
    daily_std  = df_raw[TARGET_COL].resample("1D").std()
    daily_mean.plot(ax=axes[1], label="Daily Mean", color="#f0c040")
    daily_std.plot(ax=axes[1], label="Daily Std",   color="#f07840", alpha=0.7)
    axes[1].set_title("Daily Mean & Std Dev Over Time",
                      fontsize=13, color="#e8e6e1", pad=10)
    axes[1].legend(facecolor="#1e2030", edgecolor="#2a2d36", labelcolor="#e8e6e1")

    # Correlation
    corr = df_raw[FEATURES].diff().dropna().corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdYlGn", annot=True, fmt=".2f",
                linewidths=0.4, ax=axes[2], cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7})
    axes[2].set_title("Feature Correlation Matrix",
                      fontsize=13, color="#e8e6e1", pad=10)
    axes[2].tick_params(colors="#888", labelsize=8)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    plt.close(fig)

# ── Feature engineering & windowing ───────────────────────────────────────────
st.markdown('<div class="section-header">04 · Feature Engineering & Windowing</div>',
            unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def prepare_data(file_bytes, _lookback, _test_ratio):
    df = load_data(file_bytes)
    df_model = df[FEATURES].copy()
    n_total   = len(df_model)
    split_raw = int(n_total * (1 - _test_ratio))

    train_raw = df_model.iloc[: split_raw + _lookback]
    test_raw  = df_model.iloc[split_raw:]

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    X_train_3d, y_train = create_windows(train_scaled, _lookback)
    X_test_3d,  y_test  = create_windows(test_scaled,  _lookback)

    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test_flat  = X_test_3d.reshape(X_test_3d.shape[0],  -1)

    return X_train_3d, X_train_flat, y_train, X_test_3d, X_test_flat, y_test, scaler

uploaded.seek(0)
with st.spinner("Preparing windows…"):
    (X_train_3d, X_train_flat, y_train,
     X_test_3d,  X_test_flat,  y_test, scaler) = prepare_data(
         uploaded.read(), lookback, test_ratio)

inv = lambda arr: inverse_target(arr, scaler)
y_test_real = inv(y_test)

c1, c2, c3 = st.columns(3)
for col, label, val in [
    (c1, "TRAIN SAMPLES", f"{X_train_3d.shape[0]:,}"),
    (c2, "TEST SAMPLES",  f"{X_test_3d.shape[0]:,}"),
    (c3, "LOOKBACK",      f"{lookback} steps ({lookback*10//60}h)"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="font-size:1.5rem">{val}</div>
    </div>""", unsafe_allow_html=True)

# ── Train models ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">05 · Model Training</div>', unsafe_allow_html=True)

if st.button("🚀 Run Full ML Pipeline"):
    set_seeds()
    all_preds    = {}
    all_history  = {}
    progress_bar = st.progress(0)
    status       = st.empty()

    models_to_run = ["Linear Regression", "Random Forest", "XGBoost", "ANN"]
    if run_dl:
        models_to_run += ["CNN", "CNN + LSTM"]
    total = len(models_to_run)

    # 1. Linear Regression
    status.markdown("⚙️ **Training Linear Regression…**")
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    all_preds["Linear Regression"] = inv(lr.predict(X_test_flat))
    progress_bar.progress(1 / total)

    # 2. Random Forest
    status.markdown("🌲 **Training Random Forest (100 trees)…**")
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    all_preds["Random Forest"] = inv(rf.predict(X_test_flat))
    progress_bar.progress(2 / total)

    # 3. XGBoost
    status.markdown("⚡ **Training XGBoost with early stopping…**")
    val_split = int(len(X_train_flat) * 0.9)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(
        X_train_flat[:val_split], y_train[:val_split],
        eval_set=[(X_train_flat[val_split:], y_train[val_split:])],
        verbose=False,
    )
    all_preds["XGBoost"] = inv(xgb_model.predict(X_test_flat))
    progress_bar.progress(3 / total)

    # 4. ANN / MLP
    status.markdown("🧠 **Training ANN/MLP…**")
    ann = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        max_iter=500, random_state=SEED,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, tol=1e-4,
    )
    ann.fit(X_train_flat, y_train)
    all_preds["ANN (MLP)"] = inv(ann.predict(X_test_flat))
    progress_bar.progress(4 / total)

    if run_dl:
        n_features = X_train_3d.shape[2]
        es = lambda: EarlyStopping(monitor="val_loss", patience=5,
                                   restore_best_weights=True, verbose=0)

        # 5. CNN
        status.markdown("🔬 **Building & training CNN…**")
        cnn = Sequential([
            Conv1D(64, 3, activation="relu", input_shape=(lookback, n_features)),
            Dropout(0.2), MaxPooling1D(2),
            Conv1D(32, 3, activation="relu"), Dropout(0.2),
            Flatten(), Dense(64, activation="relu"), Dense(1),
        ], name="CNN")
        cnn.compile(optimizer=Adam(1e-3), loss="mse")
        h_cnn = cnn.fit(X_train_3d, y_train, epochs=100, batch_size=64,
                        validation_split=0.1, callbacks=[es()], verbose=0)
        all_preds["CNN"] = inv(cnn.predict(X_test_3d, verbose=0))
        all_history["CNN"] = h_cnn.history
        progress_bar.progress(5 / total)

        # 6. CNN + LSTM
        status.markdown("🔁 **Building & training CNN + LSTM…**")
        cnn_lstm = Sequential([
            Conv1D(64, 3, activation="relu", input_shape=(lookback, n_features)),
            Dropout(0.2), MaxPooling1D(2),
            Conv1D(32, 3, activation="relu"), Dropout(0.2),
            LSTM(64), Dense(32, activation="relu"), Dense(1),
        ], name="CNN_LSTM")
        cnn_lstm.compile(optimizer=Adam(1e-3), loss="mse")
        h_lstm = cnn_lstm.fit(X_train_3d, y_train, epochs=100, batch_size=64,
                              validation_split=0.1, callbacks=[es()], verbose=0)
        all_preds["CNN + LSTM"] = inv(cnn_lstm.predict(X_test_3d, verbose=0))
        all_history["CNN + LSTM"] = h_lstm.history
        progress_bar.progress(6 / total)

    status.empty()
    progress_bar.empty()

    # ── Leaderboard ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">06 · Results Leaderboard</div>',
                unsafe_allow_html=True)

    results = []
    for name, pred in all_preds.items():
        row = calc_metrics(y_test_real, pred)
        row["Model"] = name
        results.append(row)

    leaderboard = (pd.DataFrame(results)
                   .set_index("Model")[["R²", "RMSE", "MAE"]]
                   .sort_values("R²", ascending=False))

    for i, (model, row) in enumerate(leaderboard.iterrows()):
        if i == 0:
            badge = '<span class="badge badge-gold">🥇 BEST</span>'
            cls   = "leaderboard-row best"
        elif row["R²"] >= 0.9:
            badge = '<span class="badge badge-green">HIGH</span>'
            cls   = "leaderboard-row"
        else:
            badge = '<span class="badge badge-gray">—</span>'
            cls   = "leaderboard-row"

        st.markdown(f"""
        <div class="{cls}">
            <span style="color:#e8e6e1;width:200px;display:inline-block">{model} {badge}</span>
            <span style="color:#f0c040">R² {row['R²']:.4f}</span>
            <span style="color:#888">RMSE {row['RMSE']:.1f}</span>
            <span style="color:#888">MAE {row['MAE']:.1f}</span>
        </div>""", unsafe_allow_html=True)

    # ── Bar charts ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">07 · Model Comparison</div>',
                unsafe_allow_html=True)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 facecolor=PLOT_STYLE["facecolor"])
        for ax in axes:
            ax.set_facecolor(PLOT_STYLE["axes.facecolor"])

        palette = ACCENT_COLORS[:len(leaderboard)]

        axes[0].bar(leaderboard.index, leaderboard["R²"], color=palette)
        axes[0].set_title("R² Score — Higher is Better",
                          color="#e8e6e1", fontsize=12)
        axes[0].set_ylabel("R²", color="#888")
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis="x", rotation=35, colors="#888", labelsize=8)
        for bar, val in zip(axes[0].patches, leaderboard["R²"]):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         min(bar.get_height() + 0.01, 1.0),
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=8, color="#e8e6e1")

        axes[1].bar(leaderboard.index, leaderboard["RMSE"], color=palette)
        axes[1].set_title("RMSE — Lower is Better",
                          color="#e8e6e1", fontsize=12)
        axes[1].set_ylabel("RMSE (kW)", color="#888")
        axes[1].tick_params(axis="x", rotation=35, colors="#888", labelsize=8)
        for bar, val in zip(axes[1].patches, leaderboard["RMSE"]):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{val:.1f}", ha="center", va="bottom",
                         fontsize=8, color="#e8e6e1")

        plt.suptitle("Power Consumption Zone 1 — All Models",
                     color="#e8e6e1", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">08 · Actual vs Predicted</div>',
                unsafe_allow_html=True)

    best_name = leaderboard.index[0]
    best_pred = all_preds[best_name]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(14, 4),
                               facecolor=PLOT_STYLE["facecolor"])
        ax.set_facecolor(PLOT_STYLE["axes.facecolor"])
        ax.plot(y_test_real[:500], label="Actual",
                linewidth=1.0, color="#40a0f0")
        ax.plot(best_pred[:500],   label=best_name,
                linewidth=1.0, color="#f0c040", alpha=0.9)
        ax.set_title(f"Actual vs Predicted — {best_name} (first 500 test samples)",
                     color="#e8e6e1", fontsize=12)
        ax.set_xlabel("Sample index", color="#888")
        ax.set_ylabel("Power (kW)", color="#888")
        ax.legend(facecolor="#1e2030", edgecolor="#2a2d36", labelcolor="#e8e6e1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── DL training curves ─────────────────────────────────────────────────────
    if all_history:
        st.markdown('<div class="section-header">09 · Deep Learning Training Curves</div>',
                    unsafe_allow_html=True)

        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(1, len(all_history), figsize=(14, 4),
                                     facecolor=PLOT_STYLE["facecolor"])
            if len(all_history) == 1:
                axes = [axes]
            for ax in axes:
                ax.set_facecolor(PLOT_STYLE["axes.facecolor"])

            for ax, (name, hist) in zip(axes, all_history.items()):
                epochs = range(len(hist["loss"]))
                ax.plot(epochs, hist["loss"],     color="#f0c040", label="Train")
                ax.plot(epochs, hist["val_loss"], color="#f07840",
                        linestyle="--", label="Validation")
                ax.axvline(len(epochs) - 1, color="#d040b0",
                           linestyle=":", linewidth=0.8, label="Early stop")
                ax.set_title(f"{name}: Training History",
                             color="#e8e6e1", fontsize=11)
                ax.set_xlabel("Epoch", color="#888")
                ax.set_ylabel("MSE Loss", color="#888")
                ax.legend(facecolor="#1e2030", edgecolor="#2a2d36",
                          labelcolor="#e8e6e1", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Download leaderboard ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">10 · Export</div>',
                unsafe_allow_html=True)
    csv_bytes = leaderboard.to_csv().encode()
    st.download_button("⬇️ Download Leaderboard CSV", csv_bytes,
                       file_name="model_leaderboard.csv", mime="text/csv")

    st.success("✅ Pipeline complete!")

else:
    st.info("Configure settings in the sidebar, then click **Run Full ML Pipeline** above.")
