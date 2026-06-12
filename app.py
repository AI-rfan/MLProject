import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import date

# Page config 
st.set_page_config(
    page_title="Zone 1 Power Forecast",
    page_icon="⚡",
    layout="centered",
)

# Helpers and constants
MODELS_DIR = "trained_models"

MODEL_FILES = {
    "XGBoost":           "xgboost_model.joblib",
    "ANN (MLP)":         "ann_model.joblib",
    "CNN + LSTM":        "hybrid_model.joblib",
    "1D CNN":            "cnn_model.joblib",
    "Linear Regression": "linear_regression_model.joblib",
}

# R² from your leaderboard (for display only)
MODEL_R2 = {
    "XGBoost":           0.8958,
    "ANN (MLP)":         0.8845,
    "CNN + LSTM":        0.8202,
    "1D CNN":            0.6723,
    "Linear Regression": 0.6741,
}

@st.cache_resource(show_spinner="Loading models…")
def load_assets():
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        return None, {}

    scaler = joblib.load(scaler_path)
    models = {}
    for name, fname in MODEL_FILES.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            models[name] = joblib.load(fpath)
    return scaler, models


def build_features(input_date: date, hour: int,
                   temp, humidity, wind, gen_diffuse, diffuse):
    """Return a (1, 9) numpy array matching training feature order."""
    dow        = input_date.weekday()          # 0 = Monday
    month      = input_date.month
    is_weekend = int(dow >= 5)
    return np.array([[temp, humidity, wind,
                      gen_diffuse, diffuse,
                      hour, dow, month, is_weekend]], dtype=np.float32)


def predict(scaler, models, features, selected_models):
    """Run inference and return {model_name: prediction_kW}."""
    scaled    = scaler.transform(features)          # (1, 9)
    scaled_dl = scaled.reshape(1, 1, scaled.shape[1])  # (1, 1, 9) for DL

    results = {}
    for name in selected_models:
        model = models[name]
        try:
            if name in ("1D CNN", "CNN + LSTM"):
                pred = model.predict(scaled_dl, verbose=0).flatten()[0]
            else:
                pred = model.predict(scaled)[0]
            results[name] = float(pred)
        except Exception as e:
            results[name] = f"Error: {e}"
    return results


# UI 
st.title("Zone 1 Power Consumption Forecast")
st.caption("Tétouan City dataset · predicts PowerConsumption_Zone1 (kW)")
st.divider()

scaler, models = load_assets()

if scaler is None or not models:
    st.error(
        "**Models or scaler not found.**\n\n"
        "Make sure the `trained_models/` folder contains:\n"
        "- `scaler.joblib`\n"
        "- `xgboost_model.joblib`, `ann_model.joblib`, etc.\n\n"
        "Add `joblib.dump(scaler, 'trained_models/scaler.joblib')` to your "
        "Colab notebook and re-run the save cell."
    )
    st.stop()

available = list(models.keys())

# Sidebar: model selector
with st.sidebar:
    st.header("Model Selection")
    selected = st.multiselect(
        "Choose models to run",
        options=available,
        default=available[:2],
        help="Select one or more trained models",
    )
    st.divider()
    st.subheader("Leaderboard (test R²)")
    lb = pd.DataFrame(
        [{"Model": m, "R²": MODEL_R2.get(m, "—")} for m in available]
    ).set_index("Model")
    st.dataframe(lb, use_container_width=True)
# Main: inputs
st.subheader("Date & Time")
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input(
        "Date", value=date(2017, 1, 15),
    )
with col2:
    hour = st.slider("Hour (0–23)", 0, 23, 12)

st.subheader("Weather Conditions")
c1, c2, c3 = st.columns(3)
with c1:
    temp     = st.number_input("Temperature (°C)", -5.0, 45.0, 15.0, step=0.5)
    humidity = st.number_input("Humidity (%)",      0.0, 100.0, 65.0, step=0.5)
with c2:
    wind       = st.number_input("Wind Speed",         0.0,    5.0,  0.08, step=0.01, format="%.3f")
    gen_diffuse= st.number_input("Gen. Diffuse Flows", 0.0, 1000.0,  50.0, step=0.1)
with c3:
    diffuse    = st.number_input("Diffuse Flows",      0.0, 1000.0,  20.0, step=0.1)

# Prediction
st.divider()
if st.button("Run Forecast", type="primary", use_container_width=True):
    if not selected:
        st.warning("Please select at least one model from the sidebar.")
    else:
        features = build_features(input_date, hour, temp, humidity,
                                  wind, gen_diffuse, diffuse)
        results  = predict(scaler, models, features, selected)

        st.subheader("Forecast Results — Zone 1")

        cols = st.columns(len(results))
        for col, (name, val) in zip(cols, results.items()):
            with col:
                if isinstance(val, float):
                    st.metric(label=name, value=f"{val:,.0f} kW")
                else:
                    st.error(f"{name}: {val}")

        # Bar chart
        valid = {k: v for k, v in results.items() if isinstance(v, float)}
        if len(valid) > 1:
            chart_df = pd.DataFrame(
                {"Model": list(valid.keys()), "Predicted kW": list(valid.values())}
            ).set_index("Model")
            st.bar_chart(chart_df)