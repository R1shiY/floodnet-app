import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model

# Load model and preprocessing
model = load_model("floodnet_model.h5")
scaler = joblib.load("floodnet_scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

def generate_sequences(X_base, timesteps=5, noise=0.05):
    sequences = []
    for row in X_base:
        sequence = [row + np.random.normal(0, noise, row.shape) for _ in range(timesteps)]
        sequences.append(sequence)
    return np.array(sequences)

st.set_page_config(layout="wide")
st.title("🌐 FloodNet AI Dashboard")

# Sidebar sliders
st.sidebar.header("Flood Risk Simulator")
user_input = {}
for feat in ["Urbanization", "Deforestation", "ClimateChange", "DrainageSystems", "DamsQuality"]:
    user_input[feat] = st.sidebar.slider(feat, 0.0, 10.0, 5.0, 0.1)

# Base input
base = pd.DataFrame([np.mean([5.0] * len(feature_cols))], columns=feature_cols)
for k, v in user_input.items():
    base[k] = v

x_scaled = scaler.transform(base)
x_cnn = x_scaled.reshape(1, 5, 4, 1)
x_lstm = generate_sequences(x_scaled, timesteps=5)

# Tabs
tab1, tab2, tab3 = st.tabs(["Flood Risk Simulator", "FloodVision Viewer", "NeuroFlood Forecast"])

# Tab 1: Simulator
with tab1:
    pred = float(model.predict([x_cnn, x_lstm]).flatten()[0])
    flood_score = round(pred * 10, 2)
    st.subheader("Predicted Flood Risk Score")
    st.markdown(f"<h2 style='color: crimson;'>{flood_score} / 10</h2>", unsafe_allow_html=True)

# Tab 2: CNN Saliency Map
with tab2:
    with tf.GradientTape() as tape:
        input_tensor = tf.convert_to_tensor(x_cnn)
        tape.watch(input_tensor)
        prediction = model([input_tensor, tf.convert_to_tensor(x_lstm)])
    grads = tape.gradient(prediction, input_tensor).numpy().squeeze()
    saliency = np.abs(grads)
    feature_names = np.array(feature_cols).reshape(5, 4)
    values = scaler.inverse_transform(x_scaled).reshape(5, 4)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(values, annot=np.round(values, 1), cmap="Blues", fmt=".1f", ax=ax[0], cbar=False)
    ax[0].set_title("Input Features")

    sns.heatmap(saliency, annot=feature_names, cmap="YlOrRd", fmt="", ax=ax[1])
    ax[1].set_title("CNN Saliency Map")
    st.pyplot(fig)

# Tab 3: Forecast
with tab3:
    st.subheader("Forecast Years Ahead")
    years = st.slider("Years", 3, 30, 10)
    drift_features = ["ClimateChange", "Urbanization", "Deforestation"]
    feature_indices = [feature_cols.index(f) for f in drift_features]
    drift_rates = np.array([0.02, 0.03, 0.015])
    predictions = []

    for t in range(years):
        new_input = x_scaled[0].copy()
        new_input[feature_indices] = np.clip(new_input[feature_indices] + (t + 1) * drift_rates, 0, 1)
        lstm_seq = generate_sequences([new_input], timesteps=5)
        cnn_input = new_input.reshape(1, 5, 4, 1)
        score = float(model.predict([cnn_input, lstm_seq]).flatten()[0]) * 10
        predictions.append(round(score, 2))

    st.line_chart(pd.Series(predictions, index=[2025 + i for i in range(years)]))
    st.markdown(f"**Final Forecasted Score (Year {2025 + years - 1}):** {predictions[-1]} / 10")
