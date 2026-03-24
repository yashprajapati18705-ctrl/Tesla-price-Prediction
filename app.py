import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_clean_data, scale_data, create_sequences
from models import build_lstm, build_rnn
from utils import train_model
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

# ======================
# HEADER
# ======================
st.title("📈 Tesla Stock Price Prediction")
st.markdown("Deep Learning Models: **SimpleRNN vs LSTM**")

# ======================
# LOAD DATA
# ======================
df = load_and_clean_data()

# Sidebar controls
st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox("Choose Model", ["LSTM", "SimpleRNN"])
days = st.sidebar.selectbox("Prediction Days", [1, 5, 10])

# ======================
# DATA PREVIEW
# ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.tail())

with col2:
    st.subheader("📉 Stock Trend")
    fig = plt.figure()
    plt.plot(df['Adj Close'])
    plt.title("Adj Close Price")
    st.pyplot(fig)

# ======================
# PREPROCESSING
# ======================
scaled_data, scaler = scale_data(df)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ======================
# MODEL TRAINING (CACHED)
# ======================
@st.cache_resource
def train_selected_model(model_name):
    if model_name == "LSTM":
        model = build_lstm((X_train.shape[1], 1))
    else:
        model = build_rnn((X_train.shape[1], 1))

    train_model(model, X_train, y_train)
    return model

# Train button
if st.button("🚀 Train Model"):
    model = train_selected_model(model_choice)
    st.success(f"{model_choice} Model Trained Successfully")

    # Save model
    model.save("model.h5")

# ======================
# EVALUATION
# ======================
try:
    model = load_model("model.h5")

    predictions = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predictions)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Model Performance")
        st.metric("MSE", round(mse, 6))

    with col4:
        st.subheader("📈 Prediction Graph")
        fig2 = plt.figure()
        plt.plot(y_test, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.legend()
        st.pyplot(fig2)

except:
    st.warning("Train the model first to see results.")

# ======================
# FUTURE PREDICTION
# ======================
def predict_future(model, data, days):
    temp_input = list(data[-60:])
    output = []

    for _ in range(days):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1, 60, 1)

        yhat = model.predict(x_input, verbose=0)
        output.append(yhat[0][0])
        temp_input.append(yhat[0])

    return output

st.subheader("🔮 Future Prediction")

if st.button("Predict Future Prices"):
    try:
        model = load_model("model.h5")
        result = predict_future(model, scaled_data, days)

        st.success(f"Prediction for next {days} days:")
        st.write(result)

        # Plot prediction
        fig3 = plt.figure()
        plt.plot(result, marker='o')
        plt.title("Future Prediction")
        st.pyplot(fig3)

    except:
        st.error("Please train the model first.")

# ======================
# FOOTER INSIGHTS
# ======================
st.markdown("---")
st.subheader("📌 Insights")

st.write("""
- LSTM generally performs better due to long-term memory capability.
- Stock prices are highly volatile and influenced by external factors.
- Model works best for short-term predictions (1–10 days).
""")