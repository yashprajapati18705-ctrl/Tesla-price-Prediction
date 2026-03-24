from preprocessing import load_and_clean_data, scale_data, create_sequences
from models import build_rnn, build_lstm
from utils import train_model, evaluate_model, predict_future
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
df = load_and_clean_data()

# Scaling
scaled_data, scaler = scale_data(df)

# Create sequences
window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# =====================
# SimpleRNN Model
# =====================
print("\nTraining SimpleRNN...\n")
rnn_model = build_rnn((X_train.shape[1], 1))
train_model(rnn_model, X_train, y_train)
rnn_mse = evaluate_model(rnn_model, X_test, y_test, scaler)

# =====================
# LSTM Model
# =====================
print("\nTraining LSTM...\n")
lstm_model = build_lstm((X_train.shape[1], 1))
train_model(lstm_model, X_train, y_train)
lstm_mse = evaluate_model(lstm_model, X_test, y_test, scaler)

# Comparison
print("\nModel Comparison:")
print(f"SimpleRNN MSE: {rnn_mse}")
print(f"LSTM MSE: {lstm_mse}")

if lstm_mse < rnn_mse:
    print("LSTM performs better")
else:
    print("SimpleRNN performs better")

# Multi-step Prediction
print("1 Day Prediction:", predict_future(lstm_model, scaled_data, 1))
print("5 Day Prediction:", predict_future(lstm_model, scaled_data, 5))
print("10 Day Prediction:", predict_future(lstm_model, scaled_data, 10))