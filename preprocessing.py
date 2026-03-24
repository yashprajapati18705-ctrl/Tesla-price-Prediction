import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data():
    df = pd.read_csv("TSLA.csv")

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Handle missing values (important for time series)
    df = df.fillna(0)  # forward fill

    return df

def scale_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Adj Close']])
    return scaled_data, scaler

def create_sequences(data, window_size):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)