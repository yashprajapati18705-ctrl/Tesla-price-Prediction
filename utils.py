import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as  np

def train_model(model, X_train, y_train):
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1
    )
    return history

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    # Plot
    plt.figure()
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.show()

    return mse

def predict_future(model, data, days):
    temp_input = list(data[-60:])
    output = []

    for _ in range(days):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1, 60, 1)

        yhat = model.predict(x_input)
        output.append(yhat[0][0])
        temp_input.append(yhat[0])

    return output