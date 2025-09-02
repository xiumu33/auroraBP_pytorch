import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "RMSE": rmse,
        "MAE": mae
    }

def plot_predictions(y_true, y_pred, title="Model Predictions vs Ground Truth"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([-60, 40], [-60, 40], 'r')
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()