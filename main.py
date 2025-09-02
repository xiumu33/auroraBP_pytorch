import torch
from torch import optim
import matplotlib.pyplot as plt
from data import prepare_data
from dataset import BPDataset
from torch.utils.data import DataLoader
from model import BPModel
from train import train_model, evaluate_model
from sklearn.model_selection import LeaveOneGroupOut
from evaluate import compute_metrics, plot_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X, y, pids = prepare_data()
groups = pids
logo = LeaveOneGroupOut()

predictions = []
ground_truths = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_loader = DataLoader(BPDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(BPDataset(X_test, y_test), batch_size=64)

    model = BPModel(input_dim=X.shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        train_model(model, train_loader, criterion, optimizer, device)

    preds, gts = evaluate_model(model, test_loader, device)
    predictions.extend(preds)
    ground_truths.extend(gts)

# 可视化
plt.scatter(ground_truths, predictions)
plt.plot([-60, 40], [-60, 40], 'r')
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.title("PyTorch Model: Leave-One-Participant-Out")
plt.show()

# 计算指标
metrics = compute_metrics(ground_truths, predictions)
print("Evaluation Metrics:", metrics)

# 可视化
plot_predictions(ground_truths, predictions)