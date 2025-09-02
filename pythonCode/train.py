import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds.extend(pred.cpu().numpy().flatten())
            gts.extend(y.cpu().numpy().flatten())
    return np.array(preds), np.array(gts)