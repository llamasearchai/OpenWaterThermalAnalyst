from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:  # Optional torch for LSTM support
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None  # type: ignore


@dataclass
class RFTrainingResult:
    model_path: str
    mae: float
    n_train: int
    n_test: int


def train_random_forest_regressor(csv_path: str, target: str, model_out: str, test_size: float = 0.2, random_state: int = 42) -> RFTrainingResult:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")
    y = df[target].values
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    joblib.dump(model, model_out)
    return RFTrainingResult(model_path=model_out, mae=mae, n_train=len(X_train), n_test=len(X_test))


class TimeSeriesLSTM(nn.Module):  # type: ignore
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for TimeSeriesLSTM. Install with extras: 'ml'.")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # type: ignore
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_lstm_regressor(
    csv_path: str,
    target: str,
    model_out: str,
    seq_len: int = 24,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Optional[str]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for LSTM training. Install with extras: 'ml'.")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")
    df = df.select_dtypes(include=[np.number]).dropna()

    y = df[target].values
    X = df.drop(columns=[target]).values

    # Create sequences
    def make_sequences(Xa, ya, L):
        Xs, ys = [], []
        for i in range(len(Xa) - L):
            Xs.append(Xa[i : i + L])
            ys.append(ya[i + L])
        return np.array(Xs), np.array(ys)

    Xs, ys = make_sequences(X, y, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesLSTM(input_dim=Xs.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_out)
    return model_out

