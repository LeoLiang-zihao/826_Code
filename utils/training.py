from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


def set_seed(seed: int = 826) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPRiskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CSRTensorDataset(Dataset):
    def __init__(self, X: csr_matrix, y: np.ndarray) -> None:
        if not isinstance(X, csr_matrix):
            raise TypeError("X must be a scipy.sparse.csr_matrix")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.X = X
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.X.getrow(idx).toarray().ravel().astype(np.float32, copy=False)
        x_tensor = torch.from_numpy(row)
        y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x_tensor, y_tensor


def build_dataloader(
    X: np.ndarray | csr_matrix,
    y: np.ndarray,
    *,
    batch_size: int = 200,
    shuffle: bool = False,
) -> DataLoader:
    if isinstance(X, csr_matrix):
        dataset: Dataset = CSRTensorDataset(X, y)
    else:
        x_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass
class TrainResult:
    history: dict[str, list[float]]
    best_state: dict[str, torch.Tensor]
    best_epoch: int


@dataclass
class TrainedModel:
    model: nn.Module
    result: TrainResult


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    l1_lambda: float = 0.0,
    device: str = "cpu",
) -> TrainResult:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model = model.to(device)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1_lambda > 0:
                l1_penalty = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l1_penalty = l1_penalty + param.abs().sum()
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, criterion, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(history=history, best_state=best_state, best_epoch=best_epoch)


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    *,
    device: str = "cpu",
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            losses.append(float(criterion(logits, yb).item()))
    return float(np.mean(losses))


def predict_logits(
    model: nn.Module, data_loader: DataLoader, *, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(1).cpu().numpy()
            preds.append(logits)
    return np.concatenate(preds)


def extract_linear_weights(model: nn.Module) -> np.ndarray:
    if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        return model.linear.weight.detach().cpu().numpy().squeeze(0)
    raise TypeError(
        "Model does not expose a top-level linear layer for coefficient extraction"
    )


def train_lr_sweep(
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rates: list[float],
    *,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> dict[float, TrainResult]:
    results: dict[float, TrainResult] = {}
    for lr in learning_rates:
        model = model_factory()
        result = train_model(
            model,
            train_loader,
            val_loader,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
        results[lr] = result
    return results


def train_lr_sweep_with_models(
    model_factory: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rates: list[float],
    *,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> dict[float, TrainedModel]:
    results: dict[float, TrainedModel] = {}
    for lr in learning_rates:
        model = model_factory()
        result = train_model(
            model,
            train_loader,
            val_loader,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
        results[lr] = TrainedModel(model=model, result=result)
    return results


def coefficient_table(
    weights: np.ndarray,
    feature_names: list[str],
    descriptions: pd.DataFrame,
    *,
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(weights) != len(feature_names):
        raise ValueError("weights and feature_names must have same length")

    frame = pd.DataFrame({"category": feature_names, "weight": weights})
    frame = frame.merge(descriptions, on="category", how="left")
    frame["long_title"] = frame["long_title"].fillna("Unknown")
    top = (
        frame.sort_values("weight", ascending=False).head(top_k).reset_index(drop=True)
    )
    bottom = (
        frame.sort_values("weight", ascending=True).head(top_k).reset_index(drop=True)
    )
    return top, bottom


def tune_l1_strength(
    model_factory: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    l1_values: list[float],
    *,
    zero_threshold: float = 1e-4,
    target_sparsity: float = 0.85,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[float, TrainedModel, pd.DataFrame]:
    records: list[dict[str, float]] = []
    trained: dict[float, TrainedModel] = {}

    for l1_lambda in l1_values:
        model = model_factory()
        result = train_model(
            model,
            train_loader,
            val_loader,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            l1_lambda=l1_lambda,
            device=device,
        )
        weights = extract_linear_weights(model)
        sparsity = float((np.abs(weights) <= zero_threshold).mean())
        best_val_loss = float(min(result.history["val_loss"]))
        trained[l1_lambda] = TrainedModel(model=model, result=result)
        records.append(
            {
                "l1_lambda": float(l1_lambda),
                "sparsity": sparsity,
                "best_val_loss": best_val_loss,
            }
        )

    table = pd.DataFrame(records).sort_values("l1_lambda").reset_index(drop=True)
    candidates = table[table["sparsity"] >= target_sparsity]
    if candidates.empty:
        chosen = float(table.iloc[table["sparsity"].argmax()]["l1_lambda"])
    else:
        chosen = float(candidates.iloc[0]["l1_lambda"])
    return chosen, trained[chosen], table
