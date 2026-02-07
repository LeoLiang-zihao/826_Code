from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class BasicMetrics:
    auroc: float
    average_precision: float
    brier: float


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> BasicMetrics:
    return BasicMetrics(
        auroc=float(roc_auc_score(y_true, y_prob)),
        average_precision=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = torch.nn.Parameter(
            torch.tensor([np.log(init_temperature)], dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
        x = torch.as_tensor(logits, dtype=torch.float32).view(-1, 1)
        y = torch.as_tensor(labels, dtype=torch.float32).view(-1, 1)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(x), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())


def calibration_curve_quantile(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    data = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)})
    data["bin"] = pd.qcut(data["p"], q=n_bins, labels=False, duplicates="drop")
    grouped = data.groupby("bin", as_index=False).agg(
        mean_pred=("p", "mean"),
        frac_pos=("y", "mean"),
        n=("y", "size"),
    )
    return grouped


def calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(lp, y_true)
    slope = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    return slope, intercept


def specificity_at_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float
) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        idx = int(np.argmax(tpr))
    else:
        idx = int(valid[0])
    sensitivity = float(tpr[idx])
    specificity = float(1.0 - fpr[idx])
    threshold = float(thresholds[idx])
    return sensitivity, specificity, threshold


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 826,
) -> tuple[float, tuple[float, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    point = float(metric_fn(y_true, y_prob))
    samples: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if np.unique(y_b).shape[0] < 2:
            continue
        samples.append(float(metric_fn(y_b, p_b)))

    boots = np.array(samples, dtype=np.float64)
    lower = float(np.quantile(boots, 0.025))
    upper = float(np.quantile(boots, 0.975))
    return point, (lower, upper), boots
