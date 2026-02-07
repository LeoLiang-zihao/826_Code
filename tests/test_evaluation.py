from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.evaluation import (
    TemperatureScaler,
    bootstrap_metric_ci,
    calibration_curve_quantile,
    compute_basic_metrics,
    sigmoid,
    specificity_at_sensitivity,
)


def test_basic_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.1, 0.9, 1.0])
    metrics = compute_basic_metrics(y, p)
    assert metrics.auroc == 1.0
    assert metrics.average_precision == 1.0


def test_temperature_scaling_properties():
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.float32)
    logits = np.array([-3.0, -2.0, 4.0, 3.0, 2.5, -1.5], dtype=np.float32)
    scaler = TemperatureScaler()
    logits_before = logits.copy()

    raw_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    ).item()

    t = scaler.fit(logits, y, max_iter=50)
    assert t > 0
    assert np.array_equal(logits, logits_before)

    cal_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits / t, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    ).item()
    assert cal_nll <= raw_nll + 1e-8

    raw_auc = roc_auc_score(y, sigmoid(logits))
    cal_auc = roc_auc_score(y, sigmoid(logits / t))
    assert abs(raw_auc - cal_auc) < 1e-12


def test_quantile_calibration_bins_are_nonempty():
    y = np.array([0] * 50 + [1] * 50)
    p = np.linspace(0.01, 0.99, 100)
    curve = calibration_curve_quantile(y, p, n_bins=10)
    assert len(curve) == 10
    assert (curve["n"] > 0).all()


def test_specificity_at_target_sensitivity_returns_valid_values():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])
    sens, spec, threshold = specificity_at_sensitivity(y, p, target_sensitivity=0.8)
    assert 0.0 <= sens <= 1.0
    assert 0.0 <= spec <= 1.0
    assert np.isfinite(threshold)


def test_bootstrap_ci_shape():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.9, 0.1, 0.6])

    def metric(a, b):
        return roc_auc_score(a, b)

    point, (lo, hi), boots = bootstrap_metric_ci(y, p, metric, n_bootstrap=200, seed=99)
    assert np.isfinite(point)
    assert lo <= hi
    assert boots.ndim == 1
    assert len(boots) > 0
