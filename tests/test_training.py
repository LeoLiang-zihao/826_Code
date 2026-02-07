from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from utils.training import (
    CSRTensorDataset,
    LogisticRegressionModel,
    MLPRiskModel,
    build_dataloader,
    coefficient_table,
    extract_linear_weights,
    train_lr_sweep_with_models,
    train_model,
    tune_l1_strength,
)


def _synthetic_binary_problem(n: int = 500, d: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,))
    logits = X @ w
    y = (logits > 0).astype(np.float32)
    return X, y


def test_logistic_output_shape():
    model = LogisticRegressionModel(10)
    x = torch.randn(32, 10)
    out = model(x)
    assert out.shape == (32, 1)


def test_training_loss_decreases():
    X, y = _synthetic_binary_problem()
    train_loader = build_dataloader(X[:400], y[:400], batch_size=64, shuffle=True)
    val_loader = build_dataloader(X[400:], y[400:], batch_size=64, shuffle=False)
    model = LogisticRegressionModel(X.shape[1])
    result = train_model(model, train_loader, val_loader, max_epochs=20, patience=20)
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]


def test_extract_linear_weights_dimension():
    model = LogisticRegressionModel(11)
    w = extract_linear_weights(model)
    assert w.shape == (11,)


def test_l1_regularization_increases_sparsity():
    X, y = _synthetic_binary_problem()
    train_loader = build_dataloader(X[:400], y[:400], batch_size=64, shuffle=True)
    val_loader = build_dataloader(X[400:], y[400:], batch_size=64, shuffle=False)

    plain = LogisticRegressionModel(X.shape[1])
    sparse = LogisticRegressionModel(X.shape[1])
    train_model(
        plain, train_loader, val_loader, max_epochs=25, patience=25, l1_lambda=0.0
    )
    train_model(
        sparse, train_loader, val_loader, max_epochs=25, patience=25, l1_lambda=1e-2
    )

    plain_zero = (np.abs(extract_linear_weights(plain)) < 1e-4).sum()
    sparse_zero = (np.abs(extract_linear_weights(sparse)) < 1e-4).sum()
    assert sparse_zero >= plain_zero


def test_mlp_shapes():
    model = MLPRiskModel(input_dim=15, hidden_dim=100)
    out = model(torch.randn(8, 15))
    assert out.shape == (8, 1)


def test_build_dataloader_accepts_csr():
    X, y = _synthetic_binary_problem(n=50, d=8)
    X_csr = csr_matrix(X)
    loader = build_dataloader(X_csr, y, batch_size=16, shuffle=False)
    xb, yb = next(iter(loader))
    assert xb.shape[1] == 8
    assert yb.shape[1] == 1


def test_coefficient_table_outputs_top_and_bottom():
    weights = np.array([0.5, -0.2, 1.2, -1.3])
    features = ["D_A10", "P_B20", "D_C30", "P_D40"]
    desc = pd.DataFrame(
        {
            "category": features,
            "long_title": ["a", "b", "c", "d"],
        }
    )
    top, bottom = coefficient_table(weights, features, desc, top_k=2)
    assert top.iloc[0]["category"] == "D_C30"
    assert bottom.iloc[0]["category"] == "P_D40"


def test_tune_l1_strength_returns_table_and_choice():
    X, y = _synthetic_binary_problem(n=240, d=12)
    train_loader = build_dataloader(X[:180], y[:180], batch_size=32, shuffle=True)
    val_loader = build_dataloader(X[180:], y[180:], batch_size=32, shuffle=False)
    chosen, trained, table = tune_l1_strength(
        lambda: LogisticRegressionModel(X.shape[1]),
        train_loader,
        val_loader,
        [0.0, 1e-4, 1e-3],
        max_epochs=10,
        patience=3,
        target_sparsity=0.2,
    )
    assert chosen in [0.0, 1e-4, 1e-3]
    assert trained.model is not None
    assert len(table) == 3


def test_train_lr_sweep_with_models_returns_models():
    X, y = _synthetic_binary_problem(n=220, d=10)
    train_loader = build_dataloader(X[:180], y[:180], batch_size=32, shuffle=True)
    val_loader = build_dataloader(X[180:], y[180:], batch_size=32, shuffle=False)
    out = train_lr_sweep_with_models(
        lambda: LogisticRegressionModel(X.shape[1]),
        train_loader,
        val_loader,
        [1e-3, 1e-2],
        max_epochs=8,
        patience=3,
    )
    assert set(out.keys()) == {1e-3, 1e-2}
    assert all(hasattr(item, "model") for item in out.values())


def test_csr_tensor_dataset_type_guard():
    y = np.array([0.0, 1.0], dtype=np.float32)
    try:
        CSRTensorDataset(np.array([[0.0], [1.0]], dtype=np.float32), y)
    except TypeError:
        assert True
        return
    raise AssertionError("Expected TypeError for non-CSR input")
