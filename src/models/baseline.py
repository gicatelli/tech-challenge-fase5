"""Modelos baseline — Scikit-Learn + MLP PyTorch.

Implementa dois baselines:
1. RandomForest (Scikit-Learn) — interpretável, rápido
2. MLP (PyTorch) — demonstra competência em deep learning
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============================================
# Baseline 1: Random Forest (Scikit-Learn)
# ============================================


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, float], dict[str, Any]]:
    """Treina Random Forest e retorna modelo + métricas.

    Args:
        X: Features de treinamento.
        y: Target.
        params: Hiperparâmetros do modelo.
        test_size: Proporção de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (modelo treinado, métricas, split info).
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test, y_pred, y_proba)

    split_info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    }

    logger.info(
        "RandomForest treinado: AUC=%.4f, F1=%.4f",
        metrics["auc"],
        metrics["f1"],
    )

    return model, metrics, split_info


# ============================================
# Baseline 2: MLP (PyTorch)
# ============================================


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron para classificação binária."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.3):
        """Inicializa MLP.

        Args:
            input_dim: Número de features de entrada.
            hidden_dims: Dimensões das camadas ocultas.
            dropout: Taxa de dropout.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor de entrada.

        Returns:
            Probabilidades de classe positiva.
        """
        return self.network(x)


def train_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    hidden_dims: list[int] | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[MLPClassifier, dict[str, float], StandardScaler]:
    """Treina MLP PyTorch e retorna modelo + métricas.

    Args:
        X: Features de treinamento.
        y: Target.
        hidden_dims: Dimensões das camadas ocultas.
        epochs: Número de épocas.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        test_size: Proporção de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (modelo treinado, métricas, scaler).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tensores
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)

    # Modelo
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims or [64, 32, 16],
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i : i + batch_size]
            batch_y = y_train_t[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.debug("Epoch %d/%d, Loss: %.4f", epoch + 1, epochs, loss.item())

    # Avaliação
    model.eval()
    with torch.no_grad():
        y_proba = model(X_test_t).numpy().flatten()
        y_pred = (y_proba >= 0.5).astype(int)

    metrics = _compute_metrics(y_test, y_pred, y_proba)

    logger.info(
        "MLP treinado: AUC=%.4f, F1=%.4f",
        metrics["auc"],
        metrics["f1"],
    )

    return model, metrics, scaler


# ============================================
# Utilitários
# ============================================


def _compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Computa métricas padronizadas de classificação.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições binárias.
        y_proba: Probabilidades.

    Returns:
        Dicionário com métricas.
    """
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
