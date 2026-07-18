"""Testes de modelos — baseline e métricas."""

import pytest

pytest.importorskip("torch", reason="torch não instalado")
pytest.importorskip("sklearn", reason="scikit-learn não instalado")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.baseline import MLPClassifier, train_mlp, train_random_forest  # noqa: E402


class TestRandomForest:
    """Testes para o baseline Random Forest."""

    def test_returns_model_and_metrics(self, sample_features: pd.DataFrame):
        """Deve retornar modelo treinado e métricas."""
        X = sample_features.drop(columns=["target"])
        y = sample_features["target"]

        model, metrics, split_info = train_random_forest(X, y)

        assert model is not None
        assert "auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_metrics_in_valid_range(self, sample_features: pd.DataFrame):
        """Métricas devem estar entre 0 e 1."""
        X = sample_features.drop(columns=["target"])
        y = sample_features["target"]

        _, metrics, _ = train_random_forest(X, y)

        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} fora do range: {value}"

    def test_predictions_are_binary(self, sample_features: pd.DataFrame):
        """Predições devem ser 0 ou 1."""
        X = sample_features.drop(columns=["target"])
        y = sample_features["target"]

        model, _, _ = train_random_forest(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0, 1})

    def test_reproducibility(self, sample_features: pd.DataFrame):
        """Resultados devem ser reprodutíveis com mesma seed."""
        X = sample_features.drop(columns=["target"])
        y = sample_features["target"]

        _, metrics1, _ = train_random_forest(X, y, random_state=42)
        _, metrics2, _ = train_random_forest(X, y, random_state=42)

        assert metrics1 == metrics2


class TestMLP:
    """Testes para o baseline MLP PyTorch."""

    def test_model_architecture(self):
        """Modelo deve ter a arquitetura correta."""
        model = MLPClassifier(input_dim=10, hidden_dims=[64, 32])
        assert model is not None

    def test_forward_pass(self):
        """Forward pass deve retornar tensor com shape correto."""
        import torch

        model = MLPClassifier(input_dim=5, hidden_dims=[16, 8])
        x = torch.randn(4, 5)
        output = model(x)

        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_train_returns_metrics(self, sample_features: pd.DataFrame):
        """Treinamento deve retornar métricas válidas."""
        X = sample_features.drop(columns=["target"])
        y = sample_features["target"]

        model, metrics, scaler = train_mlp(X, y, epochs=5, hidden_dims=[16, 8])

        assert model is not None
        assert scaler is not None
        assert "auc" in metrics
        assert 0.0 <= metrics["auc"] <= 1.0
