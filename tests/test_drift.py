"""Testes de drift detection — PSI e thresholds."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    PSI_CRITICAL_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    calculate_psi,
    detect_drift,
    should_retrain,
)


class TestCalculatePSI:
    """Testes para cálculo de PSI."""

    def test_identical_distributions_zero_psi(self):
        """Distribuições idênticas devem ter PSI próximo de 0."""
        data = np.random.normal(0, 1, 1000)
        psi = calculate_psi(data, data)
        assert psi < 0.01

    def test_different_distributions_high_psi(self):
        """Distribuições diferentes devem ter PSI alto."""
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # Média deslocada
        psi = calculate_psi(ref, cur)
        assert psi > PSI_CRITICAL_THRESHOLD

    def test_slightly_shifted_moderate_psi(self):
        """Distribuição levemente deslocada — PSI moderado."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0.3, 1, 1000)
        psi = calculate_psi(ref, cur)
        assert 0.0 < psi < 1.0

    def test_psi_non_negative(self):
        """PSI deve ser sempre não-negativo."""
        np.random.seed(42)
        ref = np.random.uniform(0, 10, 500)
        cur = np.random.uniform(2, 8, 500)
        psi = calculate_psi(ref, cur)
        assert psi >= 0

    def test_psi_symmetric_approximately(self):
        """PSI deve ser aproximadamente simétrico."""
        np.random.seed(42)
        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(1, 1, 1000)
        psi_ab = calculate_psi(a, b)
        psi_ba = calculate_psi(b, a)
        assert abs(psi_ab - psi_ba) < 0.1


class TestDetectDrift:
    """Testes para detect_drift."""

    def test_no_drift_stable(self):
        """Dados sem drift devem retornar status stable."""
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.normal(0, 1, 500), "f2": np.random.normal(5, 2, 500)})
        cur = pd.DataFrame({"f1": np.random.normal(0, 1, 200), "f2": np.random.normal(5, 2, 200)})
        result = detect_drift(ref, cur, log_to_mlflow=False)
        assert result["status"] == "stable"
        assert result["action"] == "none"

    def test_high_drift_critical(self):
        """Dados com drift alto devem retornar status critical."""
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.normal(0, 1, 500)})
        cur = pd.DataFrame({"f1": np.random.normal(5, 1, 200)})
        result = detect_drift(ref, cur, log_to_mlflow=False)
        assert result["status"] == "critical"
        assert result["action"] == "retrain"

    def test_result_has_required_keys(self):
        """Resultado deve ter todas as chaves necessárias."""
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"f1": np.random.normal(0, 1, 50)})
        result = detect_drift(ref, cur, log_to_mlflow=False)
        assert "status" in result
        assert "action" in result
        assert "max_psi" in result
        assert "psi_by_feature" in result

    def test_psi_by_feature_matches_columns(self):
        """PSI deve ser calculado para cada feature."""
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 50), "b": np.random.normal(0, 1, 50)})
        result = detect_drift(ref, cur, log_to_mlflow=False)
        assert "a" in result["psi_by_feature"]
        assert "b" in result["psi_by_feature"]


class TestShouldRetrain:
    """Testes para should_retrain."""

    def test_retrain_when_critical(self):
        """Deve retornar True quando action é retrain."""
        assert should_retrain({"action": "retrain"}) is True

    def test_no_retrain_when_stable(self):
        """Deve retornar False quando action é none."""
        assert should_retrain({"action": "none"}) is False

    def test_no_retrain_when_monitor(self):
        """Deve retornar False quando action é monitor."""
        assert should_retrain({"action": "monitor"}) is False
