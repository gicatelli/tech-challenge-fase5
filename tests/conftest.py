"""Fixtures compartilhados para testes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Dados OHLCV sintéticos simulando preços de ações (200 dias)."""
    np.random.seed(42)
    n = 200

    # Gerar série de preços com random walk
    returns = np.random.normal(0.001, 0.02, n)
    prices = 30.0 * np.cumprod(1 + returns)

    # OHLCV realista
    high = prices * (1 + np.abs(np.random.normal(0.01, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0.01, 0.005, n)))
    open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n)
    volume = np.random.lognormal(mean=17, sigma=0.5, size=n).astype(int)

    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Dados sintéticos genéricos para testes."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.uniform(0, 1, n),
            "feature_2": np.random.uniform(1, 10, n),
            "feature_cat": np.random.choice(["A", "B", "C"], n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Features processadas para testes de modelo."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "feature_1": np.random.uniform(0, 1, n),
            "feature_2": np.random.uniform(1, 10, n),
            "feature_3": np.random.normal(0, 1, n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Dados de referência para drift detection."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(5, 2, n),
            "feature_3": np.random.uniform(0, 1, n),
        }
    )


@pytest.fixture
def drifted_data() -> pd.DataFrame:
    """Dados com drift para testes."""
    np.random.seed(123)
    n = 500
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(2, 1, n),  # Drift: média mudou de 0 para 2
            "feature_2": np.random.normal(5, 2, n),  # Sem drift
            "feature_3": np.random.uniform(0, 1, n),  # Sem drift
        }
    )
