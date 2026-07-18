"""Testes de feature engineering — indicadores técnicos financeiros."""

import pytest

pytest.importorskip("pandera", reason="pandera não instalado")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.feature_engineering import (  # noqa: E402
    compute_features,
    compute_log_returns,
    compute_macd,
    compute_rsi,
    compute_sma,
    normalize_volume,
    prepare_sequences,
)


class TestComputeFeatures:
    """Testes para o pipeline completo de features."""

    def test_output_has_expected_columns(self, sample_ohlcv: pd.DataFrame):
        """Features de saída devem conter indicadores técnicos."""
        result = compute_features(sample_ohlcv, validate=False)
        expected_cols = [
            "close", "sma_7", "sma_30", "sma_90",
            "rsi_14", "macd", "macd_signal",
            "bb_upper", "bb_lower",
            "log_return", "volatility_30", "volume_norm",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Coluna {col} ausente"

    def test_no_nulls_after_processing(self, sample_ohlcv: pd.DataFrame):
        """Nenhuma feature pode ter NaN após processamento (warm-up removido)."""
        result = compute_features(sample_ohlcv, validate=False)
        assert result.isnull().sum().sum() == 0

    def test_rows_reduced_by_warmup(self, sample_ohlcv: pd.DataFrame):
        """Linhas devem ser reduzidas pelo warm-up dos indicadores (~90 dias)."""
        result = compute_features(sample_ohlcv, validate=False)
        assert len(result) < len(sample_ohlcv)
        assert len(result) > len(sample_ohlcv) * 0.4  # Pelo menos 40% preservado

    def test_close_prices_positive(self, sample_ohlcv: pd.DataFrame):
        """Preços de fechamento devem ser sempre positivos."""
        result = compute_features(sample_ohlcv, validate=False)
        assert (result["close"] > 0).all()

    def test_rsi_range(self, sample_ohlcv: pd.DataFrame):
        """RSI deve estar entre 0 e 100."""
        result = compute_features(sample_ohlcv, validate=False)
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_volatility_non_negative(self, sample_ohlcv: pd.DataFrame):
        """Volatilidade deve ser não-negativa."""
        result = compute_features(sample_ohlcv, validate=False)
        assert (result["volatility_30"] >= 0).all()

    def test_volume_norm_non_negative(self, sample_ohlcv: pd.DataFrame):
        """Volume normalizado deve ser não-negativo."""
        result = compute_features(sample_ohlcv, validate=False)
        assert (result["volume_norm"] >= 0).all()

    def test_schema_validation_passes(self, sample_ohlcv: pd.DataFrame):
        """Schema de saída deve validar corretamente."""
        # Não deve levantar exceção
        result = compute_features(sample_ohlcv, validate=True)
        assert len(result) > 0


class TestIndividualIndicators:
    """Testes para indicadores individuais."""

    def test_sma_correct_values(self):
        """SMA deve ser a média da janela."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = compute_sma(series, window=3)
        assert sma.iloc[2] == pytest.approx(2.0)  # mean(1,2,3)
        assert sma.iloc[4] == pytest.approx(4.0)  # mean(3,4,5)

    def test_rsi_boundaries(self):
        """RSI deve ficar entre 0 e 100."""
        np.random.seed(42)
        prices = pd.Series(30.0 * np.cumprod(1 + np.random.normal(0, 0.02, 100)))
        rsi = compute_rsi(prices, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_returns_three_series(self):
        """MACD deve retornar 3 séries (line, signal, histogram)."""
        prices = pd.Series(range(50), dtype=float)
        macd_line, signal_line, histogram = compute_macd(prices)
        assert len(macd_line) == 50
        assert len(signal_line) == 50
        assert len(histogram) == 50

    def test_log_returns_first_is_nan(self):
        """Primeiro retorno log deve ser NaN."""
        series = pd.Series([100.0, 105.0, 103.0])
        returns = compute_log_returns(series)
        assert pd.isna(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(np.log(105 / 100))

    def test_normalize_volume_ratio(self):
        """Volume normalizado = volume / média móvel."""
        volume = pd.Series([100, 100, 100, 200, 100], dtype=float)
        norm = normalize_volume(volume, window=3)
        # No index 3: volume=200, média(100,100,200)=133.33 → ratio≈1.5
        assert norm.iloc[3] == pytest.approx(200 / 133.333, rel=0.01)


class TestPrepareSequences:
    """Testes para preparação de sequências LSTM."""

    def test_output_shapes(self, sample_ohlcv: pd.DataFrame):
        """Shapes de saída devem ser consistentes."""
        features = compute_features(sample_ohlcv, validate=False)
        seq_len = 30
        X, y = prepare_sequences(features, target_col="close", sequence_length=seq_len)

        n_samples = len(features) - seq_len
        n_features = len(features.columns)

        assert X.shape == (n_samples, seq_len, n_features)
        assert y.shape == (n_samples,)

    def test_target_values_positive(self, sample_ohlcv: pd.DataFrame):
        """Target (close) deve ser sempre positivo."""
        features = compute_features(sample_ohlcv, validate=False)
        _, y = prepare_sequences(features, target_col="close", sequence_length=30)
        assert (y > 0).all()
