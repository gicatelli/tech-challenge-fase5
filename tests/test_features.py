"""Testes de feature engineering — schema contracts."""

import pandas as pd

from src.features.feature_engineering import compute_features, encode_categorical


class TestComputeFeatures:
    """Testes para compute_features."""

    def test_output_has_expected_columns(self, sample_data: pd.DataFrame):
        """Features de saída devem conter colunas derivadas."""
        result = compute_features(sample_data, validate=False)
        assert "feature_1_x_feature_2" in result.columns
        assert "feature_1_squared" in result.columns
        assert "feature_2_log" in result.columns

    def test_no_nulls(self, sample_data: pd.DataFrame):
        """Nenhuma feature numérica pode ter null após transformação."""
        result = compute_features(sample_data, validate=False)
        numeric_cols = result.select_dtypes(include=["float64", "int64"]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_row_count_preserved(self, sample_data: pd.DataFrame):
        """Número de registros deve ser preservado."""
        result = compute_features(sample_data, validate=False)
        assert len(result) == len(sample_data)

    def test_feature_interaction_correct(self, sample_data: pd.DataFrame):
        """Feature de interação deve ser o produto das originais."""
        result = compute_features(sample_data, validate=False)
        expected = sample_data["feature_1"] * sample_data["feature_2"]
        pd.testing.assert_series_equal(
            result["feature_1_x_feature_2"],
            expected,
            check_names=False,
        )

    def test_squared_feature_range(self, sample_data: pd.DataFrame):
        """Feature quadrática deve estar entre 0 e 1."""
        result = compute_features(sample_data, validate=False)
        assert result["feature_1_squared"].min() >= 0
        assert result["feature_1_squared"].max() <= 1


class TestEncodeCategorical:
    """Testes para encode_categorical."""

    def test_removes_original_column(self, sample_data: pd.DataFrame):
        """Coluna original deve ser removida após encoding."""
        result = encode_categorical(sample_data, columns=["feature_cat"])
        assert "feature_cat" not in result.columns

    def test_creates_dummy_columns(self, sample_data: pd.DataFrame):
        """Deve criar colunas dummy."""
        result = encode_categorical(sample_data, columns=["feature_cat"])
        dummy_cols = [c for c in result.columns if c.startswith("feature_cat_")]
        assert len(dummy_cols) >= 1  # drop_first=True remove uma

    def test_preserves_row_count(self, sample_data: pd.DataFrame):
        """Número de registros deve ser preservado."""
        result = encode_categorical(sample_data, columns=["feature_cat"])
        assert len(result) == len(sample_data)
