"""Testes para src/data_collection.py — coleta de dados financeiros."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_collection import (
    collect_multiple_stocks,
    collect_stock_data,
    generate_synthetic_stock_data,
    get_stock_info,
)


class TestCollectStockData:
    """Testes para collect_stock_data."""

    @patch("src.data_collection.yf.download")
    def test_returns_dataframe(self, mock_download):
        """Deve retornar DataFrame com dados de ações."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        mock_df = pd.DataFrame(
            {
                "Open": np.random.uniform(28, 32, 100),
                "High": np.random.uniform(30, 35, 100),
                "Low": np.random.uniform(25, 30, 100),
                "Close": np.random.uniform(28, 33, 100),
                "Volume": np.random.randint(10000000, 50000000, 100),
            },
            index=dates,
        )
        mock_df.index.name = "Date"
        mock_download.return_value = mock_df

        result = collect_stock_data("PETR4.SA", "2024-01-01", "2024-06-01")

        assert isinstance(result, pd.DataFrame)
        assert "Close" in result.columns
        assert "Volume" in result.columns
        assert len(result) == 100

    @patch("src.data_collection.yf.download")
    def test_raises_on_empty_data(self, mock_download):
        """Deve levantar ValueError se nenhum dado for encontrado."""
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Nenhum dado encontrado"):
            collect_stock_data("INVALIDO.SA", "2024-01-01")

    @patch("src.data_collection.yf.download")
    def test_handles_multiindex_columns(self, mock_download):
        """Deve lidar com MultiIndex retornado pelo yfinance."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        data = {
            ("Close", "PETR4.SA"): np.random.uniform(28, 33, 50),
            ("Open", "PETR4.SA"): np.random.uniform(28, 32, 50),
            ("High", "PETR4.SA"): np.random.uniform(30, 35, 50),
            ("Low", "PETR4.SA"): np.random.uniform(25, 30, 50),
            ("Volume", "PETR4.SA"): np.random.randint(10000000, 50000000, 50),
        }
        mock_df = pd.DataFrame(data, index=dates)
        mock_df.index.name = "Date"
        mock_download.return_value = mock_df

        result = collect_stock_data("PETR4.SA", "2024-01-01")
        assert not isinstance(result.columns, pd.MultiIndex)
        assert "Close" in result.columns

    @patch("src.data_collection.yf.download")
    def test_default_end_date_is_today(self, mock_download):
        """Deve usar data de hoje como end_date padrão."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        mock_df = pd.DataFrame(
            {"Close": np.random.uniform(28, 33, 10), "Open": [30] * 10,
             "High": [32] * 10, "Low": [28] * 10, "Volume": [1000000] * 10},
            index=dates,
        )
        mock_df.index.name = "Date"
        mock_download.return_value = mock_df

        result = collect_stock_data("PETR4.SA", "2024-01-01")
        assert result is not None
        mock_download.assert_called_once()


class TestGetStockInfo:
    """Testes para get_stock_info."""

    @patch("src.data_collection.yf.Ticker")
    def test_returns_info_dict(self, mock_ticker):
        """Deve retornar dicionário com informações da ação."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "longName": "Petróleo Brasileiro S.A. - Petrobras",
            "sector": "Energy",
            "industry": "Oil & Gas",
            "currency": "BRL",
            "marketCap": 500000000000,
        }
        mock_ticker.return_value = mock_ticker_instance

        result = get_stock_info("PETR4.SA")

        assert result["symbol"] == "PETR4.SA"
        assert result["name"] == "Petróleo Brasileiro S.A. - Petrobras"
        assert result["sector"] == "Energy"
        assert result["currency"] == "BRL"

    @patch("src.data_collection.yf.Ticker")
    def test_handles_missing_fields(self, mock_ticker):
        """Deve retornar 'N/A' para campos ausentes."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance

        result = get_stock_info("PETR4.SA")

        assert result["name"] == "N/A"
        assert result["sector"] == "N/A"


class TestGenerateSyntheticStockData:
    """Testes para generate_synthetic_stock_data."""

    def test_generates_correct_columns(self, tmp_path):
        """Deve gerar DataFrame com colunas OHLCV."""
        df = generate_synthetic_stock_data(
            "PETR4.SA",
            start_date="2024-01-02",
            end_date="2024-03-01",
            output_dir=tmp_path,
        )
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns

    def test_generates_realistic_prices(self, tmp_path):
        """Preços devem ser positivos e razoáveis."""
        df = generate_synthetic_stock_data(
            "PETR4.SA",
            start_date="2024-01-02",
            end_date="2024-06-01",
            output_dir=tmp_path,
        )
        assert (df["Close"] > 0).all()
        assert (df["High"] >= df["Low"]).all()
        assert (df["Volume"] > 0).all()

    def test_saves_csv_file(self, tmp_path):
        """Deve salvar CSV no diretório de saída."""
        generate_synthetic_stock_data(
            "PETR4.SA",
            start_date="2024-01-02",
            end_date="2024-02-01",
            output_dir=tmp_path,
        )
        csv_file = tmp_path / "PETR4_SA_historico.csv"
        assert csv_file.exists()

    def test_different_symbols_have_different_params(self, tmp_path):
        """Símbolos diferentes devem gerar dados com parâmetros diferentes."""
        df_petr = generate_synthetic_stock_data(
            "PETR4.SA", "2024-01-02", "2024-02-01", tmp_path
        )
        df_vale = generate_synthetic_stock_data(
            "VALE3.SA", "2024-01-02", "2024-02-01", tmp_path
        )
        # VALE3 começa com preço mais alto que PETR4
        assert df_vale["Close"].iloc[0] != df_petr["Close"].iloc[0]

    def test_uses_business_days(self, tmp_path):
        """Deve usar apenas dias úteis."""
        df = generate_synthetic_stock_data(
            "PETR4.SA", "2024-01-02", "2024-01-15", tmp_path
        )
        # Verificar que o index é DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
        # Dias úteis apenas (sem sábado=5 e domingo=6)
        weekdays = df.index.dayofweek
        assert (weekdays < 5).all()

    def test_reproducible_with_seed(self, tmp_path):
        """Dados devem ser reprodutíveis (seed fixa internamente)."""
        df1 = generate_synthetic_stock_data(
            "PETR4.SA", "2024-01-02", "2024-02-01", tmp_path
        )
        df2 = generate_synthetic_stock_data(
            "PETR4.SA", "2024-01-02", "2024-02-01", tmp_path
        )
        pd.testing.assert_frame_equal(df1, df2)


class TestCollectMultipleStocks:
    """Testes para collect_multiple_stocks."""

    @patch("src.data_collection.collect_stock_data")
    @patch("src.data_collection.time.sleep")
    def test_collects_all_symbols(self, mock_sleep, mock_collect, tmp_path):
        """Deve coletar dados de todos os símbolos."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        mock_df = pd.DataFrame(
            {
                "Open": [30.0] * 50,
                "High": [32.0] * 50,
                "Low": [28.0] * 50,
                "Close": [30.0] * 50,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )
        mock_df.index.name = "Date"
        mock_collect.return_value = mock_df

        results = collect_multiple_stocks(
            symbols=["PETR4.SA", "VALE3.SA"],
            start_date="2024-01-01",
            output_dir=tmp_path,
        )

        assert "PETR4.SA" in results
        assert "VALE3.SA" in results
        assert len(results) == 2

    @patch("src.data_collection.collect_stock_data")
    @patch("src.data_collection.time.sleep")
    def test_saves_metadata(self, mock_sleep, mock_collect, tmp_path):
        """Deve salvar metadata da coleta em JSON."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        mock_df = pd.DataFrame(
            {"Open": [30] * 20, "High": [32] * 20, "Low": [28] * 20,
             "Close": [30] * 20, "Volume": [1000000] * 20},
            index=dates,
        )
        mock_df.index.name = "Date"
        mock_collect.return_value = mock_df

        collect_multiple_stocks(
            symbols=["PETR4.SA"],
            start_date="2024-01-01",
            output_dir=tmp_path,
        )

        metadata_file = tmp_path / "collection_metadata.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert "symbols" in metadata
        assert "collected_at" in metadata
        assert "records" in metadata

    @patch("src.data_collection.collect_stock_data")
    @patch("src.data_collection.time.sleep")
    def test_handles_failed_symbol(self, mock_sleep, mock_collect, tmp_path):
        """Deve continuar se um símbolo falhar."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        mock_df = pd.DataFrame(
            {"Open": [30] * 20, "High": [32] * 20, "Low": [28] * 20,
             "Close": [30] * 20, "Volume": [1000000] * 20},
            index=dates,
        )
        mock_df.index.name = "Date"

        # Primeiro símbolo falha, segundo sucesso
        mock_collect.side_effect = [ValueError("not found"), mock_df]

        results = collect_multiple_stocks(
            symbols=["INVALIDO.SA", "PETR4.SA"],
            start_date="2024-01-01",
            output_dir=tmp_path,
        )

        assert "INVALIDO.SA" not in results
        assert "PETR4.SA" in results
