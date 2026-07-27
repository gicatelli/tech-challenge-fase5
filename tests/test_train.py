"""Testes para src/models/train.py — pipeline de treinamento."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.models.train import (
    compute_regression_metrics,
    load_config,
    select_champion,
)


class TestComputeRegressionMetrics:
    """Testes para compute_regression_metrics."""

    def test_perfect_predictions(self):
        """Predições perfeitas devem ter MAE=0, RMSE=0, R2=1."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0.0

    def test_imperfect_predictions(self):
        """Predições imperfeitas devem ter métricas positivas."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 18.0, 33.0, 37.0, 52.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0
        assert metrics["r2"] < 1.0

    def test_returns_all_required_keys(self):
        """Deve retornar todas as métricas esperadas."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics

    def test_rmse_greater_or_equal_mae(self):
        """RMSE deve ser >= MAE (propriedade matemática)."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([15.0, 18.0, 35.0, 38.0, 55.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["rmse"] >= metrics["mae"]

    def test_metrics_are_float(self):
        """Todas as métricas devem ser float."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = compute_regression_metrics(y_true, y_pred)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} não é float: {type(value)}"


class TestSelectChampion:
    """Testes para select_champion."""

    def test_lstm_wins_when_lower_rmse(self):
        """LSTM deve vencer quando tem RMSE menor."""
        lstm_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        rf_metrics = {"mae": 3.5, "rmse": 6.0, "mape": 12.0, "r2": 0.80}
        result = select_champion(lstm_metrics, rf_metrics, 0.5, 0.1)
        assert result["champion"] == "lstm"
        assert "LSTM" in result["reason"]

    def test_rf_wins_when_lower_rmse(self):
        """RF deve vencer quando tem RMSE menor."""
        lstm_metrics = {"mae": 4.0, "rmse": 7.0, "mape": 15.0, "r2": 0.70}
        rf_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        result = select_champion(lstm_metrics, rf_metrics, 0.5, 0.1)
        assert result["champion"] == "random_forest"
        assert "Random Forest" in result["reason"]

    def test_returns_both_metrics(self):
        """Resultado deve conter métricas de ambos os modelos."""
        lstm_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        rf_metrics = {"mae": 3.5, "rmse": 6.0, "mape": 12.0, "r2": 0.80}
        result = select_champion(lstm_metrics, rf_metrics, 0.5, 0.1)
        assert "lstm" in result
        assert "random_forest" in result
        assert "champion" in result
        assert "reason" in result

    def test_latency_included(self):
        """Latência deve ser incluída nos resultados."""
        lstm_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        rf_metrics = {"mae": 3.5, "rmse": 6.0, "mape": 12.0, "r2": 0.80}
        result = select_champion(lstm_metrics, rf_metrics, 0.47, 0.11)
        assert result["lstm"]["latency_ms"] == 0.47
        assert result["random_forest"]["latency_ms"] == 0.11

    def test_equal_rmse_rf_wins(self):
        """Com RMSE igual, RF deve vencer (simplicidade)."""
        lstm_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        rf_metrics = {"mae": 3.0, "rmse": 5.0, "mape": 10.0, "r2": 0.85}
        result = select_champion(lstm_metrics, rf_metrics, 0.5, 0.1)
        assert result["champion"] == "random_forest"


class TestLoadConfig:
    """Testes para load_config."""

    def test_loads_yaml_file(self, tmp_path):
        """Deve carregar arquivo YAML corretamente."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "lstm_optimized:\n  hidden_size_1: 128\n  sequence_length: 60\n"
        )
        config = load_config(str(config_file))
        assert config["lstm_optimized"]["hidden_size_1"] == 128
        assert config["lstm_optimized"]["sequence_length"] == 60

    def test_returns_dict(self, tmp_path):
        """Deve retornar dicionário."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")
        config = load_config(str(config_file))
        assert isinstance(config, dict)


class TestTrainRandomForestRegressor:
    """Testes para train_random_forest_regressor com MLflow mockado."""

    @patch("src.models.train.mlflow")
    def test_trains_and_returns_metrics(self, mock_mlflow):
        """Deve treinar RF e retornar métricas válidas."""
        from src.models.train import train_random_forest_regressor

        # Setup mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Dados sintéticos
        np.random.seed(42)
        n_train, n_test, n_features = 100, 20, 10
        X_train = np.random.randn(n_train, n_features)
        y_train = np.random.uniform(0, 1, n_train)
        X_test = np.random.randn(n_test, n_features)
        y_test = np.random.uniform(0, 1, n_test)

        config = {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": 1,
            }
        }

        run_id, metrics, latency = train_random_forest_regressor(
            X_train, y_train, X_test, y_test, config,
            close_min=10.0, close_max=50.0,
        )

        assert run_id == "test-run-123"
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert latency > 0

    @patch("src.models.train.mlflow")
    def test_metrics_in_real_scale(self, mock_mlflow):
        """Métricas devem estar na escala real (R$)."""
        from src.models.train import train_random_forest_regressor

        mock_run = MagicMock()
        mock_run.info.run_id = "run-456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        np.random.seed(42)
        X_train = np.random.randn(80, 5)
        y_train = np.random.uniform(0, 1, 80)  # Normalizado
        X_test = np.random.randn(20, 5)
        y_test = np.random.uniform(0, 1, 20)

        config = {"random_forest": {"n_estimators": 5, "max_depth": 3, "random_state": 42}}

        _, metrics, _ = train_random_forest_regressor(
            X_train, y_train, X_test, y_test, config,
            close_min=20.0, close_max=50.0,
        )

        # MAE na escala real deve ser > 0 e razoável para faixa [20, 50]
        assert metrics["mae"] > 0
        assert metrics["mae"] < 30  # não pode ser maior que o range


class TestTrainLSTM:
    """Testes para train_lstm com dependências mockadas."""

    @patch("src.models.train.train_lstm_model")
    @patch("src.models.train.mlflow")
    def test_trains_and_returns_metrics(self, mock_mlflow, mock_train_model):
        """Deve treinar LSTM e retornar métricas."""
        from src.models.train import train_lstm

        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = "lstm-run-789"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Mock modelo LSTM
        import torch

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.cpu = MagicMock(return_value=mock_model)

        # Simular forward pass
        n_test = 20
        mock_output = torch.FloatTensor(np.random.uniform(0, 1, n_test).reshape(-1, 1))
        mock_model.return_value = mock_output

        mock_train_model.return_value = (mock_model, 0.01)

        # Dados
        np.random.seed(42)
        X_train = np.random.randn(100, 60, 10)
        y_train = np.random.uniform(0, 1, 100)
        X_test = np.random.randn(n_test, 60, 10)
        y_test = np.random.uniform(0, 1, n_test)

        config = {
            "lstm_optimized": {
                "hidden_size_1": 64,
                "hidden_size_2": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 16,
                "sequence_length": 60,
            }
        }

        run_id, metrics, latency = train_lstm(
            X_train, y_train, X_test, y_test, config,
            close_min=20.0, close_max=50.0,
        )

        assert run_id == "lstm-run-789"
        assert "mae" in metrics
        assert "rmse" in metrics
        assert latency >= 0
        mock_train_model.assert_called_once()

    @patch("src.models.train.train_lstm_model")
    @patch("src.models.train.mlflow")
    def test_lstm_default_config(self, mock_mlflow, mock_train_model):
        """Deve usar valores padrão quando config parcial."""
        from src.models.train import train_lstm

        import torch

        mock_run = MagicMock()
        mock_run.info.run_id = "lstm-default"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.cpu = MagicMock(return_value=mock_model)
        n_test = 10
        mock_model.return_value = torch.FloatTensor(
            np.random.uniform(0, 1, n_test).reshape(-1, 1)
        )
        mock_train_model.return_value = (mock_model, 0.02)

        np.random.seed(42)
        X_train = np.random.randn(50, 60, 5)
        y_train = np.random.uniform(0, 1, 50)
        X_test = np.random.randn(n_test, 60, 5)
        y_test = np.random.uniform(0, 1, n_test)

        # Config vazia — deve usar defaults
        config = {}

        run_id, metrics, latency = train_lstm(
            X_train, y_train, X_test, y_test, config,
            close_min=10.0, close_max=60.0,
        )

        assert run_id == "lstm-default"
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


class TestRunTrainingPipeline:
    """Testes para run_training_pipeline com mocks."""

    @patch("src.models.train.register_and_promote")
    @patch("src.models.train.train_random_forest_regressor")
    @patch("src.models.train.train_lstm")
    @patch("src.models.train.mlflow")
    @patch("src.models.train.compute_features")
    @patch("src.models.train.prepare_sequences")
    @patch("src.models.train.load_config")
    @patch("pandas.read_csv")
    def test_pipeline_returns_comparison(
        self, mock_read_csv, mock_load_config, mock_prepare,
        mock_features, mock_mlflow, mock_train_lstm, mock_train_rf,
        mock_register,
    ):
        """Pipeline deve retornar dicionário com comparação."""
        import pandas as pd

        from src.models.train import run_training_pipeline

        # Mock dados
        n = 200
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(30, 50, n),
            "High": np.random.uniform(30, 50, n),
            "Low": np.random.uniform(30, 50, n),
            "Close": np.random.uniform(30, 50, n),
            "Volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2024-01-01", periods=n))
        mock_read_csv.return_value = mock_df

        # Mock config
        mock_load_config.return_value = {
            "lstm_optimized": {"sequence_length": 60},
            "random_forest": {"n_estimators": 10},
        }

        # Mock features
        features_df = pd.DataFrame(
            np.random.randn(n, 10),
            columns=[f"f{i}" for i in range(9)] + ["close"],
            index=mock_df.index,
        )
        mock_features.return_value = features_df

        # Mock sequences
        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        # Mock MLflow
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        # Mock training results
        mock_train_lstm.return_value = (
            "lstm-run-001",
            {"mae": 3.5, "rmse": 5.5, "mape": 9.0, "r2": 0.30, "val_loss": 0.01},
            0.5,
        )
        mock_train_rf.return_value = (
            "rf-run-002",
            {"mae": 4.0, "rmse": 6.2, "mape": 11.0, "r2": 0.15},
            0.1,
        )

        # Mock registry
        mock_register.return_value = "1"

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_training_pipeline(
                data_path="fake.csv",
                config_path="configs/model_config.yaml",
                output_dir=tmpdir,
            )

        assert isinstance(result, dict)
        assert "champion" in result
        assert "lstm_run_id" in result
        assert "rf_run_id" in result
        assert result["champion"] == "lstm"

    @patch("src.models.train.register_and_promote")
    @patch("src.models.train.train_random_forest_regressor")
    @patch("src.models.train.train_lstm")
    @patch("src.models.train.mlflow")
    @patch("src.models.train.compute_features")
    @patch("src.models.train.prepare_sequences")
    @patch("src.models.train.load_config")
    @patch("pandas.read_csv")
    def test_pipeline_rf_champion(
        self, mock_read_csv, mock_load_config, mock_prepare,
        mock_features, mock_mlflow, mock_train_lstm, mock_train_rf,
        mock_register,
    ):
        """Pipeline deve selecionar RF como champion quando tem menor RMSE."""
        import pandas as pd

        from src.models.train import run_training_pipeline

        n = 200
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(30, 50, n),
            "High": np.random.uniform(30, 50, n),
            "Low": np.random.uniform(30, 50, n),
            "Close": np.random.uniform(30, 50, n),
            "Volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2024-01-01", periods=n))
        mock_read_csv.return_value = mock_df

        mock_load_config.return_value = {
            "lstm_optimized": {"sequence_length": 60},
            "random_forest": {"n_estimators": 10},
        }

        features_df = pd.DataFrame(
            np.random.randn(n, 10),
            columns=[f"f{i}" for i in range(9)] + ["close"],
            index=mock_df.index,
        )
        mock_features.return_value = features_df

        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        # RF wins
        mock_train_lstm.return_value = (
            "lstm-run-001",
            {"mae": 5.0, "rmse": 7.5, "mape": 15.0, "r2": 0.10, "val_loss": 0.05},
            0.5,
        )
        mock_train_rf.return_value = (
            "rf-run-002",
            {"mae": 3.0, "rmse": 5.0, "mape": 9.0, "r2": 0.35},
            0.1,
        )

        mock_register.return_value = "2"

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_training_pipeline(
                data_path="fake.csv",
                config_path="configs/model_config.yaml",
                output_dir=tmpdir,
            )

        assert result["champion"] == "random_forest"
        assert result["rf_run_id"] == "rf-run-002"

    @patch("src.models.train.register_and_promote")
    @patch("src.models.train.train_random_forest_regressor")
    @patch("src.models.train.train_lstm")
    @patch("src.models.train.mlflow")
    @patch("src.models.train.compute_features")
    @patch("src.models.train.prepare_sequences")
    @patch("src.models.train.load_config")
    @patch("pandas.read_csv")
    def test_pipeline_handles_registry_failure(
        self, mock_read_csv, mock_load_config, mock_prepare,
        mock_features, mock_mlflow, mock_train_lstm, mock_train_rf,
        mock_register,
    ):
        """Pipeline deve continuar mesmo se registry falhar."""
        import pandas as pd

        from src.models.train import run_training_pipeline

        n = 200
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(30, 50, n),
            "High": np.random.uniform(30, 50, n),
            "Low": np.random.uniform(30, 50, n),
            "Close": np.random.uniform(30, 50, n),
            "Volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2024-01-01", periods=n))
        mock_read_csv.return_value = mock_df

        mock_load_config.return_value = {
            "lstm_optimized": {"sequence_length": 60},
            "random_forest": {"n_estimators": 10},
        }

        features_df = pd.DataFrame(
            np.random.randn(n, 10),
            columns=[f"f{i}" for i in range(9)] + ["close"],
            index=mock_df.index,
        )
        mock_features.return_value = features_df

        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        mock_train_lstm.return_value = (
            "lstm-run-001",
            {"mae": 3.5, "rmse": 5.5, "mape": 9.0, "r2": 0.30, "val_loss": 0.01},
            0.5,
        )
        mock_train_rf.return_value = (
            "rf-run-002",
            {"mae": 4.0, "rmse": 6.2, "mape": 11.0, "r2": 0.15},
            0.1,
        )

        # Registry falha
        mock_register.side_effect = Exception("MLflow server offline")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_training_pipeline(
                data_path="fake.csv",
                config_path="configs/model_config.yaml",
                output_dir=tmpdir,
            )

        # Deve continuar e marcar como not_registered
        assert result["registry_version"] == "not_registered"
        assert result["champion"] == "lstm"

    @patch("src.models.train.register_and_promote")
    @patch("src.models.train.train_random_forest_regressor")
    @patch("src.models.train.train_lstm")
    @patch("src.models.train.mlflow")
    @patch("src.models.train.compute_features")
    @patch("src.models.train.prepare_sequences")
    @patch("src.models.train.load_config")
    @patch("pandas.read_csv")
    def test_pipeline_saves_metrics_json(
        self, mock_read_csv, mock_load_config, mock_prepare,
        mock_features, mock_mlflow, mock_train_lstm, mock_train_rf,
        mock_register,
    ):
        """Pipeline deve salvar métricas em JSON no output_dir."""
        import json
        import pandas as pd
        from pathlib import Path

        from src.models.train import run_training_pipeline

        n = 200
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(30, 50, n),
            "High": np.random.uniform(30, 50, n),
            "Low": np.random.uniform(30, 50, n),
            "Close": np.random.uniform(30, 50, n),
            "Volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2024-01-01", periods=n))
        mock_read_csv.return_value = mock_df

        mock_load_config.return_value = {
            "lstm_optimized": {"sequence_length": 60},
            "random_forest": {"n_estimators": 10},
        }

        features_df = pd.DataFrame(
            np.random.randn(n, 10),
            columns=[f"f{i}" for i in range(9)] + ["close"],
            index=mock_df.index,
        )
        mock_features.return_value = features_df

        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        mock_train_lstm.return_value = (
            "lstm-run-001",
            {"mae": 3.5, "rmse": 5.5, "mape": 9.0, "r2": 0.30, "val_loss": 0.01},
            0.5,
        )
        mock_train_rf.return_value = (
            "rf-run-002",
            {"mae": 4.0, "rmse": 6.2, "mape": 11.0, "r2": 0.15},
            0.1,
        )

        mock_register.return_value = "1"

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            run_training_pipeline(
                data_path="fake.csv",
                config_path="configs/model_config.yaml",
                output_dir=tmpdir,
            )

            metrics_file = Path(tmpdir) / "train_metrics.json"
            assert metrics_file.exists()

            with open(metrics_file) as f:
                saved = json.load(f)
            assert "champion" in saved
            assert "lstm" in saved
            assert "random_forest" in saved
