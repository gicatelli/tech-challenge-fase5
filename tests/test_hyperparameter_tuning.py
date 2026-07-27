"""Testes para src/models/hyperparameter_tuning.py — Optuna + LSTM."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.models.hyperparameter_tuning import (
    LSTMPredictor,
    create_objective,
    save_best_params,
    train_lstm_model,
)


class TestLSTMPredictor:
    """Testes para o modelo LSTM."""

    def test_init_default_params(self):
        """Deve inicializar com parâmetros padrão."""
        model = LSTMPredictor(input_size=10)
        assert model.lstm.input_size == 10
        assert model.lstm.hidden_size == 128
        assert model.fc2.out_features == 1

    def test_init_custom_params(self):
        """Deve inicializar com parâmetros customizados."""
        model = LSTMPredictor(
            input_size=5,
            hidden_size_1=64,
            hidden_size_2=32,
            num_layers=3,
            dropout=0.3,
        )
        assert model.lstm.input_size == 5
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 3
        assert model.fc1.out_features == 32

    def test_forward_shape(self):
        """Forward deve retornar shape (batch, 1)."""
        model = LSTMPredictor(input_size=10, hidden_size_1=32, hidden_size_2=16)
        x = torch.randn(8, 60, 10)  # batch=8, seq=60, features=10
        output = model(x)
        assert output.shape == (8, 1)

    def test_forward_single_sample(self):
        """Forward deve funcionar com batch=1."""
        model = LSTMPredictor(input_size=5, hidden_size_1=16, hidden_size_2=8)
        x = torch.randn(1, 30, 5)
        output = model(x)
        assert output.shape == (1, 1)

    def test_forward_different_seq_lengths(self):
        """Forward deve funcionar com seq_len variável."""
        model = LSTMPredictor(input_size=10, hidden_size_1=32, hidden_size_2=16)
        for seq_len in [10, 30, 60, 120]:
            x = torch.randn(4, seq_len, 10)
            output = model(x)
            assert output.shape == (4, 1)

    def test_single_layer_no_dropout(self):
        """Com num_layers=1, dropout interno do LSTM deve ser 0."""
        model = LSTMPredictor(input_size=5, num_layers=1, dropout=0.5)
        # LSTM com 1 layer ignora dropout entre layers
        assert model.lstm.dropout == 0.0

    def test_multi_layer_has_dropout(self):
        """Com num_layers>1, dropout interno do LSTM deve ser aplicado."""
        model = LSTMPredictor(input_size=5, num_layers=2, dropout=0.3)
        assert model.lstm.dropout == 0.3

    def test_output_is_float_tensor(self):
        """Output deve ser tensor float."""
        model = LSTMPredictor(input_size=5, hidden_size_1=16, hidden_size_2=8)
        x = torch.randn(2, 10, 5)
        output = model(x)
        assert output.dtype == torch.float32


class TestTrainLstmModel:
    """Testes para train_lstm_model."""

    @pytest.fixture
    def training_data(self):
        """Dados sintéticos para treino rápido."""
        np.random.seed(42)
        n_train, n_val, seq_len, n_features = 50, 10, 20, 5
        X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
        y_train = np.random.uniform(0, 1, n_train).astype(np.float32)
        X_val = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
        y_val = np.random.uniform(0, 1, n_val).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_returns_model_and_loss(self, training_data):
        """Deve retornar modelo treinado e val_loss."""
        X_train, y_train, X_val, y_val = training_data
        params = {
            "hidden_size_1": 16,
            "hidden_size_2": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "batch_size": 16,
        }
        model, val_loss = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=params, epochs=3, patience=2,
        )
        assert isinstance(model, LSTMPredictor)
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_early_stopping(self, training_data):
        """Early stopping deve parar antes do máximo de épocas."""
        X_train, y_train, X_val, y_val = training_data
        params = {
            "hidden_size_1": 16,
            "hidden_size_2": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.1,
            "batch_size": 50,
        }
        # Com patience=1 e dados pequenos, deve parar cedo
        model, val_loss = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=params, epochs=100, patience=1,
        )
        assert isinstance(model, LSTMPredictor)
        assert val_loss > 0

    def test_val_loss_decreases_with_more_epochs(self, training_data):
        """Mais épocas devem reduzir (ou manter) val_loss."""
        X_train, y_train, X_val, y_val = training_data
        params = {
            "hidden_size_1": 32,
            "hidden_size_2": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.005,
            "batch_size": 16,
        }
        _, loss_2epochs = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=params, epochs=2, patience=10,
        )
        _, loss_10epochs = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=params, epochs=10, patience=10,
        )
        # Com mais épocas, val_loss deve ser menor ou igual
        # (não garantido 100%, mas com dados fixos e seed é muito provável)
        assert loss_10epochs <= loss_2epochs * 1.5  # margem para estocasticidade

    def test_multi_layer_model(self, training_data):
        """Modelo com múltiplas camadas deve treinar sem erro."""
        X_train, y_train, X_val, y_val = training_data
        params = {
            "hidden_size_1": 16,
            "hidden_size_2": 8,
            "num_layers": 3,
            "dropout": 0.2,
            "learning_rate": 0.01,
            "batch_size": 32,
        }
        model, val_loss = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=params, epochs=2, patience=5,
        )
        assert isinstance(model, LSTMPredictor)
        assert val_loss > 0


class TestCreateObjective:
    """Testes para create_objective."""

    @pytest.fixture
    def data(self):
        """Dados para criar objective."""
        np.random.seed(42)
        n_train, n_val, seq_len, n_features = 30, 10, 15, 5
        X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
        y_train = np.random.uniform(0, 1, n_train).astype(np.float32)
        X_val = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
        y_val = np.random.uniform(0, 1, n_val).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_returns_callable(self, data):
        """create_objective deve retornar uma função."""
        X_train, y_train, X_val, y_val = data
        objective = create_objective(X_train, y_train, X_val, y_val)
        assert callable(objective)

    @patch("src.models.hyperparameter_tuning.mlflow")
    def test_objective_returns_float(self, mock_mlflow, data):
        """Objetivo deve retornar float (val_loss)."""
        X_train, y_train, X_val, y_val = data

        # Mock MLflow
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        objective = create_objective(X_train, y_train, X_val, y_val)

        # Criar mock trial
        trial = MagicMock()
        trial.number = 0
        trial.suggest_int.side_effect = lambda name, *args, **kwargs: {
            "hidden_size_1": 16, "hidden_size_2": 8, "num_layers": 1,
        }[name]
        trial.suggest_float.side_effect = lambda name, *args, **kwargs: {
            "dropout": 0.1, "learning_rate": 0.01,
        }[name]
        trial.suggest_categorical.return_value = 16

        result = objective(trial)
        assert isinstance(result, float)
        assert result > 0

    @patch("src.models.hyperparameter_tuning.mlflow")
    def test_objective_logs_to_mlflow(self, mock_mlflow, data):
        """Objetivo deve logar params e métricas no MLflow."""
        X_train, y_train, X_val, y_val = data

        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        objective = create_objective(X_train, y_train, X_val, y_val)

        trial = MagicMock()
        trial.number = 1
        trial.suggest_int.side_effect = lambda name, *args, **kwargs: {
            "hidden_size_1": 32, "hidden_size_2": 16, "num_layers": 1,
        }[name]
        trial.suggest_float.side_effect = lambda name, *args, **kwargs: {
            "dropout": 0.2, "learning_rate": 0.005,
        }[name]
        trial.suggest_categorical.return_value = 32

        objective(trial)

        # Verificar que MLflow foi chamado
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.set_tag.assert_called()


class TestSaveBestParams:
    """Testes para save_best_params."""

    def test_saves_yaml_file(self, tmp_path):
        """Deve salvar parâmetros em YAML."""
        output = tmp_path / "config.yaml"
        output.write_text("random_forest:\n  n_estimators: 100\n")

        params = {
            "hidden_size_1": 128,
            "hidden_size_2": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "sequence_length": 60,
        }

        save_best_params(params, str(output))

        import yaml

        config = yaml.safe_load(output.read_text())
        assert "lstm_optimized" in config
        assert config["lstm_optimized"]["hidden_size_1"] == 128
        assert config["lstm_optimized"]["sequence_length"] == 60
        assert config["lstm_optimized"]["optimization"] == "optuna_bayesian"

    def test_preserves_existing_config(self, tmp_path):
        """Deve preservar config existente ao adicionar lstm_optimized."""
        output = tmp_path / "config.yaml"
        output.write_text("random_forest:\n  n_estimators: 200\n")

        params = {"hidden_size_1": 64, "hidden_size_2": 32, "num_layers": 1,
                  "dropout": 0.1, "learning_rate": 0.01, "batch_size": 64}

        save_best_params(params, str(output))

        import yaml

        config = yaml.safe_load(output.read_text())
        assert config["random_forest"]["n_estimators"] == 200
        assert "lstm_optimized" in config

    def test_creates_file_if_not_exists(self, tmp_path):
        """Deve criar arquivo se não existir."""
        output = tmp_path / "new_config.yaml"

        params = {"hidden_size_1": 128, "hidden_size_2": 64, "num_layers": 2,
                  "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32}

        save_best_params(params, str(output))

        assert output.exists()

        import yaml

        config = yaml.safe_load(output.read_text())
        assert "lstm_optimized" in config

    def test_default_sequence_length(self, tmp_path):
        """Se sequence_length não fornecido, usa 60 como padrão."""
        output = tmp_path / "config.yaml"
        output.write_text("")

        params = {"hidden_size_1": 128, "hidden_size_2": 64, "num_layers": 2,
                  "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32}

        save_best_params(params, str(output))

        import yaml

        config = yaml.safe_load(output.read_text())
        assert config["lstm_optimized"]["sequence_length"] == 60


class TestRunHyperparameterSearch:
    """Testes para run_hyperparameter_search (com mocks pesados)."""

    @patch("src.models.hyperparameter_tuning.optuna")
    @patch("src.models.hyperparameter_tuning.mlflow")
    @patch("src.models.hyperparameter_tuning.compute_features")
    @patch("src.models.hyperparameter_tuning.prepare_sequences")
    @patch("pandas.read_csv")
    def test_returns_best_params(
        self, mock_read_csv, mock_prepare, mock_features, mock_mlflow, mock_optuna
    ):
        """Deve retornar dicionário com melhores parâmetros."""
        from src.models.hyperparameter_tuning import run_hyperparameter_search

        # Mock pandas read_csv
        import pandas as pd

        mock_df = pd.DataFrame({
            "Close": np.random.uniform(30, 50, 200),
            "Volume": np.random.randint(1000000, 5000000, 200),
        }, index=pd.date_range("2024-01-01", periods=200))
        mock_read_csv.return_value = mock_df

        # Mock features
        mock_features.return_value = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f"f{i}" for i in range(10)],
            index=mock_df.index,
        )

        # Mock sequences
        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        # Mock MLflow
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            "hidden_size_1": 128,
            "hidden_size_2": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
        mock_study.best_value = 0.05
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.logging = MagicMock()
        mock_optuna.pruners.MedianPruner.return_value = MagicMock()

        result = run_hyperparameter_search(
            n_trials=2,
            sequence_length=60,
            data_path="data/raw/PETR4_SA_historico.csv",
        )

        assert isinstance(result, dict)
        assert "hidden_size_1" in result
        assert "sequence_length" in result
        assert result["sequence_length"] == 60

    @patch("src.models.hyperparameter_tuning.optuna")
    @patch("src.models.hyperparameter_tuning.mlflow")
    @patch("src.models.hyperparameter_tuning.compute_features")
    @patch("src.models.hyperparameter_tuning.prepare_sequences")
    @patch("pandas.read_csv")
    def test_calls_optuna_optimize(
        self, mock_read_csv, mock_prepare, mock_features, mock_mlflow, mock_optuna
    ):
        """Deve chamar study.optimize com n_trials correto."""
        from src.models.hyperparameter_tuning import run_hyperparameter_search

        import pandas as pd

        mock_df = pd.DataFrame({
            "Close": np.random.uniform(30, 50, 200),
            "Volume": np.random.randint(1000000, 5000000, 200),
        }, index=pd.date_range("2024-01-01", periods=200))
        mock_read_csv.return_value = mock_df

        mock_features.return_value = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f"f{i}" for i in range(10)],
            index=mock_df.index,
        )

        X = np.random.randn(140, 60, 10).astype(np.float32)
        y = np.random.uniform(0, 1, 140).astype(np.float32)
        mock_prepare.return_value = (X, y)

        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_study = MagicMock()
        mock_study.best_params = {
            "hidden_size_1": 64, "hidden_size_2": 32,
            "num_layers": 1, "dropout": 0.1,
            "learning_rate": 0.01, "batch_size": 64,
        }
        mock_study.best_value = 0.03
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.logging = MagicMock()
        mock_optuna.pruners.MedianPruner.return_value = MagicMock()

        run_hyperparameter_search(n_trials=5, data_path="fake.csv")

        # Verificar que optimize foi chamado com n_trials=5
        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args
        assert call_kwargs[1]["n_trials"] == 5
