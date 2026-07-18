"""Otimização de hiperparâmetros com Optuna + MLflow.

Corrige gap identificado na Fase 4: hiperparâmetros definidos manualmente.
Agora usa busca bayesiana estruturada via Optuna com tracking no MLflow.

Referência: Akiba et al. (2019) — Optuna: A Next-generation Hyperparameter
            Optimization Framework. https://arxiv.org/abs/1907.10902
"""

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

from src.features.feature_engineering import compute_features, prepare_sequences

load_dotenv()
logger = logging.getLogger(__name__)


# === MODELO LSTM (evolução da Fase 4) ===


class LSTMPredictor(nn.Module):
    """LSTM para previsão de preços de ações.

    Evolução da Fase 4: agora com arquitetura parametrizável
    para otimização via Optuna.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Inicializa LSTM.

        Args:
            input_size: Número de features de entrada.
            hidden_size_1: Neurônios na primeira camada LSTM.
            hidden_size_2: Neurônios na camada densa.
            num_layers: Número de camadas LSTM empilhadas.
            dropout: Taxa de dropout entre camadas.

        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (batch, seq_len, features).

        Returns:
            Predição de preço (batch, 1).

        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# === FUNÇÕES DE TREINO ===


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    epochs: int = 50,
    patience: int = 10,
) -> tuple[LSTMPredictor, float]:
    """Treina modelo LSTM com early stopping.

    Args:
        X_train: Features de treino (n, seq_len, n_features).
        y_train: Target de treino.
        X_val: Features de validação.
        y_val: Target de validação.
        params: Hiperparâmetros do modelo.
        epochs: Máximo de épocas.
        patience: Épocas sem melhora para parar.

    Returns:
        Tupla (modelo treinado, melhor val_loss).

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMPredictor(
        input_size=X_train.shape[2],
        hidden_size_1=params["hidden_size_1"],
        hidden_size_2=params["hidden_size_2"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Tensores
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    batch_size = params["batch_size"]
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Treino
        model.train()
        train_losses = []
        for i in range(0, len(X_train_t), batch_size):
            batch_x = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validação
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.debug("Early stopping na época %d", epoch + 1)
                break

    return model, best_val_loss


# === OPTUNA OBJECTIVE ===


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """Cria função objetivo para Optuna.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        X_val: Features de validação.
        y_val: Target de validação.

    Returns:
        Função objetivo que recebe um trial Optuna.

    """
    def objective(trial: optuna.Trial) -> float:
        """Função objetivo — minimiza val_loss."""
        params = {
            "hidden_size_1": trial.suggest_int("hidden_size_1", 32, 256, step=32),
            "hidden_size_2": trial.suggest_int("hidden_size_2", 16, 128, step=16),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(params)
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("optimization", "optuna")

            _, val_loss = train_lstm_model(
                X_train, y_train, X_val, y_val,
                params=params,
                epochs=30,
                patience=7,
            )

            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_rmse", np.sqrt(val_loss))

        return val_loss

    return objective


# === PIPELINE PRINCIPAL ===


def run_hyperparameter_search(
    n_trials: int = 30,
    sequence_length: int = 60,
    data_path: str = "data/raw/PETR4_SA_historico.csv",
    study_name: str = "lstm_petr4_optimization",
) -> dict:
    """Executa busca de hiperparâmetros com Optuna + MLflow.

    Args:
        n_trials: Número de trials (combinações a testar).
        sequence_length: Tamanho da sequência de entrada para o LSTM.
        data_path: Caminho para os dados.
        study_name: Nome do estudo Optuna.

    Returns:
        Dicionário com melhores hiperparâmetros encontrados.

    """
    import pandas as pd

    logger.info("Iniciando busca de hiperparâmetros: %d trials", n_trials)

    # Carregar e preparar dados
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"].astype(int)

    features = compute_features(df, validate=False)

    # Normalizar features
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    # Preparar sequências
    X, y = prepare_sequences(features_scaled, target_col="close", sequence_length=sequence_length)

    # Split temporal (80/20 — sem shuffle para séries temporais)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info("Dados: X_train=%s, X_val=%s", X_train.shape, X_val.shape)

    # Configurar MLflow
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-fase05-tuning")
    mlflow.set_experiment(experiment_name)

    # Criar estudo Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # Executar otimização dentro de um run MLflow pai
    with mlflow.start_run(run_name="hyperparameter_search"):
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("optimization", "optuna")
        mlflow.set_tag("n_trials", n_trials)
        mlflow.set_tag("sequence_length", sequence_length)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))

        # Suprimir logs do Optuna durante a busca
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective = create_objective(X_train, y_train, X_val, y_val)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Melhores parâmetros
        best_params = study.best_params
        best_params["sequence_length"] = sequence_length

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_loss", study.best_value)
        mlflow.log_metric("best_val_rmse", np.sqrt(study.best_value))

    logger.info("Melhor val_loss: %.6f (RMSE: %.6f)", study.best_value, np.sqrt(study.best_value))
    logger.info("Melhores parâmetros: %s", best_params)

    return best_params


def save_best_params(params: dict, output_path: str = "configs/model_config.yaml") -> None:
    """Salva melhores hiperparâmetros no arquivo de configuração.

    Args:
        params: Dicionário com hiperparâmetros otimizados.
        output_path: Caminho para o arquivo YAML.

    """
    config_path = Path(output_path)
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    # Adicionar/atualizar seção LSTM otimizada
    config["lstm_optimized"] = {
        "hidden_size_1": params["hidden_size_1"],
        "hidden_size_2": params["hidden_size_2"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "sequence_length": params.get("sequence_length", 60),
        "optimization": "optuna_bayesian",
        "note": "Hiperparâmetros otimizados via Optuna (busca bayesiana)",
    }

    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    logger.info("Melhores parâmetros salvos em %s", output_path)


# === ENTRY POINT ===


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("  OTIMIZAÇÃO DE HIPERPARÂMETROS — OPTUNA + MLFLOW")
    print("  Corrige gap da Fase 4: hiperparâmetros manuais")
    print("=" * 60)

    # Executar busca (30 trials por padrão)
    best_params = run_hyperparameter_search(n_trials=30)

    # Salvar no config
    save_best_params(best_params)

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("\n  Salvo em: configs/model_config.yaml")
