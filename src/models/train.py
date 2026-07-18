"""Pipeline de treinamento com MLflow tracking padronizado.

Treina dois modelos e compara performance:
1. LSTM (PyTorch) — modelo principal para séries temporais
2. Random Forest (Scikit-Learn) — baseline interpretável

Ambos são logados no MLflow com params, metrics, tags e artifacts.
O champion é selecionado automaticamente por menor RMSE.
"""

import json
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler

from src.features.feature_engineering import compute_features, prepare_sequences
from src.models.hyperparameter_tuning import train_lstm_model
from src.models.registry import register_and_promote

load_dotenv()
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega configuração de hiperparâmetros.

    Args:
        config_path: Caminho para o arquivo YAML.

    Returns:
        Dicionário com configurações.

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Computa métricas de regressão padronizadas.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.

    Returns:
        Dicionário com MAE, RMSE, MAPE, R².

    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> tuple[str, dict[str, float], float]:
    """Treina LSTM e loga no MLflow.

    Args:
        X_train: Sequências de treino (n, seq_len, features).
        y_train: Target de treino.
        X_test: Sequências de teste.
        y_test: Target de teste.
        config: Configurações do modelo.

    Returns:
        Tupla (run_id, metrics, latency_ms).

    """
    params = config.get("lstm_optimized", {})
    model_params = {
        "hidden_size_1": params.get("hidden_size_1", 128),
        "hidden_size_2": params.get("hidden_size_2", 64),
        "num_layers": params.get("num_layers", 2),
        "dropout": params.get("dropout", 0.2),
        "learning_rate": params.get("learning_rate", 0.001),
        "batch_size": params.get("batch_size", 32),
    }

    with mlflow.start_run(run_name="lstm-petr4") as run:
        mlflow.log_params(model_params)
        mlflow.log_param("sequence_length", params.get("sequence_length", 60))
        mlflow.log_param("epochs", 50)
        mlflow.log_param("n_features", X_train.shape[2])
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))

        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("architecture", "LSTM")
        mlflow.set_tag("owner", "giovanna-catelli")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("risk_level", "medium")
        mlflow.set_tag("optimization", params.get("optimization", "manual"))
        mlflow.set_tag("asset", "PETR4.SA")

        # Treinar
        model, val_loss = train_lstm_model(
            X_train, y_train, X_test, y_test,
            params=model_params,
            epochs=50,
            patience=10,
        )

        # Inferência no teste
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_test_t).cpu().numpy().flatten()
        latency_ms = (time.time() - start_time) * 1000 / len(X_test)

        # Métricas
        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["val_loss"] = val_loss
        mlflow.log_metrics(metrics)
        mlflow.log_metric("inference_latency_ms", latency_ms)

        # Salvar modelo PyTorch
        mlflow.pytorch.log_model(model, "model")

        logger.info(
            "LSTM treinado: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, R²=%.4f",
            metrics["mae"], metrics["rmse"], metrics["mape"], metrics["r2"],
        )

        return run.info.run_id, metrics, latency_ms  # type: ignore[return-value]


def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> tuple[str, dict[str, float], float]:
    """Treina Random Forest Regressor e loga no MLflow.

    Args:
        X_train: Features de treino (2D — último timestep da sequência).
        y_train: Target de treino.
        X_test: Features de teste.
        y_test: Target de teste.
        config: Configurações do modelo.

    Returns:
        Tupla (run_id, metrics, latency_ms).

    """
    rf_config = config.get("random_forest", {})
    model_params = {
        "n_estimators": rf_config.get("n_estimators", 100),
        "max_depth": rf_config.get("max_depth", 10),
        "min_samples_split": rf_config.get("min_samples_split", 5),
        "min_samples_leaf": rf_config.get("min_samples_leaf", 2),
        "random_state": rf_config.get("random_state", 42),
        "n_jobs": rf_config.get("n_jobs", -1),
    }

    with mlflow.start_run(run_name="random-forest-petr4") as run:
        mlflow.log_params(model_params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))

        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("architecture", "RandomForest")
        mlflow.set_tag("owner", "giovanna-catelli")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("risk_level", "low")
        mlflow.set_tag("asset", "PETR4.SA")

        # Treinar
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        # Inferência
        start_time = time.time()
        y_pred = model.predict(X_test)
        latency_ms = (time.time() - start_time) * 1000 / len(X_test)

        # Métricas
        metrics = compute_regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("inference_latency_ms", latency_ms)

        # Salvar modelo
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "RandomForest treinado: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, R²=%.4f",
            metrics["mae"], metrics["rmse"], metrics["mape"], metrics["r2"],
        )

        return run.info.run_id, metrics, latency_ms  # type: ignore[return-value]


def select_champion(
    lstm_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    lstm_latency: float,
    rf_latency: float,
) -> dict:
    """Seleciona o champion model baseado em RMSE.

    Args:
        lstm_metrics: Métricas do LSTM.
        rf_metrics: Métricas do Random Forest.
        lstm_latency: Latência do LSTM (ms/sample).
        rf_latency: Latência do RF (ms/sample).

    Returns:
        Dicionário com resultado da comparação e champion selecionado.

    """
    comparison = {
        "lstm": {**lstm_metrics, "latency_ms": lstm_latency},
        "random_forest": {**rf_metrics, "latency_ms": rf_latency},
        "champion": "lstm" if lstm_metrics["rmse"] < rf_metrics["rmse"] else "random_forest",
        "reason": "",
    }

    if lstm_metrics["rmse"] < rf_metrics["rmse"]:
        improvement = (rf_metrics["rmse"] - lstm_metrics["rmse"]) / rf_metrics["rmse"] * 100
        comparison["reason"] = (
            f"LSTM selecionado como champion: RMSE {improvement:.1f}% menor que RF. "
            f"LSTM captura padrões temporais que RF não consegue."
        )
    else:
        comparison["reason"] = (
            "Random Forest selecionado como champion: RMSE igual ou menor que LSTM "
            "com latência significativamente menor. Para produção, "
            "simplicidade e interpretabilidade são priorizadas."
        )

    return comparison


def run_training_pipeline(
    data_path: str = "data/raw/PETR4_SA_historico.csv",
    config_path: str = "configs/model_config.yaml",
    output_dir: str = "metrics",
) -> dict:
    """Executa pipeline completo de treinamento.

    1. Carrega dados e computa features
    2. Treina LSTM com params otimizados
    3. Treina Random Forest como baseline
    4. Compara e seleciona champion
    5. Salva métricas e resultados

    Args:
        data_path: Caminho para os dados brutos.
        config_path: Caminho para configuração YAML.
        output_dir: Diretório para salvar métricas.

    Returns:
        Dicionário com resultados da comparação.

    """
    logger.info("Iniciando pipeline de treinamento")

    # Carregar config
    config = load_config(config_path)
    seq_length = config.get("lstm_optimized", {}).get("sequence_length", 60)

    # Carregar e preparar dados
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"].astype(int)
    logger.info("Dados carregados: %s", df.shape)

    # Feature engineering
    features = compute_features(df, validate=False)
    logger.info("Features computadas: %s", features.shape)

    # Normalizar
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    # Preparar sequências para LSTM
    X_seq, y_seq = prepare_sequences(
        features_scaled, target_col="close", sequence_length=seq_length
    )

    # Split temporal (80/20)
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq = X_seq[:split_idx]
    X_test_seq = X_seq[split_idx:]
    y_train_seq = y_seq[:split_idx]
    y_test_seq = y_seq[split_idx:]

    # Para RF: usar último timestep de cada sequência (flatten)
    X_train_rf = X_train_seq[:, -1, :]  # (n, features)
    X_test_rf = X_test_seq[:, -1, :]
    y_train_rf = y_train_seq
    y_test_rf = y_test_seq

    logger.info(
        "Split: train=%d, test=%d (seq_len=%d)",
        len(X_train_seq), len(X_test_seq), seq_length,
    )

    # Configurar MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-fase05"))

    # === TREINAR LSTM ===
    logger.info("Treinando LSTM...")
    lstm_run_id, lstm_metrics, lstm_latency = train_lstm(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, config
    )

    # === TREINAR RANDOM FOREST ===
    logger.info("Treinando Random Forest...")
    rf_run_id, rf_metrics, rf_latency = train_random_forest_regressor(
        X_train_rf, y_train_rf, X_test_rf, y_test_rf, config
    )

    # === COMPARAR E SELECIONAR CHAMPION ===
    comparison = select_champion(lstm_metrics, rf_metrics, lstm_latency, rf_latency)
    comparison["lstm_run_id"] = lstm_run_id
    comparison["rf_run_id"] = rf_run_id

    # Salvar métricas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "train_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Métricas salvas em %s", metrics_file)

    # Registrar champion no Model Registry
    champion_run_id = (
        lstm_run_id if comparison["champion"] == "lstm" else rf_run_id
    )
    try:
        version = register_and_promote(
            run_id=champion_run_id,
            model_name="lstm-petr4-predictor" if comparison["champion"] == "lstm"
            else "rf-petr4-predictor",
            model_version="1.0.0",
        )
        comparison["registry_version"] = version
        logger.info("Champion registrado no Model Registry: v%s", version)
    except Exception as e:
        logger.warning("Registro no Model Registry falhou (MLflow offline?): %s", e)
        comparison["registry_version"] = "not_registered"

    # Log resumo
    print("\n" + "=" * 70)
    print("  COMPARAÇÃO DE MODELOS — DATATHON FASE 05")
    print("=" * 70)
    print(f"\n{'Métrica':<20} {'LSTM':<15} {'Random Forest':<15} {'Melhor':<10}")
    print("-" * 60)
    for metric in ["mae", "rmse", "mape", "r2"]:
        lstm_val = lstm_metrics[metric]
        rf_val = rf_metrics[metric]
        better = "LSTM" if (
            lstm_val < rf_val if metric != "r2" else lstm_val > rf_val
        ) else "RF"
        print(f"  {metric.upper():<18} {lstm_val:<15.4f} {rf_val:<15.4f} {better:<10}")

    print(f"\n  {'Latência (ms)':<18} {lstm_latency:<15.2f} {rf_latency:<15.2f}")
    print(f"\n  Champion: {comparison['champion'].upper()}")
    print(f"  Razão: {comparison['reason']}")

    return comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 70)
    print("  PIPELINE DE TREINAMENTO — DATATHON FASE 05")
    print("  PETR4.SA | LSTM vs Random Forest")
    print("=" * 70)

    results = run_training_pipeline()
