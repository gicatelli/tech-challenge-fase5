"""Pipeline de treinamento com MLflow tracking padronizado.

Implementa o padrão de logging obrigatório para o Datathon:
- Params: hiperparâmetros + metadata
- Metrics: AUC, F1, Precision, Recall
- Tags: model_type, framework, owner, phase
- Artifacts: modelo serializado
"""

import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)


def train_and_log(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    model_class,
    model_params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Treina modelo, loga tudo no MLflow, retorna run_id.

    Args:
        df: DataFrame com features e target.
        target_col: Nome da coluna target.
        model_name: Nome para registro no MLflow.
        model_class: Classe do modelo (ex: RandomForestClassifier).
        model_params: Hiperparâmetros do modelo.
        test_size: Proporção de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        run_id do experimento MLflow.

    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-fase05"))

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run(run_name=model_name) as run:
        # Log de parâmetros
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])

        # Tags padronizadas (obrigatório)
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", model_class.__module__.split(".")[0])
        mlflow.set_tag("owner", "grupo-XX")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("risk_level", "medium")

        # Treino
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Métricas padronizadas
        metrics = {
            "auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        # Log do modelo
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "Modelo %s treinado: AUC=%.4f, F1=%.4f",
            model_name,
            metrics["auc"],
            metrics["f1"],
        )

        return run.info.run_id


if __name__ == "__main__":
    # Exemplo de uso com dados sintéticos
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame(
        {
            "feature_1": np.random.uniform(0, 1, n_samples),
            "feature_2": np.random.uniform(1, 10, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )

    run_id = train_and_log(
        df=df,
        target_col="target",
        model_name="random-forest-baseline",
        model_class=RandomForestClassifier,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
    )
    print(f"Run ID: {run_id}")
