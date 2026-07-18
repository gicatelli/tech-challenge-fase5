"""MLflow Model Registry — Registro e promoção de modelos.

Implementa o workflow de governança de modelos:
1. Registrar modelo com metadata obrigatória
2. Adicionar tags padronizadas (rastreabilidade)
3. Promover para Production após validação

Referência: Microsoft MLOps Maturity Model — Nível 2 (Model Management)
"""

import hashlib
import logging
import os
import subprocess
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
logger = logging.getLogger(__name__)

# Tags obrigatórias para registro (governança)
REQUIRED_TAGS = [
    "model_name",
    "model_version",
    "model_type",
    "training_data_version",
    "owner",
    "risk_level",
]


def get_git_sha() -> str:
    """Obtém o SHA do commit atual.

    Returns:
        SHA curto do commit ou 'unknown'.

    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_version(data_path: str = "data/raw/PETR4_SA_historico.csv") -> str:
    """Calcula hash MD5 dos dados para versionamento.

    Args:
        data_path: Caminho para o arquivo de dados.

    Returns:
        Hash MD5 curto (8 chars) ou 'unknown'.

    """
    try:
        with open(data_path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()[:8]  # noqa: S324
    except FileNotFoundError:
        return "unknown"


def register_champion_model(
    run_id: str,
    model_name: str = "lstm-petr4-predictor",
    model_version: str = "1.0.0",
    model_type: str = "regression",
    risk_level: str = "medium",
    optimization: str = "optuna",
    artifact_path: str = "model",
) -> str:
    """Registra modelo no MLflow Model Registry com tags obrigatórias.

    Args:
        run_id: ID do run MLflow que contém o modelo.
        model_name: Nome do modelo no registry.
        model_version: Versão semântica.
        model_type: Tipo do modelo (regression, classification).
        risk_level: Nível de risco (low, medium, high, critical).
        optimization: Método de otimização usado.
        artifact_path: Caminho do artifact no run.

    Returns:
        Versão do modelo registrado no registry.

    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info("Registrando modelo '%s' do run %s", model_name, run_id)

    # Registrar modelo
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = model_details.version

    # Tags obrigatórias
    tags = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": model_type,
        "training_data_version": get_data_version(),
        "owner": "giovanna-catelli",
        "risk_level": risk_level,
        "optimization": optimization,
        "git_sha": get_git_sha(),
        "registered_at": datetime.now().isoformat(),
        "fairness_checked": "false",
        "phase": "datathon-fase05",
        "asset": "PETR4.SA",
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=str(value),
        )

    logger.info(
        "Modelo registrado: %s v%s (registry version: %s)",
        model_name, model_version, version,
    )

    return version


def promote_to_production(
    model_name: str = "lstm-petr4-predictor",
    version: str = "1",
    description: str = "",
) -> None:
    """Promove modelo para stage Production.

    Args:
        model_name: Nome do modelo no registry.
        version: Versão a promover.
        description: Descrição da promoção.

    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    # Transicionar para Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Adicionar descrição
    if description:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description,
        )

    # Tag de promoção
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_to_production_at",
        value=datetime.now().isoformat(),
    )

    logger.info("Modelo %s v%s promovido para Production", model_name, version)


def get_production_model(
    model_name: str = "lstm-petr4-predictor",
):
    """Carrega modelo em Production do registry.

    Args:
        model_name: Nome do modelo no registry.

    Returns:
        Modelo carregado pronto para inferência.

    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pytorch.load_model(model_uri)

    logger.info("Modelo carregado do registry: %s (Production)", model_name)
    return model


def register_and_promote(
    run_id: str,
    model_name: str = "lstm-petr4-predictor",
    model_version: str = "1.0.0",
) -> str:
    """Registra e promove modelo em um único passo.

    Convenience function que combina registro + promoção.

    Args:
        run_id: ID do run MLflow.
        model_name: Nome do modelo.
        model_version: Versão semântica.

    Returns:
        Versão do modelo no registry.

    """
    version = register_champion_model(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
    )

    promote_to_production(
        model_name=model_name,
        version=version,
        description=(
            f"LSTM otimizado via Optuna para previsão de preços PETR4.SA. "
            f"Versão {model_version}. Git SHA: {get_git_sha()}."
        ),
    )

    return version


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("  MODEL REGISTRY — DATATHON FASE 05")
    print("=" * 60)
    print("\nUso:")
    print("  1. Treinar modelo: python -m src.models.train")
    print("  2. Obter run_id do champion em metrics/train_metrics.json")
    print("  3. Registrar:")
    print("     python -m src.models.registry --run-id <RUN_ID>")
    print("\nOu programaticamente:")
    print("  from src.models.registry import register_and_promote")
    print("  register_and_promote(run_id='abc123')")

    # Se run_id passado via argumento
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--run-id" and len(sys.argv) > 2:
        run_id = sys.argv[2]
        version = register_and_promote(run_id=run_id)
        print(f"\n✅ Modelo registrado e promovido: version={version}")
    else:
        print("\nPara registrar, passe: --run-id <RUN_ID>")
