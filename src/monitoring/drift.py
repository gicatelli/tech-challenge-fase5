"""Drift detection com Evidently.

Implementa detecção de data drift e concept drift:
- PSI (Population Stability Index) como métrica principal
- Threshold: PSI > 0.1 = warning, PSI > 0.2 = retrain trigger
- Integração com MLflow para logging de métricas de drift
"""

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.monitoring.metrics import track_drift

logger = logging.getLogger(__name__)

# Thresholds de drift
PSI_WARNING_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.2


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula Population Stability Index (PSI).

    Args:
        reference: Distribuição de referência (treino).
        current: Distribuição atual (produção).
        n_bins: Número de bins para discretização.

    Returns:
        Score PSI.

    """
    # Discretizar em bins
    bins = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    # Evitar divisão por zero
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Detecta drift usando Evidently.

    Args:
        reference_data: Dados de referência (treino).
        current_data: Dados atuais (produção).
        log_to_mlflow: Se True, loga métricas no MLflow.

    Returns:
        Dicionário com resultados de drift.

    """
    logger.info(
        "Detectando drift: ref=%d samples, cur=%d samples",
        len(reference_data),
        len(current_data),
    )

    # Report Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_result = report.as_dict()
    drift_share = drift_result["metrics"][0]["result"]["share_of_drifted_columns"]

    # PSI por feature
    psi_scores = {}
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        psi = calculate_psi(
            reference_data[col].values,
            current_data[col].values,
        )
        psi_scores[col] = psi
        track_drift(col, psi)

    # Classificar resultado
    max_psi = max(psi_scores.values()) if psi_scores else 0.0

    if max_psi > PSI_CRITICAL_THRESHOLD:
        status = "critical"
        action = "retrain"
    elif max_psi > PSI_WARNING_THRESHOLD:
        status = "warning"
        action = "monitor"
    else:
        status = "stable"
        action = "none"

    result = {
        "status": status,
        "action": action,
        "drift_share": drift_share,
        "max_psi": max_psi,
        "psi_by_feature": psi_scores,
        "threshold_warning": PSI_WARNING_THRESHOLD,
        "threshold_critical": PSI_CRITICAL_THRESHOLD,
    }

    # Log no MLflow
    if log_to_mlflow:
        try:
            with mlflow.start_run(run_name="drift-detection", nested=True):
                mlflow.log_metrics({
                    "drift_share": drift_share,
                    "max_psi": max_psi,
                })
                for col, psi in psi_scores.items():
                    mlflow.log_metric(f"psi_{col}", psi)
                mlflow.set_tag("drift_status", status)
                mlflow.set_tag("drift_action", action)
        except Exception as e:
            logger.warning("Falha ao logar drift no MLflow: %s", str(e))

    logger.info(
        "Drift detection: status=%s, action=%s, max_psi=%.4f",
        status,
        action,
        max_psi,
    )

    return result


def should_retrain(drift_result: dict[str, Any]) -> bool:
    """Determina se o modelo deve ser retreinado.

    Args:
        drift_result: Resultado da detecção de drift.

    Returns:
        True se retrain é necessário.

    """
    return drift_result["action"] == "retrain"
