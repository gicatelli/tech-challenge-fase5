"""Drift detection com Evidently.

Implementa detecção de data drift e concept drift:
- PSI (Population Stability Index) como métrica principal
- Threshold: PSI > 0.1 = warning, PSI > 0.2 = retrain trigger
- Integração com MLflow para logging de métricas de drift
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

try:
    from src.monitoring.metrics import track_drift
except ImportError:
    def track_drift(feature_name: str, psi_score: float) -> None:  # type: ignore[misc]
        """Fallback quando prometheus não disponível."""

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

    # Report Evidently (se disponível)
    drift_share = 0.0
    if HAS_EVIDENTLY:
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
    if log_to_mlflow and mlflow is not None:
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
    return drift_result["action"] == "retrain"  # type: ignore[no-any-return]


def run_drift_detection(
    data_path: str = "data/raw/PETR4_SA_historico.csv",
    reference_months: int = 6,
    current_months: int = 1,
) -> dict:
    """Executa drift detection com dados reais da PETR4.

    Usa os últimos N meses como referência e o último mês como current.

    Args:
        data_path: Caminho para dados históricos.
        reference_months: Meses para dados de referência.
        current_months: Meses para dados atuais.

    Returns:
        Resultado da detecção de drift.

    """
    from pathlib import Path

    try:
        from src.features.feature_engineering import compute_features
        has_feature_eng = True
    except ImportError:
        has_feature_eng = False

    # Carregar dados
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"].astype(int)

    # Computar features (ou usar raw se pandera indisponível)
    if has_feature_eng:
        features = compute_features(df, validate=False)
    else:
        # Fallback: calcular features básicas sem pandera
        features = pd.DataFrame(index=df.index)
        features["close"] = df["Close"]
        features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        features["volatility_30"] = features["log_return"].rolling(30).std() * np.sqrt(252)
        features["volume_norm"] = df["Volume"] / df["Volume"].rolling(30).mean()
        sma_30 = df["Close"].rolling(30).mean()
        features["price_sma30_ratio"] = df["Close"] / sma_30
        # RSI simplificado
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))
        features = features.dropna()

    # Split temporal: referência (6 meses antes) vs current (último mês)
    total_days = len(features)
    current_size = int(total_days * (current_months / (reference_months + current_months)))
    reference_size = total_days - current_size

    reference_data = features.iloc[:reference_size]
    current_data = features.iloc[reference_size:]

    logger.info(
        "Split: referência=%d dias, current=%d dias",
        len(reference_data),
        len(current_data),
    )

    # Selecionar features numéricas para drift (excluir OHLC raw)
    drift_features = [
        "rsi_14", "macd", "bb_width", "log_return",
        "volatility_30", "volume_norm", "price_sma30_ratio",
    ]
    available = [f for f in drift_features if f in features.columns]

    ref_subset = reference_data[available]
    cur_subset = current_data[available]

    # Detectar drift
    result = detect_drift(ref_subset, cur_subset, log_to_mlflow=False)

    # Salvar resultado
    import json

    output_path = Path("metrics/drift_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("  DRIFT DETECTION — DATATHON FASE 05")
    print("  Referência: 6 meses | Current: 1 mês")
    print("=" * 60)

    result = run_drift_detection()

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  Status: {result['status'].upper()}")
    print(f"  Ação: {result['action']}")
    print(f"  Max PSI: {result['max_psi']:.4f}")
    print(f"  Drift share: {result['drift_share']:.2%}")
    print("\n  PSI por feature:")
    for feat, psi in sorted(
        result["psi_by_feature"].items(), key=lambda x: x[1], reverse=True
    ):
        status = (
            "CRITICAL" if psi > PSI_CRITICAL_THRESHOLD
            else "WARNING" if psi > PSI_WARNING_THRESHOLD
            else "OK"
        )
        bar = "#" * int(psi * 50)
        print(f"    {feat:<20} {psi:.4f} [{bar:<10}] {status}")
