"""Retraining automático disparado por drift detection.

Implementa o ciclo completo de Champion-Challenger:
1. Detecta drift (PSI > 0.2)
2. Se CRITICAL: dispara retrain com dados recentes
3. Treina challenger (novo modelo)
4. Compara com champion atual
5. Só promove se challenger for melhor (delta >= 0.5% RMSE)

Uso:
    python scripts/retrain_on_drift.py

Este script é o equivalente ao que seria um cron job ou trigger em produção.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Adicionar root ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def check_drift() -> dict:
    """Verifica status de drift atual.

    Returns:
        Resultado do drift detection.
    """
    from src.monitoring.drift import run_drift_detection

    logger.info("Verificando drift...")
    result = run_drift_detection()
    return result


def retrain_challenger(data_path: str = "data/raw/PETR4_SA_historico.csv") -> dict:
    """Treina challenger model com dados mais recentes.

    Args:
        data_path: Caminho para dados históricos.

    Returns:
        Métricas do challenger.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler

    from src.features.feature_engineering import compute_features, prepare_sequences

    logger.info("Treinando challenger com dados atualizados...")

    # Carregar dados
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"].astype(int)

    # Feature engineering
    features = compute_features(df, validate=False)

    # Normalizar
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    close_idx = features.columns.get_loc("close")
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]

    # Preparar sequências
    seq_length = 60
    X_seq, y_seq = prepare_sequences(
        features_scaled, target_col="close", sequence_length=seq_length
    )

    # Split temporal (usar últimos 20% como teste)
    split_idx = int(len(X_seq) * 0.8)
    X_train = X_seq[:split_idx, -1, :]  # RF usa último timestep
    X_test = X_seq[split_idx:, -1, :]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]

    # Treinar challenger (RF com hiperparâmetros ajustados)
    challenger = RandomForestRegressor(
        n_estimators=200,  # Mais árvores
        max_depth=15,      # Mais profundo
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    challenger.fit(X_train, y_train)

    # Predizer e desnormalizar
    y_pred_norm = challenger.predict(X_test)
    y_pred_real = y_pred_norm * (close_max - close_min) + close_min
    y_test_real = y_test * (close_max - close_min) + close_min

    # Métricas
    metrics = {
        "mae": float(mean_absolute_error(y_test_real, y_pred_real)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_real, y_pred_real))),
        "r2": float(r2_score(y_test_real, y_pred_real)),
    }

    logger.info(
        "Challenger treinado: MAE=R$%.2f, RMSE=R$%.2f, R²=%.4f",
        metrics["mae"], metrics["rmse"], metrics["r2"],
    )

    return metrics


def load_champion_metrics() -> dict:
    """Carrega métricas do champion atual.

    Returns:
        Métricas do champion.
    """
    metrics_file = Path("metrics/train_metrics.json")
    if not metrics_file.exists():
        logger.warning("Métricas do champion não encontradas. Usando defaults.")
        return {"mae": 999.0, "rmse": 999.0, "r2": -999.0}

    with open(metrics_file) as f:
        data = json.load(f)

    champion_name = data.get("champion", "lstm")
    champion_metrics = data.get(champion_name, {})

    return {
        "mae": champion_metrics.get("mae", 999.0),
        "rmse": champion_metrics.get("rmse", 999.0),
        "r2": champion_metrics.get("r2", -999.0),
    }


def compare_and_promote(
    champion_metrics: dict,
    challenger_metrics: dict,
    min_improvement_pct: float = 0.5,
) -> dict:
    """Compara challenger com champion e decide promoção.

    Args:
        champion_metrics: Métricas do modelo atual em produção.
        challenger_metrics: Métricas do novo modelo treinado.
        min_improvement_pct: Melhoria mínima em RMSE para promover (%).

    Returns:
        Resultado da comparação com decisão.
    """
    champion_rmse = champion_metrics["rmse"]
    challenger_rmse = challenger_metrics["rmse"]

    improvement_pct = (champion_rmse - challenger_rmse) / champion_rmse * 100

    result = {
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "improvement_rmse_pct": round(improvement_pct, 2),
        "min_required_pct": min_improvement_pct,
        "decision": "",
        "promoted": False,
    }

    if improvement_pct >= min_improvement_pct:
        result["decision"] = (
            f"PROMOVER: Challenger é {improvement_pct:.2f}% melhor que champion "
            f"(mínimo exigido: {min_improvement_pct}%). "
            f"RMSE: R${challenger_rmse:.2f} vs R${champion_rmse:.2f}."
        )
        result["promoted"] = True
    elif improvement_pct > 0:
        result["decision"] = (
            f"MANTER: Challenger é apenas {improvement_pct:.2f}% melhor "
            f"(mínimo exigido: {min_improvement_pct}%). "
            f"Melhoria insuficiente para justificar troca."
        )
    else:
        result["decision"] = (
            f"MANTER: Challenger é {abs(improvement_pct):.2f}% PIOR que champion. "
            f"RMSE: R${challenger_rmse:.2f} vs R${champion_rmse:.2f}."
        )

    return result


def run_retrain_pipeline():
    """Executa pipeline completo de retraining condicional."""
    print("=" * 70)
    print("  RETRAINING AUTOMÁTICO — CHAMPION-CHALLENGER")
    print("  Trigger: Drift Detection")
    print("=" * 70)

    # Step 1: Verificar drift
    print("\n[1/4] Verificando drift...")
    drift_result = check_drift()
    print(f"  Status: {drift_result['status'].upper()}")
    print(f"  Max PSI: {drift_result['max_psi']:.4f}")
    print(f"  Ação recomendada: {drift_result['action']}")

    if drift_result["action"] != "retrain":
        print(f"\n✓ Drift status: {drift_result['status'].upper()}")
        print("  Retrain NÃO necessário. Champion mantido.")
        return {
            "triggered": False,
            "reason": f"Drift status={drift_result['status']}, action={drift_result['action']}",
        }

    # Step 2: Carregar métricas do champion
    print("\n[2/4] Carregando métricas do champion atual...")
    champion_metrics = load_champion_metrics()
    print(f"  Champion RMSE: R${champion_metrics['rmse']:.2f}")
    print(f"  Champion MAE:  R${champion_metrics['mae']:.2f}")
    print(f"  Champion R²:   {champion_metrics['r2']:.4f}")

    # Step 3: Treinar challenger
    print("\n[3/4] Treinando challenger...")
    challenger_metrics = retrain_challenger()
    print(f"  Challenger RMSE: R${challenger_metrics['rmse']:.2f}")
    print(f"  Challenger MAE:  R${challenger_metrics['mae']:.2f}")
    print(f"  Challenger R²:   {challenger_metrics['r2']:.4f}")

    # Step 4: Comparar e decidir
    print("\n[4/4] Comparando champion vs challenger...")
    comparison = compare_and_promote(champion_metrics, challenger_metrics)

    print("\n" + "=" * 70)
    print("  RESULTADO")
    print("=" * 70)
    print(f"\n  Melhoria RMSE: {comparison['improvement_rmse_pct']:.2f}%")
    print(f"  Mínimo para promoção: {comparison['min_required_pct']}%")
    print(f"  Decisão: {comparison['decision']}")

    if comparison["promoted"]:
        print("\n  🏆 CHALLENGER PROMOVIDO A CHAMPION")
        print("  → Em produção: Model Registry seria atualizado automaticamente")
    else:
        print("\n  🛡️ CHAMPION MANTIDO")
        print("  → Challenger descartado. Aguardando próximo ciclo de retrain.")

    # Salvar resultado
    output_path = Path("metrics/retrain_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "drift": drift_result,
                "comparison": comparison,
                "triggered": True,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Resultado salvo em: {output_path}")

    return {"triggered": True, "comparison": comparison}


if __name__ == "__main__":
    run_retrain_pipeline()
