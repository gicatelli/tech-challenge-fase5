"""Prometheus custom metrics para observabilidade.

Métricas operacionais:
- Latência de requests
- Contagem de requests (sucesso/erro)
- Throughput
- Métricas de qualidade do agente
"""

import logging

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

# ============================================
# Métricas de Request
# ============================================

REQUEST_COUNT = Counter(
    "datathon_requests_total",
    "Total de requests processados",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "datathon_request_latency_ms",
    "Latência de requests em milissegundos",
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

# ============================================
# Métricas do Agente
# ============================================

AGENT_STEPS = Histogram(
    "datathon_agent_steps",
    "Número de steps do agente por query",
    buckets=[1, 2, 3, 5, 7, 10],
)

AGENT_TOOL_USAGE = Counter(
    "datathon_agent_tool_usage_total",
    "Uso de tools pelo agente",
    ["tool_name"],
)

# ============================================
# Métricas de RAG
# ============================================

RAG_RETRIEVAL_COUNT = Histogram(
    "datathon_rag_contexts_retrieved",
    "Número de contextos recuperados por query",
    buckets=[0, 1, 2, 3, 5, 10],
)

RAG_CONTEXT_RELEVANCE = Gauge(
    "datathon_rag_context_relevance",
    "Score médio de relevância dos contextos",
)

# ============================================
# Métricas de Drift
# ============================================

DRIFT_SCORE = Gauge(
    "datathon_drift_score",
    "Score de drift atual (PSI)",
    ["feature_name"],
)

DRIFT_ALERT = Counter(
    "datathon_drift_alerts_total",
    "Total de alertas de drift disparados",
    ["severity"],
)

# ============================================
# Info do modelo
# ============================================

MODEL_INFO = Info(
    "datathon_model",
    "Informações do modelo em produção",
)


# ============================================
# Funções auxiliares
# ============================================


def track_request(latency_ms: float, success: bool) -> None:
    """Registra métricas de um request.

    Args:
        latency_ms: Latência em milissegundos.
        success: Se o request foi bem-sucedido.

    """
    status = "success" if success else "error"
    REQUEST_COUNT.labels(status=status).inc()
    REQUEST_LATENCY.observe(latency_ms)


def track_agent_execution(steps: int, tools_used: list[str]) -> None:
    """Registra métricas de execução do agente.

    Args:
        steps: Número de steps executados.
        tools_used: Lista de tools utilizadas.

    """
    AGENT_STEPS.observe(steps)
    for tool in tools_used:
        AGENT_TOOL_USAGE.labels(tool_name=tool).inc()


def track_drift(feature_name: str, psi_score: float) -> None:
    """Registra score de drift para uma feature.

    Args:
        feature_name: Nome da feature.
        psi_score: Score PSI calculado.

    """
    DRIFT_SCORE.labels(feature_name=feature_name).set(psi_score)

    if psi_score > 0.2:
        DRIFT_ALERT.labels(severity="critical").inc()
        logger.warning("DRIFT CRÍTICO: %s PSI=%.4f", feature_name, psi_score)
    elif psi_score > 0.1:
        DRIFT_ALERT.labels(severity="warning").inc()
        logger.warning("DRIFT WARNING: %s PSI=%.4f", feature_name, psi_score)


def update_model_info(model_name: str, version: str, framework: str) -> None:
    """Atualiza informações do modelo em produção.

    Args:
        model_name: Nome do modelo.
        version: Versão do modelo.
        framework: Framework utilizado.

    """
    MODEL_INFO.info({
        "model_name": model_name,
        "version": version,
        "framework": framework,
    })
