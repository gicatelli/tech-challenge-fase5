"""Telemetria e tracing para RAG pipeline e agente.

Registra cada interação com: input, output, latência, tokens, tools usadas.
Suporta dois backends:
- MLflow (padrão): tracking de runs com params/metrics/tags
- Langfuse (opcional): se LANGFUSE_PUBLIC_KEY configurada

Uso:
    from src.monitoring.telemetry import trace_query

    with trace_query("minha pergunta") as trace:
        # fazer processamento
        trace.set_output("resposta")
        trace.set_tools(["tool1", "tool2"])
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TRACES_DIR = Path("metrics/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TraceRecord:
    """Registro de uma query processada."""

    query: str
    output: str = ""
    latency_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tools_used: list[str] = field(default_factory=list)
    contexts_retrieved: int = 0
    method: str = "unknown"
    success: bool = True
    error: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Converta para dicionário."""
        return {
            "query": self.query,
            "output": self.output[:200],
            "latency_ms": round(self.latency_ms, 2),
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tools_used": self.tools_used,
            "contexts_retrieved": self.contexts_retrieved,
            "method": self.method,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class QueryTracer:
    """Context manager para tracing de queries.

    Uso:
        with QueryTracer("pergunta") as trace:
            result = process(...)
            trace.set_output(result)
            trace.set_tools(["tool1"])
    """

    def __init__(self, query: str, method: str = "unknown"):
        """Inicializa tracer.

        Args:
            query: Pergunta do usuário.
            method: Método usado (agent, rag, tool).

        """
        self.record = TraceRecord(query=query, method=method)
        self._start_time = 0.0

    def __enter__(self):
        """Inicia medição."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza medição e loga."""
        self.record.latency_ms = (time.time() - self._start_time) * 1000
        self.record.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        if exc_type:
            self.record.success = False
            self.record.error = str(exc_val)

        # Estimar tokens (aproximação: 1 token ≈ 4 chars)
        if self.record.tokens_input == 0:
            self.record.tokens_input = len(self.record.query) // 4
        if self.record.tokens_output == 0:
            self.record.tokens_output = len(self.record.output) // 4

        # Logar
        _log_trace(self.record)

        return False  # Não suprimir exceções

    def set_output(self, output: str) -> None:
        """Define output da query."""
        self.record.output = output

    def set_tools(self, tools: list[str]) -> None:
        """Define tools utilizadas."""
        self.record.tools_used = tools

    def set_contexts(self, count: int) -> None:
        """Define número de contextos recuperados."""
        self.record.contexts_retrieved = count

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Define contagem de tokens."""
        self.record.tokens_input = input_tokens
        self.record.tokens_output = output_tokens


def _log_trace(record: TraceRecord) -> None:
    """Loga trace nos backends disponíveis."""
    trace_dict = record.to_dict()

    # 1. Log local (sempre)
    _log_to_file(trace_dict)

    # 2. MLflow (se disponível)
    _log_to_mlflow(trace_dict)

    # 3. Langfuse (se configurado)
    _log_to_langfuse(trace_dict)

    logger.info(
        "Trace: %s | %s | %.0fms | tools=%s",
        record.method,
        record.query[:40],
        record.latency_ms,
        record.tools_used,
    )


def _log_to_file(trace_dict: dict) -> None:
    """Salva trace em arquivo JSON (append)."""
    traces_file = TRACES_DIR / "query_traces.jsonl"
    with open(traces_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")


def _log_to_mlflow(trace_dict: dict) -> None:
    """Loga trace no MLflow como nested run."""
    try:
        import mlflow

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-fase05")
        )

        with mlflow.start_run(run_name=f"trace-{trace_dict['method']}", nested=True):
            mlflow.log_metrics({
                "latency_ms": trace_dict["latency_ms"],
                "tokens_input": trace_dict["tokens_input"],
                "tokens_output": trace_dict["tokens_output"],
                "contexts_retrieved": trace_dict["contexts_retrieved"],
                "num_tools": len(trace_dict["tools_used"]),
            })
            mlflow.log_params({
                "method": trace_dict["method"],
                "success": str(trace_dict["success"]),
            })
            mlflow.set_tag("query", trace_dict["query"][:100])
            mlflow.set_tag("tools", ",".join(trace_dict["tools_used"]))
    except Exception:
        pass  # MLflow offline — silencioso


def _log_to_langfuse(trace_dict: dict) -> None:
    """Loga trace no Langfuse (se configurado)."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not public_key:
        return

    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=public_key,
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        trace = langfuse.trace(
            name=f"query-{trace_dict['method']}",
            input=trace_dict["query"],
            output=trace_dict["output"],
            metadata={
                "latency_ms": trace_dict["latency_ms"],
                "tools_used": trace_dict["tools_used"],
                "contexts_retrieved": trace_dict["contexts_retrieved"],
                "success": trace_dict["success"],
            },
        )

        # Span para tokens
        trace.span(
            name="llm_call",
            input=trace_dict["query"],
            output=trace_dict["output"],
            metadata={
                "tokens_input": trace_dict["tokens_input"],
                "tokens_output": trace_dict["tokens_output"],
            },
        )

        langfuse.flush()
    except Exception as e:
        logger.debug("Langfuse indisponível: %s", e)


def trace_query(query: str, method: str = "unknown") -> QueryTracer:
    """Cria tracer para uma query.

    Args:
        query: Pergunta do usuário.
        method: Método (agent, rag, tool_direct).

    Returns:
        QueryTracer context manager.

    """
    return QueryTracer(query=query, method=method)


def get_trace_summary() -> dict:
    """Retorna resumo dos traces logados.

    Returns:
        Estatísticas agregadas dos traces.

    """
    traces_file = TRACES_DIR / "query_traces.jsonl"
    if not traces_file.exists():
        return {"total_queries": 0}

    traces = []
    with open(traces_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    if not traces:
        return {"total_queries": 0}

    latencies = [t["latency_ms"] for t in traces]
    return {
        "total_queries": len(traces),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "success_rate": round(
            sum(1 for t in traces if t["success"]) / len(traces), 2
        ),
        "tools_usage": _count_tools(traces),
        "methods": _count_methods(traces),
    }


def _count_tools(traces: list[dict]) -> dict[str, int]:
    """Conta uso de cada tool."""
    counts: dict[str, int] = {}
    for t in traces:
        for tool in t.get("tools_used", []):
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def _count_methods(traces: list[dict]) -> dict[str, int]:
    """Conta uso de cada método."""
    counts: dict[str, int] = {}
    for t in traces:
        method = t.get("method", "unknown")
        counts[method] = counts.get(method, 0) + 1
    return counts
