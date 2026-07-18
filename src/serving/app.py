"""FastAPI endpoint para o Datathon.

Expõe o agente ReAct e o pipeline RAG via API REST documentada.
Inclui health check, métricas Prometheus e guardrails de segurança.

Endpoints:
- GET  /health   — Health check
- POST /query    — Consulta via agente ReAct ou RAG
- POST /analyze  — Análise direta de dados (sem LLM)
- POST /predict  — Previsão de preços (sem LLM)
- POST /risk     — Cálculo de risco (sem LLM)
- GET  /tools    — Lista tools disponíveis
- GET  /metrics  — Métricas Prometheus
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.agent.tools import (
    analisar_historico,
    buscar_conhecimento,
    calcular_risco,
    prever_preco,
)
from src.monitoring.metrics import (
    track_request,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)

# Guardrails
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()

# Agent (lazy init — requer OPENAI_API_KEY)
_agent = None


def get_agent():
    """Lazy initialization do agente."""
    global _agent
    if _agent is None:
        from src.agent.react_agent import create_datathon_agent

        _agent = create_datathon_agent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle da aplicação."""
    logger.info("Inicializando API Datathon Fase 05")
    yield
    logger.info("Encerrando API")


app = FastAPI(
    title="Datathon Fase 05 - API",
    description="API do agente inteligente para o Datathon FIAP Pós-Tech",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Schemas
# ============================================


class QueryRequest(BaseModel):
    """Request para consulta ao agente."""

    query: str = Field(..., min_length=1, max_length=4096, description="Pergunta do usuário")
    use_agent: bool = Field(
        default=True,
        description="Se True, usa agente ReAct. Se False, RAG direto.",
    )


class QueryResponse(BaseModel):
    """Response da consulta."""

    answer: str = Field(..., description="Resposta gerada")
    contexts: list[str] = Field(default_factory=list, description="Contextos utilizados")
    steps: int = Field(default=0, description="Número de steps do agente")
    latency_ms: float = Field(..., description="Latência em milissegundos")


class HealthResponse(BaseModel):
    """Response do health check."""

    status: str
    version: str
    model: str


# ============================================
# Endpoints
# ============================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check da API."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model="gpt-4o-mini",
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """Consulta o agente inteligente.

    Fluxo:
    1. Valida input (guardrails)
    2. Processa via agente ReAct ou RAG direto
    3. Sanitiza output (PII removal)
    4. Retorna resposta
    """
    start_time = time.time()

    # Guardrail de input
    is_valid, reason = input_guardrail.validate(request.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    try:
        if request.use_agent:
            from src.agent.react_agent import run_agent

            agent = get_agent()
            result = run_agent(request.query, agent)
            answer = result["answer"]
            contexts: list[str] = []
            steps = result["steps"]
        else:
            from src.agent.rag_pipeline import rag_query

            answer, contexts = rag_query(request.query)
            steps = 0

        # Guardrail de output (PII removal)
        answer = output_guardrail.sanitize(answer)

        latency_ms = (time.time() - start_time) * 1000

        # Métricas
        track_request(latency_ms=latency_ms, success=True)

        return QueryResponse(
            answer=answer,
            contexts=contexts,
            steps=steps,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        track_request(latency_ms=latency_ms, success=False)
        logger.error("Erro no processamento: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/metrics", tags=["System"])
async def metrics():
    """Endpoint Prometheus para métricas."""
    return Response(content=generate_latest(), media_type="text/plain")


# ============================================
# Endpoints diretos (sem LLM — tools reais)
# ============================================


class ToolRequest(BaseModel):
    """Request genérica para tools diretas."""

    input: str = Field(..., min_length=1, max_length=2048, description="Input para a tool")


class ToolResponse(BaseModel):
    """Response de tool direta."""

    tool: str
    result: str
    latency_ms: float


@app.post("/analyze", response_model=ToolResponse, tags=["Tools"])
async def analyze_endpoint(request: ToolRequest):
    """Análise de dados históricos (sem LLM)."""
    start = time.time()
    result = analisar_historico(request.input)
    latency = (time.time() - start) * 1000
    return ToolResponse(tool="analisar_historico", result=result, latency_ms=round(latency, 2))


@app.post("/predict", response_model=ToolResponse, tags=["Tools"])
async def predict_endpoint(request: ToolRequest):
    """Previsão de preço (sem LLM)."""
    start = time.time()
    result = prever_preco(request.input)
    latency = (time.time() - start) * 1000
    return ToolResponse(tool="prever_preco", result=result, latency_ms=round(latency, 2))


@app.post("/risk", response_model=ToolResponse, tags=["Tools"])
async def risk_endpoint(request: ToolRequest):
    """Cálculo de risco (sem LLM)."""
    start = time.time()
    result = calcular_risco(request.input)
    latency = (time.time() - start) * 1000
    return ToolResponse(tool="calcular_risco", result=result, latency_ms=round(latency, 2))


@app.post("/search", response_model=ToolResponse, tags=["Tools"])
async def search_endpoint(request: ToolRequest):
    """Busca na base de conhecimento (sem LLM)."""
    start = time.time()
    result = buscar_conhecimento(request.input)
    latency = (time.time() - start) * 1000
    return ToolResponse(tool="buscar_conhecimento", result=result, latency_ms=round(latency, 2))


@app.get("/tools", tags=["System"])
async def list_tools():
    """Lista tools disponíveis do agente."""
    return {
        "tools": [
            {
                "name": "prever_preco",
                "endpoint": "/predict",
                "description": "Previsão de preço PETR4",
            },
            {
                "name": "analisar_historico",
                "endpoint": "/analyze",
                "description": "Análise de dados históricos",
            },
            {
                "name": "buscar_conhecimento",
                "endpoint": "/search",
                "description": "Busca na base de conhecimento",
            },
            {
                "name": "calcular_risco",
                "endpoint": "/risk",
                "description": "Cálculo de métricas de risco",
            },
        ],
        "agent_endpoint": "/query",
        "note": "Use /query para consultas que combinam múltiplas tools via agente ReAct.",
    }


if __name__ == "__main__":
    import uvicorn

    # Bind em 0.0.0.0 é intencional para acesso dentro do container Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
