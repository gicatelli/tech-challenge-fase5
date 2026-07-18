"""FastAPI endpoint para o Datathon.

Expõe o agente ReAct e o pipeline RAG via API REST documentada.
Inclui health check, métricas Prometheus e guardrails de segurança.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.agent.rag_pipeline import rag_query
from src.agent.react_agent import create_datathon_agent, run_agent
from src.monitoring.metrics import (
    track_request,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)

# Guardrails
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()

# Agent (lazy init)
_agent = None


def get_agent():
    """Lazy initialization do agente."""
    global _agent
    if _agent is None:
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
            agent = get_agent()
            result = run_agent(request.query, agent)
            answer = result["answer"]
            contexts: list[str] = []
            steps = result["steps"]
        else:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
