"""Tools customizadas para o agente ReAct.

Implementa ≥ 3 tools relevantes ao domínio do Datathon.
Cada tool é uma função que o agente pode invocar para
resolver problemas específicos.

Adapte as tools ao domínio da empresa convidada.
"""

import json
import logging

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


def search_knowledge_base(query: str) -> str:
    """Busca informações na base de conhecimento via RAG.

    Args:
        query: Pergunta do usuário.

    Returns:
        Contexto relevante encontrado na base.

    """
    # TODO: Integrar com o RAG pipeline real
    from src.agent.rag_pipeline import retrieve_context

    contexts = retrieve_context(query, top_k=3)
    if not contexts:
        return "Nenhuma informação relevante encontrada na base de conhecimento."

    formatted = "\n\n".join([f"[Fonte {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
    logger.info("Busca RAG retornou %d contextos para: %s", len(contexts), query[:50])
    return formatted


def analyze_data(query: str) -> str:
    """Analisa dados e retorna insights estatísticos.

    Args:
        query: Descrição da análise desejada.

    Returns:
        Resultado da análise em formato textual.

    """
    # TODO: Implementar análise real com os dados do domínio
    logger.info("Análise de dados solicitada: %s", query[:50])
    return (
        "Análise de dados executada. "
        "Adapte esta tool para realizar análises específicas "
        "do domínio da empresa convidada."
    )


def get_model_prediction(input_data: str) -> str:
    """Obtém predição do modelo baseline.

    Args:
        input_data: JSON com features para predição.

    Returns:
        Resultado da predição.

    """
    try:
        data = json.loads(input_data)
        # TODO: Carregar modelo do MLflow e fazer predição real
        logger.info("Predição solicitada para: %s", str(data)[:100])
        return json.dumps(
            {
                "prediction": 1,
                "probability": 0.85,
                "model_version": "1.0.0",
                "note": "Adapte para usar o modelo real do MLflow Registry",
            }
        )
    except json.JSONDecodeError:
        return "Erro: input deve ser um JSON válido com as features do modelo."


def calculate_risk_score(query: str) -> str:
    """Calcula score de risco baseado em parâmetros fornecidos.

    Args:
        query: Descrição do cenário para cálculo de risco.

    Returns:
        Score de risco calculado.

    """
    # TODO: Implementar cálculo de risco real
    logger.info("Cálculo de risco solicitado: %s", query[:50])
    return (
        "Score de risco: 0.65 (médio). "
        "Adapte esta tool ao domínio específico da empresa."
    )


def get_available_tools() -> list[Tool]:
    """Retorna lista de tools disponíveis para o agente.

    Returns:
        Lista com ≥ 3 tools configuradas.

    """
    tools = [
        Tool(
            name="buscar_base_conhecimento",
            func=search_knowledge_base,
            description=(
                "Busca informações na base de conhecimento da empresa. "
                "Use quando precisar de contexto sobre políticas, procedimentos, "
                "dados históricos ou documentação interna."
            ),
        ),
        Tool(
            name="analisar_dados",
            func=analyze_data,
            description=(
                "Analisa dados e retorna insights estatísticos. "
                "Use quando precisar de análises quantitativas, "
                "tendências ou padrões nos dados."
            ),
        ),
        Tool(
            name="obter_predicao",
            func=get_model_prediction,
            description=(
                "Obtém predição do modelo de ML. "
                "Input deve ser JSON com as features necessárias. "
                "Use quando precisar de uma predição do modelo treinado."
            ),
        ),
        Tool(
            name="calcular_risco",
            func=calculate_risk_score,
            description=(
                "Calcula score de risco para um cenário específico. "
                "Use quando precisar avaliar o nível de risco de uma situação."
            ),
        ),
    ]

    logger.info("Tools disponíveis: %d", len(tools))
    return tools
