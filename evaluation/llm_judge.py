"""LLM-as-Judge — Avaliação com ≥ 3 critérios.

Implementa avaliação automatizada usando LLM como juiz,
com critérios técnicos e de negócio.

Critérios implementados:
1. Correção factual — a resposta está correta?
2. Completude — a resposta cobre todos os aspectos?
3. Relevância de negócio — a resposta é útil para o domínio?
4. Clareza — a resposta é clara e bem estruturada?
"""

import json
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """Você é um avaliador especialista. Avalie a resposta do assistente
com base nos critérios abaixo. Para cada critério, dê uma nota de 1 a 5.

## Pergunta do usuário:
{question}

## Resposta esperada (ground truth):
{ground_truth}

## Resposta do assistente:
{answer}

## Critérios de avaliação:

1. **Correção factual** (1-5): A resposta está factualmente correta?
   Compara com a resposta esperada.

2. **Completude** (1-5): A resposta cobre todos os aspectos relevantes
   da pergunta?

3. **Relevância de negócio** (1-5): A resposta é útil e aplicável ao
   contexto de negócio? Fornece insights acionáveis?

4. **Clareza** (1-5): A resposta é clara, bem estruturada e fácil
   de entender?

Responda APENAS no formato JSON:
{{
    "correcao_factual": <nota>,
    "completude": <nota>,
    "relevancia_negocio": <nota>,
    "clareza": <nota>,
    "justificativa": "<breve justificativa>"
}}"""


def evaluate_with_llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    model_name: str | None = None,
) -> dict:
    """Avalia uma resposta usando LLM como juiz.

    Args:
        question: Pergunta original.
        answer: Resposta do assistente.
        ground_truth: Resposta esperada.
        model_name: Modelo do juiz.

    Returns:
        Dicionário com notas e justificativa.

    """
    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    response = llm.invoke(prompt)

    try:
        scores = json.loads(response.content)
    except json.JSONDecodeError:
        logger.error("Falha ao parsear resposta do juiz: %s", response.content[:200])
        scores = {
            "correcao_factual": 0,
            "completude": 0,
            "relevancia_negocio": 0,
            "clareza": 0,
            "justificativa": "Erro no parsing da resposta do juiz.",
        }

    return scores


def evaluate_golden_set(
    golden_set_path: str,
    rag_fn,
) -> dict[str, float]:
    """Avalia todo o golden set com LLM-as-judge.

    Args:
        golden_set_path: Caminho para golden set JSON.
        rag_fn: Função RAG que recebe query e retorna (answer, contexts).

    Returns:
        Médias dos critérios de avaliação.

    """
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    all_scores = []

    for item in golden_set:
        answer, _ = rag_fn(item["query"])
        scores = evaluate_with_llm_judge(
            question=item["query"],
            answer=answer,
            ground_truth=item["expected_answer"],
        )
        all_scores.append(scores)

    # Calcular médias
    criteria = ["correcao_factual", "completude", "relevancia_negocio", "clareza"]
    averages = {}
    for criterion in criteria:
        values = [s[criterion] for s in all_scores if isinstance(s.get(criterion), (int, float))]
        averages[criterion] = sum(values) / len(values) if values else 0.0

    averages["overall"] = sum(averages.values()) / len(averages) if averages else 0.0

    logger.info("LLM Judge averages: %s", averages)
    return averages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.agent.rag_pipeline import rag_query

    golden_path = "data/golden_set/golden_set.json"
    averages = evaluate_golden_set(golden_path, rag_fn=rag_query)
    print("\n=== LLM-as-Judge Results ===")
    for criterion, score in averages.items():
        print(f"  {criterion}: {score:.2f}/5.0")
