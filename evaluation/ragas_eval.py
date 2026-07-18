"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217

Métricas implementadas:
1. Faithfulness — resposta é fiel aos contextos?
2. Answer Relevancy — resposta é relevante à pergunta?
3. Context Precision — contextos recuperados são precisos?
4. Context Recall — contextos cobrem a resposta esperada?
"""

import json
import logging

import mlflow
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

load_dotenv()

logger = logging.getLogger(__name__)


def load_golden_set(golden_set_path: str) -> list[dict]:
    """Carrega golden set de avaliação.

    Args:
        golden_set_path: Caminho para JSON com golden set.

    Returns:
        Lista de dicionários com query, expected_answer, contexts.

    """
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    logger.info("Golden set carregado: %d pares", len(golden_set))
    return golden_set


def evaluate_rag_pipeline(
    golden_set_path: str,
    rag_fn,
    log_to_mlflow: bool = True,
) -> dict[str, float]:
    """Avalia pipeline RAG contra golden set.

    Args:
        golden_set_path: Caminho para JSON com golden set.
        rag_fn: Função que recebe query e retorna (answer, contexts).
        log_to_mlflow: Se True, loga métricas no MLflow.

    Returns:
        Dicionário com 4 métricas RAGAS.

    """
    golden_set = load_golden_set(golden_set_path)

    # Gera respostas do pipeline
    results = []
    for item in golden_set:
        answer, contexts = rag_fn(item["query"])
        results.append(
            {
                "question": item["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            }
        )

    dataset = Dataset.from_list(results)

    # Avaliação RAGAS — 4 métricas obrigatórias
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    # Log no MLflow
    if log_to_mlflow:
        try:
            with mlflow.start_run(run_name="ragas-evaluation", nested=True):
                mlflow.log_metrics(metrics)
                mlflow.log_param("golden_set_size", len(golden_set))
                mlflow.set_tag("evaluation_type", "ragas")
        except Exception as e:
            logger.warning("Falha ao logar RAGAS no MLflow: %s", str(e))

    logger.info("RAGAS scores: %s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.agent.rag_pipeline import rag_query

    golden_path = "data/golden_set/golden_set.json"
    metrics = evaluate_rag_pipeline(golden_path, rag_fn=rag_query)
    print("\n=== RAGAS Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
