"""A/B Test de Prompts — Benchmark com ≥ 3 configurações.

Compara diferentes configurações de prompt/RAG para
identificar a melhor combinação.
"""

import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuração de prompt para teste."""

    name: str
    system_prompt: str
    temperature: float
    top_k: int
    chunk_size: int


# ≥ 3 configurações para benchmark
CONFIGS = [
    PromptConfig(
        name="config_conservative",
        system_prompt="Responda de forma precisa e concisa, citando fontes.",
        temperature=0.0,
        top_k=3,
        chunk_size=1000,
    ),
    PromptConfig(
        name="config_balanced",
        system_prompt="Responda de forma completa e informativa, com exemplos quando possível.",
        temperature=0.3,
        top_k=5,
        chunk_size=800,
    ),
    PromptConfig(
        name="config_creative",
        system_prompt="Responda de forma detalhada, explorando diferentes perspectivas.",
        temperature=0.7,
        top_k=7,
        chunk_size=500,
    ),
]


def run_ab_test(
    queries: list[str],
    configs: list[PromptConfig] | None = None,
) -> dict[str, dict]:
    """Executa A/B test com diferentes configurações.

    Args:
        queries: Lista de queries para teste.
        configs: Configurações a testar. Se None, usa padrão.

    Returns:
        Resultados por configuração.
    """
    if configs is None:
        configs = CONFIGS

    results = {}

    for config in configs:
        logger.info("Testando configuração: %s", config.name)
        config_results = {
            "latencies": [],
            "answers": [],
            "config": {
                "temperature": config.temperature,
                "top_k": config.top_k,
                "chunk_size": config.chunk_size,
            },
        }

        for query in queries:
            start = time.time()
            # TODO: Integrar com RAG pipeline real usando config
            answer = f"[Resposta para '{query[:30]}...' com config {config.name}]"
            latency = (time.time() - start) * 1000

            config_results["latencies"].append(latency)
            config_results["answers"].append(answer)

        # Métricas agregadas
        config_results["avg_latency_ms"] = sum(config_results["latencies"]) / len(
            config_results["latencies"]
        )
        config_results["total_queries"] = len(queries)

        results[config.name] = config_results

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "Qual a política de crédito?",
        "Como funciona a detecção de fraude?",
        "Quais são os limites de transação?",
    ]

    results = run_ab_test(test_queries)

    print("\n=== A/B Test Results ===")
    for config_name, data in results.items():
        print(f"\n{config_name}:")
        print(f"  Avg Latency: {data['avg_latency_ms']:.2f}ms")
        print(f"  Queries: {data['total_queries']}")
