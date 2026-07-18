"""LLM-as-Judge — Avaliação com 4 critérios (incluindo negócio).

Implementa avaliação automatizada usando LLM como juiz,
com critérios técnicos e de negócio adaptados ao domínio financeiro.

Critérios:
1. Correção factual — a resposta está correta comparada ao ground truth?
2. Completude — cobre todos os aspectos relevantes da pergunta?
3. Relevância de negócio — ajudaria um investidor a tomar decisão?
4. Clareza — é clara, bem estruturada e fácil de entender?

Executar: python -m evaluation.llm_judge
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

load_dotenv()

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_set/golden_set.json"
METRICS_OUTPUT = "metrics/llm_judge_metrics.json"

JUDGE_PROMPT = """Você é um avaliador especialista em mercado financeiro.
Avalie a resposta do assistente considerando que o público-alvo são INVESTIDORES
que precisam de informações para TOMAR DECISÕES de investimento.

## Pergunta do usuário:
{question}

## Resposta esperada (ground truth):
{ground_truth}

## Resposta do assistente (para avaliar):
{answer}

## Critérios de avaliação (nota 1 a 5 cada):

1. **Correção factual**: A resposta está factualmente correta?
   Dados, números e conceitos estão alinhados com o ground truth?
   (1=incorreto, 3=parcialmente correto, 5=totalmente correto)

2. **Completude**: A resposta cobre todos os aspectos relevantes?
   Faltam informações importantes que estão no ground truth?
   (1=muito incompleto, 3=cobre o básico, 5=completo)

3. **Relevância de negócio**: A resposta ajudaria um INVESTIDOR a tomar
   uma decisão? Fornece insights acionáveis sobre risco, retorno ou timing?
   (1=inútil para decisão, 3=informativa, 5=diretamente acionável)

4. **Clareza**: A resposta é clara, bem organizada e acessível?
   Um investidor não-técnico conseguiria entender?
   (1=confusa, 3=razoável, 5=cristalina)

Responda APENAS no formato JSON (sem markdown):
{{"correcao_factual": <1-5>, "completude": <1-5>, "relevancia_negocio": <1-5>, "clareza": <1-5>, "justificativa": "<1 frase>"}}"""


def evaluate_single_with_llm(
    question: str,
    answer: str,
    ground_truth: str,
) -> dict:
    """Avalia uma resposta usando LLM como juiz.

    Args:
        question: Pergunta original.
        answer: Resposta do assistente.
        ground_truth: Resposta esperada.

    Returns:
        Dicionário com notas (1-5) por critério.

    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore[arg-type]
    )

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    response = llm.invoke(prompt)

    try:
        scores = json.loads(str(response.content))
    except json.JSONDecodeError:
        logger.error("Falha ao parsear resposta do juiz")
        scores = {
            "correcao_factual": 3,
            "completude": 3,
            "relevancia_negocio": 3,
            "clareza": 3,
            "justificativa": "Erro no parsing — score default aplicado.",
        }

    return scores


def evaluate_single_proxy(
    question: str,
    answer: str,
    ground_truth: str,
) -> dict:
    """Avalia com heurísticas (fallback sem OpenAI).

    Calcula scores baseados em overlap de tokens e comprimento.

    Args:
        question: Pergunta original.
        answer: Resposta do assistente.
        ground_truth: Resposta esperada.

    Returns:
        Dicionário com notas aproximadas (1-5).

    """
    answer_words = set(answer.lower().split())
    truth_words = set(ground_truth.lower().split())
    query_words = set(question.lower().split())

    # Correção: overlap entre answer e ground truth
    if truth_words:
        overlap = len(answer_words & truth_words) / len(truth_words)
        correcao = min(5, max(1, int(overlap * 5) + 1))
    else:
        correcao = 3

    # Completude: comprimento da answer vs ground truth
    ratio = len(answer) / max(len(ground_truth), 1)
    completude = min(5, max(1, int(ratio * 3) + 1))

    # Relevância: palavras de ação financeira na resposta
    finance_words = {
        "risco", "retorno", "investir", "comprar", "vender",
        "preço", "volatilidade", "dividendo", "lucro", "perda",
        "alta", "baixa", "tendência", "mercado", "ação",
    }
    finance_overlap = len(answer_words & finance_words)
    relevancia = min(5, max(1, finance_overlap + 1))

    # Clareza: frases curtas e estruturadas
    sentences = answer.split(".")
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    clareza = 5 if avg_sentence_len < 20 else 4 if avg_sentence_len < 30 else 3

    return {
        "correcao_factual": correcao,
        "completude": completude,
        "relevancia_negocio": relevancia,
        "clareza": clareza,
        "justificativa": "Avaliação por heurísticas (proxy, sem LLM juiz).",
    }


def run_llm_judge(
    golden_set_path: str = GOLDEN_SET_PATH,
    output_path: str = METRICS_OUTPUT,
    log_to_mlflow: bool = True,
) -> dict[str, float]:
    """Executa LLM-as-judge em todo o golden set.

    Args:
        golden_set_path: Caminho para golden set.
        output_path: Caminho para salvar métricas.
        log_to_mlflow: Se True, loga no MLflow.

    Returns:
        Médias dos 4 critérios.

    """
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    use_llm = os.getenv("OPENAI_API_KEY") is not None
    method = "llm_judge" if use_llm else "proxy_heuristic"
    logger.info("Método: %s | Golden set: %d pares", method, len(golden_set))

    all_scores = []

    for i, item in enumerate(golden_set):
        logger.info("Avaliando %d/%d: %s", i + 1, len(golden_set), item["query"][:40])

        # Gerar resposta (usando fallback keyword search)
        answer = _get_answer(item["query"])

        # Avaliar
        if use_llm:
            try:
                scores = evaluate_single_with_llm(
                    item["query"], answer, item["expected_answer"]
                )
            except Exception as e:
                logger.warning("LLM judge falhou (%s), usando proxy", e)
                scores = evaluate_single_proxy(
                    item["query"], answer, item["expected_answer"]
                )
        else:
            scores = evaluate_single_proxy(
                item["query"], answer, item["expected_answer"]
            )

        all_scores.append(scores)

    # Calcular médias
    criteria = ["correcao_factual", "completude", "relevancia_negocio", "clareza"]
    averages = {}
    for criterion in criteria:
        values = [s[criterion] for s in all_scores if isinstance(s.get(criterion), (int, float))]
        averages[criterion] = round(sum(values) / len(values), 2) if values else 0.0

    averages["overall"] = round(sum(averages[c] for c in criteria) / len(criteria), 2)

    # Resultado completo
    result = {
        "averages": averages,
        "method": method,
        "golden_set_size": len(golden_set),
        "individual_scores": all_scores,
    }

    # Salvar
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Métricas salvas em %s", output)

    # MLflow
    if log_to_mlflow and mlflow is not None:
        try:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment(
                os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-fase05")
            )
            with mlflow.start_run(run_name=f"llm-judge-{method}"):
                mlflow.log_metrics(averages)
                mlflow.log_param("method", method)
                mlflow.log_param("golden_set_size", len(golden_set))
                mlflow.set_tag("evaluation_type", "llm_judge")
                mlflow.set_tag("phase", "datathon-fase05")
        except Exception as e:
            logger.warning("MLflow indisponível: %s", e)

    return averages


def _get_answer(query: str) -> str:
    """Gera resposta para uma query (com fallback)."""
    try:
        from src.agent.rag_pipeline import rag_query
        answer, _ = rag_query(query)
        return answer
    except (ImportError, Exception):
        # Fallback: keyword search
        kb_dir = Path("data/knowledge_base")
        if not kb_dir.exists():
            return "Informação não disponível."

        query_lower = query.lower()
        results = []
        for f in kb_dir.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            for para in content.split("\n\n"):
                score = sum(1 for w in query_lower.split() if w in para.lower())
                if score >= 2 and len(para) > 50:
                    results.append((score, para.strip()))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[0][1][:500] if results else "Informação não encontrada."


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("  LLM-AS-JUDGE — DATATHON FASE 05")
    print("  4 critérios: correção, completude, negócio, clareza")
    print("=" * 60)

    averages = run_llm_judge()

    print("\n" + "=" * 60)
    print("  RESULTADOS (notas 1-5)")
    print("=" * 60)
    for criterion, score in averages.items():
        bar = "#" * int(score * 4)
        print(f"  {criterion:<25} {score:.2f}/5.0 [{bar:<20}]")
