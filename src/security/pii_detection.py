"""Detecção e anonimização de PII (Personally Identifiable Information).

Usa Microsoft Presidio para detectar e anonimizar dados pessoais
em conformidade com a LGPD (Lei Geral de Proteção de Dados).

Entidades detectadas:
- CPF (BR_CPF)
- CNPJ
- Nomes de pessoas
- Emails
- Telefones
- Cartões de crédito
"""

import logging

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


def create_br_recognizers() -> list[PatternRecognizer]:
    """Cria reconhecedores de PII específicos para Brasil.

    Returns:
        Lista de PatternRecognizers para dados brasileiros.

    """
    # CPF: XXX.XXX.XXX-XX
    cpf_recognizer = PatternRecognizer(
        supported_entity="BR_CPF",
        patterns=[
            Pattern(
                name="cpf_pattern",
                regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
                score=0.85,
            )
        ],
    )

    # CNPJ: XX.XXX.XXX/XXXX-XX
    cnpj_recognizer = PatternRecognizer(
        supported_entity="BR_CNPJ",
        patterns=[
            Pattern(
                name="cnpj_pattern",
                regex=r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
                score=0.85,
            )
        ],
    )

    # Telefone BR: (XX) XXXXX-XXXX ou (XX) XXXX-XXXX
    phone_recognizer = PatternRecognizer(
        supported_entity="BR_PHONE",
        patterns=[
            Pattern(
                name="phone_pattern",
                regex=r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}",
                score=0.75,
            )
        ],
    )

    return [cpf_recognizer, cnpj_recognizer, phone_recognizer]


def create_analyzer() -> AnalyzerEngine:
    """Cria analyzer com reconhecedores brasileiros.

    Returns:
        AnalyzerEngine configurado.

    """
    analyzer = AnalyzerEngine()

    # Adicionar reconhecedores brasileiros
    for recognizer in create_br_recognizers():
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


def detect_pii(text: str, language: str = "pt") -> list[dict]:
    """Detecta PII em um texto.

    Args:
        text: Texto para análise.
        language: Idioma do texto.

    Returns:
        Lista de entidades PII encontradas.

    """
    analyzer = create_analyzer()

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "BR_CPF",
            "BR_CNPJ",
            "BR_PHONE",
        ],
    )

    pii_found = [
        {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": result.score,
            "text": text[result.start : result.end],
        }
        for result in results
    ]

    if pii_found:
        logger.warning("PII detectado: %d entidades", len(pii_found))

    return pii_found


def anonymize_text(text: str, language: str = "pt") -> str:
    """Anonimiza PII em um texto.

    Args:
        text: Texto com PII.
        language: Idioma do texto.

    Returns:
        Texto anonimizado.

    """
    analyzer = create_analyzer()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "BR_CPF",
            "BR_CNPJ",
            "BR_PHONE",
        ],
    )

    if not results:
        return text

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PESSOA>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CARTAO>"}),
            "BR_CPF": OperatorConfig("replace", {"new_value": "<CPF>"}),
            "BR_CNPJ": OperatorConfig("replace", {"new_value": "<CNPJ>"}),
            "BR_PHONE": OperatorConfig("replace", {"new_value": "<TELEFONE>"}),
        },
    )

    logger.info("Texto anonimizado: %d entidades substituídas", len(results))
    return anonymized.text
