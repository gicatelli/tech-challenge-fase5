"""Guardrails de segurança para input e output do agente.

Referência: OWASP Top 10 for LLM Applications (2025)
            https://owasp.org/www-project-top-10-for-large-language-model-applications/

Implementa proteções contra:
- LLM01: Prompt Injection
- LLM02: Insecure Output Handling
- LLM06: Sensitive Information Disclosure
"""

import logging
import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class InputGuardrail:
    """Valida e sanitiza input do usuário antes de enviar ao LLM."""

    # Padrões comuns de prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
        r"disregard\s+(all|previous|your)",
        r"new\s+instructions?:",
        r"override\s+(system|instructions)",
    ]

    # Padrões de tentativa de exfiltração
    EXFILTRATION_PATTERNS = [
        r"(curl|wget|fetch)\s+https?://",
        r"send\s+(to|data|info)\s+",
        r"upload\s+to\s+",
    ]

    def __init__(
        self,
        allowed_topics: list[str] | None = None,
        max_length: int = 4096,
    ):
        """Inicializa guardrail de input.

        Args:
            allowed_topics: Tópicos permitidos (se vazio, permite todos).
            max_length: Tamanho máximo do input.
        """
        self.allowed_topics = allowed_topics or []
        self.max_length = max_length
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._exfiltration_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.EXFILTRATION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.

        Args:
            user_input: Texto do usuário.

        Returns:
            Tupla (is_valid, reason).
        """
        # Check 1: Input vazio
        if not user_input or not user_input.strip():
            return False, "Input bloqueado: input vazio."

        # Check 2: Tamanho máximo (evitar context stuffing)
        if len(user_input) > self.max_length:
            return False, f"Input bloqueado: excede tamanho máximo ({self.max_length} chars)."

        # Check 3: Prompt injection detection
        for pattern in self._injection_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %s", user_input[:100])
                return False, "Input bloqueado: padrão suspeito detectado."

        # Check 4: Exfiltração de dados
        for pattern in self._exfiltration_patterns:
            if pattern.search(user_input):
                logger.warning("Tentativa de exfiltração detectada: %s", user_input[:100])
                return False, "Input bloqueado: padrão de exfiltração detectado."

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário."""

    def __init__(self, language: str = "pt"):
        """Inicializa guardrail de output.

        Args:
            language: Idioma para detecção de PII.
        """
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.language = language

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado.
        """
        results = self.analyzer.analyze(
            text=llm_output,
            language=self.language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
        )

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            anonymized = self.anonymizer.anonymize(
                text=llm_output,
                analyzer_results=results,
            )
            return anonymized.text

        return llm_output

    def validate_output(self, llm_output: str) -> tuple[bool, str]:
        """Valida se o output é seguro para retornar.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Tupla (is_safe, reason).
        """
        # Check: Output não deve conter instruções de sistema
        system_patterns = [
            r"<\|system\|>",
            r"\[SYSTEM\]",
            r"INTERNAL:",
        ]

        for pattern in system_patterns:
            if re.search(pattern, llm_output, re.IGNORECASE):
                logger.warning("Output contém padrão de sistema: %s", llm_output[:100])
                return False, "Output bloqueado: contém informação de sistema."

        return True, "OK"
