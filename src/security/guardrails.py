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

logger = logging.getLogger(__name__)

# Import Presidio com fallback
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


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

    # Padrões de PII via regex (fallback quando Presidio indisponível)
    PII_PATTERNS = [
        (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "<CPF_REDACTED>"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "<EMAIL_REDACTED>"),
        (r"\b\d{2}\s?\d{4,5}-?\d{4}\b", "<PHONE_REDACTED>"),
        (r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "<CREDIT_CARD_REDACTED>"),
    ]

    def __init__(self, language: str = "pt"):
        """Inicializa guardrail de output.

        Args:
            language: Idioma para detecção de PII.

        """
        self.language = language
        self._use_presidio = False
        self._compiled_pii = [(re.compile(p), r) for p, r in self.PII_PATTERNS]

        if PRESIDIO_AVAILABLE:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                # Testar se tem recognizers funcionais
                test_results = self.analyzer.analyze(
                    text="test@email.com", language="en",
                    entities=["EMAIL_ADDRESS"],
                )
                if test_results:
                    self._use_presidio = True
                    logger.info("OutputGuardrail: usando Presidio")
            except Exception as e:
                logger.info("OutputGuardrail: Presidio indisponível (%s), usando regex", e)
        else:
            logger.info("OutputGuardrail: Presidio não instalado, usando regex")

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Usa Presidio se disponível, caso contrário aplica regex patterns.
        Sempre aplica regex como segunda camada para capturar cartões de crédito
        e outros padrões que o Presidio pode não detectar em textos PT-BR.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado.

        """
        sanitized = llm_output

        if self._use_presidio:
            try:
                results = self.analyzer.analyze(
                    text=sanitized,
                    language="en",  # Presidio funciona melhor em inglês
                    entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
                )
                if results:
                    logger.warning("PII detectado no output: %d entidades", len(results))
                    anonymized = self.anonymizer.anonymize(
                        text=sanitized,
                        analyzer_results=results,  # type: ignore[arg-type]
                    )
                    sanitized = anonymized.text
            except Exception as e:
                logger.warning("Presidio falhou (%s), usando regex fallback", e)

        # Segunda camada: regex-based PII detection (captura cartões e padrões BR)
        pii_found = False
        for pattern, replacement in self._compiled_pii:
            if pattern.search(sanitized):
                sanitized = pattern.sub(replacement, sanitized)
                pii_found = True

        if pii_found:
            logger.warning("PII detectado (regex) e removido do output")

        return sanitized

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
