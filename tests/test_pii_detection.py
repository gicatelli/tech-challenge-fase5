"""Testes para src/security/pii_detection.py — detecção e anonimização de PII."""

import pytest

pytest.importorskip("presidio_analyzer", reason="presidio não instalado")

from src.security.pii_detection import (  # noqa: E402
    anonymize_text,
    create_analyzer,
    create_br_recognizers,
    detect_pii,
)


class TestCreateBrRecognizers:
    """Testes para create_br_recognizers."""

    def test_returns_list(self):
        """Deve retornar uma lista de recognizers."""
        recognizers = create_br_recognizers()
        assert isinstance(recognizers, list)

    def test_returns_three_recognizers(self):
        """Deve retornar exatamente 3 recognizers (CPF, CNPJ, Phone)."""
        recognizers = create_br_recognizers()
        assert len(recognizers) == 3

    def test_cpf_recognizer_entity(self):
        """Primeiro recognizer deve ser BR_CPF."""
        recognizers = create_br_recognizers()
        assert recognizers[0].supported_entities == ["BR_CPF"]

    def test_cnpj_recognizer_entity(self):
        """Segundo recognizer deve ser BR_CNPJ."""
        recognizers = create_br_recognizers()
        assert recognizers[1].supported_entities == ["BR_CNPJ"]

    def test_phone_recognizer_entity(self):
        """Terceiro recognizer deve ser BR_PHONE."""
        recognizers = create_br_recognizers()
        assert recognizers[2].supported_entities == ["BR_PHONE"]


class TestCreateAnalyzer:
    """Testes para create_analyzer."""

    def test_returns_analyzer_engine(self):
        """Deve retornar instância de AnalyzerEngine."""
        from presidio_analyzer import AnalyzerEngine

        analyzer = create_analyzer()
        assert isinstance(analyzer, AnalyzerEngine)

    def test_has_br_recognizers(self):
        """Analyzer deve conter recognizers brasileiros."""
        analyzer = create_analyzer()
        # Verificar que consegue analisar com entidades BR
        results = analyzer.analyze(
            text="Meu CPF é 123.456.789-00",
            language="pt",
            entities=["BR_CPF"],
        )
        assert len(results) > 0


class TestDetectPii:
    """Testes para detect_pii."""

    def test_detects_cpf_formatted(self):
        """Deve detectar CPF formatado (XXX.XXX.XXX-XX)."""
        results = detect_pii("Meu CPF é 123.456.789-00")
        cpf_entities = [r for r in results if r["entity_type"] == "BR_CPF"]
        assert len(cpf_entities) > 0

    def test_detects_cpf_unformatted(self):
        """Deve detectar CPF sem formatação (XXXXXXXXXXX)."""
        results = detect_pii("CPF: 12345678900")
        cpf_entities = [r for r in results if r["entity_type"] == "BR_CPF"]
        assert len(cpf_entities) > 0

    def test_detects_email(self):
        """Deve detectar endereço de email."""
        results = detect_pii("Email: joao.silva@empresa.com.br")
        email_entities = [r for r in results if r["entity_type"] == "EMAIL_ADDRESS"]
        assert len(email_entities) > 0

    def test_detects_phone_br(self):
        """Deve detectar telefone brasileiro."""
        results = detect_pii("Ligue para (11) 99999-8888")
        phone_entities = [r for r in results if r["entity_type"] in ("PHONE_NUMBER", "BR_PHONE")]
        assert len(phone_entities) > 0

    def test_no_pii_in_clean_text(self):
        """Texto sem PII deve retornar lista vazia."""
        results = detect_pii("A Petrobras é uma empresa de petróleo.")
        # Pode ter falsos positivos com PERSON, filtrar apenas entidades BR
        br_entities = [r for r in results if r["entity_type"].startswith("BR_")]
        assert len(br_entities) == 0

    def test_returns_correct_structure(self):
        """Cada resultado deve ter entity_type, start, end, score, text."""
        results = detect_pii("Email: test@email.com CPF: 111.222.333-44")
        assert len(results) > 0
        for result in results:
            assert "entity_type" in result
            assert "start" in result
            assert "end" in result
            assert "score" in result
            assert "text" in result

    def test_detects_multiple_pii(self):
        """Deve detectar múltiplas entidades PII no mesmo texto."""
        text = "João Silva, CPF 123.456.789-00, email joao@teste.com, tel (11) 98765-4321"
        results = detect_pii(text)
        assert len(results) >= 2  # Pelo menos CPF + email

    def test_detects_cnpj(self):
        """Deve detectar CNPJ."""
        results = detect_pii("CNPJ: 12.345.678/0001-90")
        cnpj_entities = [r for r in results if r["entity_type"] == "BR_CNPJ"]
        assert len(cnpj_entities) > 0

    def test_score_is_positive(self):
        """Score de detecção deve ser positivo."""
        results = detect_pii("CPF: 111.222.333-44")
        for result in results:
            assert result["score"] > 0

    def test_text_field_matches_span(self):
        """Campo text deve corresponder à substring start:end."""
        text = "Meu CPF é 123.456.789-00 ok"
        results = detect_pii(text)
        cpf_results = [r for r in results if r["entity_type"] == "BR_CPF"]
        if cpf_results:
            r = cpf_results[0]
            assert r["text"] == text[r["start"]:r["end"]]


class TestAnonymizeText:
    """Testes para anonymize_text."""

    def test_anonymizes_cpf(self):
        """CPF deve ser substituído por placeholder."""
        text = "CPF do cliente: 123.456.789-00"
        result = anonymize_text(text)
        assert "123.456.789-00" not in result
        assert "<CPF>" in result

    def test_anonymizes_email(self):
        """Email deve ser substituído por placeholder."""
        text = "Enviar para joao@empresa.com"
        result = anonymize_text(text)
        assert "joao@empresa.com" not in result
        assert "<EMAIL>" in result

    def test_clean_text_unchanged(self):
        """Texto sem PII deve permanecer inalterado."""
        text = "A PETR4 fechou em alta hoje."
        result = anonymize_text(text)
        assert result == text

    def test_anonymizes_phone(self):
        """Telefone deve ser substituído por placeholder."""
        text = "Ligue para (11) 98765-4321 para mais informações"
        result = anonymize_text(text)
        # Deve ter removido o telefone
        assert "(11) 98765-4321" not in result

    def test_anonymizes_cnpj(self):
        """CNPJ deve ser substituído por placeholder."""
        text = "CNPJ da empresa: 12.345.678/0001-90"
        result = anonymize_text(text)
        assert "12.345.678/0001-90" not in result
        assert "<CNPJ>" in result

    def test_anonymizes_multiple_entities(self):
        """Múltiplas entidades devem ser anonimizadas."""
        text = "João, CPF 111.222.333-44, email joao@test.com"
        result = anonymize_text(text)
        assert "111.222.333-44" not in result
        assert "joao@test.com" not in result

    def test_returns_string(self):
        """Resultado deve ser sempre uma string."""
        result = anonymize_text("Texto qualquer com CPF 999.888.777-66")
        assert isinstance(result, str)

    def test_preserves_non_pii_content(self):
        """Conteúdo sem PII deve ser preservado na saída."""
        text = "O analista recomenda compra. CPF: 111.222.333-44. Fim."
        result = anonymize_text(text)
        assert "O analista recomenda compra." in result
        assert "Fim." in result
