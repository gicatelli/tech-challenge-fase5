"""Testes do agente ReAct e tools."""

import pytest

langchain = pytest.importorskip("langchain", reason="langchain não instalado")

from src.agent.tools import (  # noqa: E402
    analisar_historico,
    buscar_conhecimento,
    calcular_risco,
    get_available_tools,
    prever_preco,
)


class TestTools:
    """Testes para as tools do agente."""

    def test_minimum_tools_count(self):
        """Deve ter pelo menos 3 tools (requisito do Datathon)."""
        tools = get_available_tools()
        assert len(tools) >= 3

    def test_tools_have_names(self):
        """Todas as tools devem ter nome."""
        tools = get_available_tools()
        for tool in tools:
            assert tool.name is not None
            assert len(tool.name) > 0

    def test_tools_have_descriptions(self):
        """Todas as tools devem ter descrição."""
        tools = get_available_tools()
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 10

    def test_analisar_historico_returns_json(self):
        """Tool de análise deve retornar JSON válido."""
        import json

        result = analisar_historico("últimos 30 dias")
        parsed = json.loads(result)
        assert "preco_atual" in parsed or "erro" in parsed

    def test_prever_preco_returns_json(self):
        """Tool de previsão deve retornar JSON válido."""
        import json

        result = prever_preco("próximos 5 dias")
        parsed = json.loads(result)
        assert "previsoes" in parsed or "erro" in parsed

    def test_calcular_risco_returns_json(self):
        """Tool de risco deve retornar JSON válido."""
        import json

        result = calcular_risco("último trimestre")
        parsed = json.loads(result)
        assert "metricas_risco" in parsed or "erro" in parsed

    def test_buscar_conhecimento_returns_json(self):
        """Tool de busca deve retornar JSON válido."""
        import json

        result = buscar_conhecimento("o que é RSI")
        parsed = json.loads(result)
        assert "informacoes" in parsed or "resultado" in parsed or "erro" in parsed
