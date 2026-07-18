"""Testes do agente ReAct e tools."""


from src.agent.tools import (
    analyze_data,
    calculate_risk_score,
    get_available_tools,
    get_model_prediction,
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

    def test_analyze_data_returns_string(self):
        """Tool de análise deve retornar string."""
        result = analyze_data("Qual a média de transações?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_model_prediction_valid_json(self):
        """Tool de predição deve aceitar JSON válido."""
        import json

        result = get_model_prediction('{"feature_1": 0.5, "feature_2": 3.0}')
        parsed = json.loads(result)
        assert "prediction" in parsed

    def test_get_model_prediction_invalid_json(self):
        """Tool de predição deve tratar JSON inválido."""
        result = get_model_prediction("not a json")
        assert "Erro" in result

    def test_calculate_risk_score_returns_string(self):
        """Tool de risco deve retornar string."""
        result = calculate_risk_score("Transação de alto valor")
        assert isinstance(result, str)
        assert len(result) > 0
