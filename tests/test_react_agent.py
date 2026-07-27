"""Testes para src/agent/react_agent.py — agente ReAct."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.react_agent import REACT_PROMPT, create_datathon_agent, run_agent


class TestReactPrompt:
    """Testes para o template do prompt ReAct."""

    def test_prompt_has_required_variables(self):
        """Prompt deve ter as variáveis necessárias."""
        assert "tools" in REACT_PROMPT.input_variables or "tools" in REACT_PROMPT.template
        assert "input" in REACT_PROMPT.template
        assert "agent_scratchpad" in REACT_PROMPT.template

    def test_prompt_has_format_instructions(self):
        """Prompt deve conter instruções de formato ReAct."""
        assert "Thought:" in REACT_PROMPT.template
        assert "Action:" in REACT_PROMPT.template
        assert "Action Input:" in REACT_PROMPT.template
        assert "Observation:" in REACT_PROMPT.template
        assert "Final Answer:" in REACT_PROMPT.template

    def test_prompt_has_domain_context(self):
        """Prompt deve ter contexto do domínio financeiro."""
        assert "financeiro" in REACT_PROMPT.template.lower()

    def test_prompt_instructs_not_to_invent(self):
        """Prompt deve instruir a não inventar dados."""
        template_lower = REACT_PROMPT.template.lower()
        assert "nunca invente" in template_lower or "não invente" in template_lower


class TestCreateDatathonAgent:
    """Testes para create_datathon_agent."""

    @patch("src.agent.react_agent.AgentExecutor")
    @patch("src.agent.react_agent.create_react_agent")
    @patch("src.agent.react_agent.get_available_tools")
    def test_creates_agent_with_default_tools(self, mock_tools, mock_create, mock_executor):
        """Deve criar agente com tools padrão se nenhuma fornecida."""
        mock_tool_1 = MagicMock()
        mock_tool_1.name = "tool1"
        mock_tool_2 = MagicMock()
        mock_tool_2.name = "tool2"
        mock_tool_3 = MagicMock()
        mock_tool_3.name = "tool3"
        mock_tools.return_value = [mock_tool_1, mock_tool_2, mock_tool_3]
        mock_create.return_value = MagicMock()
        mock_executor.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {"USE_OLLAMA": "false", "OPENAI_API_KEY": "sk-test-key-123"},
        ):
            agent = create_datathon_agent()

        assert agent is not None
        mock_tools.assert_called_once()

    @patch("src.agent.react_agent.AgentExecutor")
    @patch("src.agent.react_agent.create_react_agent")
    def test_warns_if_less_than_3_tools(self, mock_create, mock_executor):
        """Deve emitir warning se menos de 3 tools."""
        mock_tool = MagicMock()
        mock_tool.name = "only_tool"
        mock_create.return_value = MagicMock()
        mock_executor.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {"USE_OLLAMA": "false", "OPENAI_API_KEY": "sk-test-key-123"},
        ):
            import logging

            with patch.object(logging.getLogger("src.agent.react_agent"), "warning") as mock_warn:
                create_datathon_agent(tools=[mock_tool])
                mock_warn.assert_called_once()

    @patch("src.agent.react_agent.create_react_agent")
    def test_raises_without_any_llm(self, mock_create):
        """Deve levantar ValueError sem LLM configurado."""
        mock_tool = MagicMock()
        mock_tool.name = "t1"
        tools = [mock_tool, mock_tool, mock_tool]

        with patch.dict(
            "os.environ",
            {
                "USE_OLLAMA": "false",
                "OPENAI_API_KEY": "",
                "GOOGLE_API_KEY": "",
            },
            clear=False,
        ):
            # Remover as chaves se existirem
            import os
            env_backup = {}
            for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY"]:
                env_backup[key] = os.environ.pop(key, None)

            try:
                with pytest.raises(ValueError, match="Nenhum LLM"):
                    create_datathon_agent(tools=tools)
            finally:
                # Restaurar
                for key, val in env_backup.items():
                    if val is not None:
                        os.environ[key] = val

    @patch("src.agent.react_agent.AgentExecutor")
    @patch("src.agent.react_agent.create_react_agent")
    def test_custom_max_iterations(self, mock_create, mock_executor):
        """Deve respeitar max_iterations configurado."""
        mock_tool = MagicMock()
        mock_tool.name = "t1"
        tools = [mock_tool, mock_tool, mock_tool]
        mock_create.return_value = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance

        with patch.dict(
            "os.environ",
            {"USE_OLLAMA": "false", "OPENAI_API_KEY": "sk-test-key-123"},
        ):
            create_datathon_agent(tools=tools, max_iterations=5)

        # Verificar que AgentExecutor foi chamado com max_iterations=5
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["max_iterations"] == 5


class TestRunAgent:
    """Testes para run_agent."""

    @patch("src.monitoring.telemetry.trace_query")
    @patch("src.agent.react_agent.create_datathon_agent")
    def test_returns_answer_dict(self, mock_create_agent, mock_trace):
        """Deve retornar dicionário com answer, steps e tools_used."""
        # Mock tracer
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        # Mock agent
        mock_agent = MagicMock()
        mock_step = MagicMock()
        mock_step.tool = "calcular_risco"
        mock_agent.invoke.return_value = {
            "output": "O risco é moderado.",
            "intermediate_steps": [(mock_step, "resultado")],
        }
        mock_create_agent.return_value = mock_agent

        result = run_agent("Qual o risco da PETR4?")

        assert "answer" in result
        assert "steps" in result
        assert "tools_used" in result
        assert result["answer"] == "O risco é moderado."
        assert result["steps"] == 1
        assert "calcular_risco" in result["tools_used"]

    @patch("src.monitoring.telemetry.trace_query")
    @patch("src.agent.react_agent.get_available_tools")
    @patch("src.agent.react_agent.create_datathon_agent")
    def test_fallback_on_agent_failure(self, mock_create, mock_tools, mock_trace):
        """Deve usar fallback quando agente falha."""
        # Mock tracer
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        # Agente falha
        mock_create.side_effect = Exception("LLM indisponível")

        # Tools retornam resultado
        mock_tool = MagicMock()
        mock_tool.name = "prever_preco"
        mock_tool.run.return_value = "Previsão: R$42.00"
        mock_tools.return_value = [mock_tool]

        result = run_agent("Qual o preço?")

        assert "answer" in result
        assert "prever_preco" in result["tools_used"]

    @patch("src.monitoring.telemetry.trace_query")
    @patch("src.agent.react_agent.get_available_tools")
    @patch("src.agent.react_agent.create_datathon_agent")
    def test_fallback_message_when_all_fails(self, mock_create, mock_tools, mock_trace):
        """Deve retornar mensagem padrão se tudo falhar."""
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        mock_create.side_effect = Exception("LLM offline")

        # Tools também falham
        mock_tool = MagicMock()
        mock_tool.name = "t1"
        mock_tool.run.side_effect = Exception("tool error")
        mock_tools.return_value = [mock_tool]

        result = run_agent("Pergunta qualquer")

        assert (
            "indisponível" in result["answer"].lower()
            or "não foi possível" in result["answer"].lower()
        )

    @patch("src.monitoring.telemetry.trace_query")
    def test_uses_provided_agent(self, mock_trace):
        """Deve usar agente fornecido se passado."""
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "Resposta personalizada",
            "intermediate_steps": [],
        }

        result = run_agent("test query", agent=mock_agent)

        assert result["answer"] == "Resposta personalizada"
        mock_agent.invoke.assert_called_once_with({"input": "test query"})
