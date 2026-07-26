"""Testes para src/monitoring/telemetry.py — tracing e observabilidade."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.telemetry import (
    QueryTracer,
    TraceRecord,
    _count_methods,
    _count_tools,
    _log_to_file,
    get_trace_summary,
    trace_query,
)


class TestTraceRecord:
    """Testes para TraceRecord dataclass."""

    def test_default_values(self):
        """Deve ter valores default corretos."""
        record = TraceRecord(query="test query")
        assert record.query == "test query"
        assert record.output == ""
        assert record.latency_ms == 0.0
        assert record.tokens_input == 0
        assert record.tokens_output == 0
        assert record.tools_used == []
        assert record.contexts_retrieved == 0
        assert record.method == "unknown"
        assert record.success is True
        assert record.error == ""

    def test_to_dict(self):
        """to_dict deve retornar dicionário com todas as chaves."""
        record = TraceRecord(
            query="Qual o preço da PETR4?",
            output="O preço atual é R$ 42.00",
            latency_ms=150.5,
            tokens_input=10,
            tokens_output=15,
            tools_used=["prever_preco"],
            contexts_retrieved=3,
            method="agent",
            success=True,
            timestamp="2026-07-26T10:00:00",
        )
        d = record.to_dict()
        assert d["query"] == "Qual o preço da PETR4?"
        assert d["latency_ms"] == 150.5
        assert d["method"] == "agent"
        assert d["tools_used"] == ["prever_preco"]
        assert d["contexts_retrieved"] == 3

    def test_to_dict_truncates_output(self):
        """to_dict deve truncar output em 200 chars."""
        long_output = "x" * 500
        record = TraceRecord(query="q", output=long_output)
        d = record.to_dict()
        assert len(d["output"]) == 200


class TestQueryTracer:
    """Testes para QueryTracer context manager."""

    def test_measures_latency(self, tmp_path, monkeypatch):
        """Deve medir latência corretamente."""
        # Redirecionar output para tmp
        monkeypatch.setattr(
            "src.monitoring.telemetry.TRACES_DIR", tmp_path
        )

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                tracer = QueryTracer("test query", method="test")
                with tracer:
                    time.sleep(0.05)  # 50ms

                assert tracer.record.latency_ms >= 40  # tolerância

    def test_sets_output(self, tmp_path, monkeypatch):
        """set_output deve definir output no record."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                with QueryTracer("q", "agent") as tracer:
                    tracer.set_output("resposta do agente")

                assert tracer.record.output == "resposta do agente"

    def test_sets_tools(self, tmp_path, monkeypatch):
        """set_tools deve definir tools usadas."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                with QueryTracer("q", "agent") as tracer:
                    tracer.set_tools(["calcular_risco", "prever_preco"])

                assert tracer.record.tools_used == ["calcular_risco", "prever_preco"]

    def test_sets_contexts(self, tmp_path, monkeypatch):
        """set_contexts deve definir contagem de contextos."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                with QueryTracer("q", "rag") as tracer:
                    tracer.set_contexts(5)

                assert tracer.record.contexts_retrieved == 5

    def test_sets_tokens(self, tmp_path, monkeypatch):
        """set_tokens deve definir contagem de tokens."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                with QueryTracer("q", "agent") as tracer:
                    tracer.set_tokens(100, 200)

                assert tracer.record.tokens_input == 100
                assert tracer.record.tokens_output == 200

    def test_handles_exception(self, tmp_path, monkeypatch):
        """Deve registrar erro quando exceção ocorre."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                tracer = QueryTracer("q", "agent")
                with pytest.raises(ValueError):
                    with tracer:
                        raise ValueError("test error")

                assert tracer.record.success is False
                assert "test error" in tracer.record.error

    def test_estimates_tokens_from_text(self, tmp_path, monkeypatch):
        """Deve estimar tokens se não forem definidos explicitamente."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        with patch("src.monitoring.telemetry._log_to_mlflow"):
            with patch("src.monitoring.telemetry._log_to_langfuse"):
                with QueryTracer("uma pergunta de teste", "agent") as tracer:
                    tracer.set_output("uma resposta longa com varias palavras aqui")

                # Tokens estimados = len(text) // 4
                assert tracer.record.tokens_input == len("uma pergunta de teste") // 4
                assert tracer.record.tokens_output == len("uma resposta longa com varias palavras aqui") // 4


class TestTraceQuery:
    """Testes para a função trace_query."""

    def test_returns_query_tracer(self):
        """trace_query deve retornar instância de QueryTracer."""
        tracer = trace_query("test", method="rag")
        assert isinstance(tracer, QueryTracer)
        assert tracer.record.query == "test"
        assert tracer.record.method == "rag"


class TestLogToFile:
    """Testes para _log_to_file."""

    def test_appends_to_jsonl(self, tmp_path, monkeypatch):
        """Deve criar/append em arquivo JSONL."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        trace_dict = {
            "query": "test",
            "output": "resp",
            "latency_ms": 100.0,
            "tokens_input": 5,
            "tokens_output": 10,
            "tools_used": [],
            "contexts_retrieved": 0,
            "method": "test",
            "success": True,
            "error": "",
            "timestamp": "2026-07-26T10:00:00",
        }

        _log_to_file(trace_dict)
        _log_to_file(trace_dict)

        traces_file = tmp_path / "query_traces.jsonl"
        assert traces_file.exists()

        lines = traces_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        parsed = json.loads(lines[0])
        assert parsed["query"] == "test"


class TestGetTraceSummary:
    """Testes para get_trace_summary."""

    def test_empty_when_no_traces(self, tmp_path, monkeypatch):
        """Deve retornar total_queries=0 sem arquivo."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)
        summary = get_trace_summary()
        assert summary["total_queries"] == 0

    def test_summarizes_traces(self, tmp_path, monkeypatch):
        """Deve sumarizar traces existentes."""
        monkeypatch.setattr("src.monitoring.telemetry.TRACES_DIR", tmp_path)

        traces_file = tmp_path / "query_traces.jsonl"
        traces = [
            {"query": "q1", "latency_ms": 100, "success": True, "tools_used": ["t1"], "method": "agent"},
            {"query": "q2", "latency_ms": 200, "success": True, "tools_used": ["t2"], "method": "rag"},
            {"query": "q3", "latency_ms": 150, "success": False, "tools_used": [], "method": "agent"},
        ]
        traces_file.write_text(
            "\n".join(json.dumps(t) for t in traces), encoding="utf-8"
        )

        summary = get_trace_summary()
        assert summary["total_queries"] == 3
        assert summary["avg_latency_ms"] == 150.0
        assert summary["success_rate"] == pytest.approx(0.67, abs=0.01)


class TestCountHelpers:
    """Testes para funções auxiliares de contagem."""

    def test_count_tools(self):
        """Deve contar uso de cada tool."""
        traces = [
            {"tools_used": ["t1", "t2"]},
            {"tools_used": ["t1"]},
            {"tools_used": ["t3"]},
        ]
        counts = _count_tools(traces)
        assert counts["t1"] == 2
        assert counts["t2"] == 1
        assert counts["t3"] == 1

    def test_count_tools_empty(self):
        """Deve retornar vazio sem tools."""
        traces = [{"tools_used": []}, {"tools_used": []}]
        counts = _count_tools(traces)
        assert counts == {}

    def test_count_methods(self):
        """Deve contar uso de cada método."""
        traces = [
            {"method": "agent"},
            {"method": "rag"},
            {"method": "agent"},
        ]
        counts = _count_methods(traces)
        assert counts["agent"] == 2
        assert counts["rag"] == 1

    def test_count_methods_unknown_default(self):
        """Deve usar 'unknown' como default."""
        traces = [{}]
        counts = _count_methods(traces)
        assert counts["unknown"] == 1
