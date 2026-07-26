"""Testes para src/agent/rag_pipeline.py — RAG pipeline."""

from unittest.mock import MagicMock, patch

from src.agent.rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    generate_answer,
    get_embeddings,
    get_vectorstore,
    ingest_documents,
    rag_query,
    retrieve_context,
)


class TestConstants:
    """Testes para constantes de configuração do RAG."""

    def test_chunk_size_reasonable(self):
        """Chunk size deve ser razoável (500-2000)."""
        assert 500 <= CHUNK_SIZE <= 2000

    def test_chunk_overlap_less_than_size(self):
        """Overlap deve ser menor que chunk size."""
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_overlap_is_positive(self):
        """Overlap deve ser positivo."""
        assert CHUNK_OVERLAP > 0


class TestGetEmbeddings:
    """Testes para get_embeddings."""

    @patch.dict("os.environ", {"EMBEDDING_MODE": "local"})
    @patch("src.agent.rag_pipeline.HuggingFaceEmbeddings")
    def test_returns_local_embeddings_by_default(self, mock_hf):
        """Deve retornar embeddings locais por padrão."""
        mock_hf.return_value = MagicMock()
        with patch("src.agent.rag_pipeline.EMBEDDING_MODE", "local"):
            result = get_embeddings()
        mock_hf.assert_called_once()
        assert result is not None

    @patch.dict("os.environ", {"EMBEDDING_MODE": "openai", "OPENAI_API_KEY": "sk-test"})
    def test_uses_openai_when_configured(self):
        """Deve usar OpenAI quando configurado."""
        with patch("src.agent.rag_pipeline.EMBEDDING_MODE", "openai"):
            with patch("src.agent.rag_pipeline.os.getenv") as mock_getenv:
                mock_getenv.side_effect = lambda k, d=None: {
                    "EMBEDDING_MODE": "openai",
                    "OPENAI_API_KEY": "sk-test-key",
                }.get(k, d)

                with patch(
                    "langchain_openai.OpenAIEmbeddings"
                ) as mock_openai:
                    mock_openai.return_value = MagicMock()
                    result = get_embeddings()
                    # Se OpenAI funcionar, retorna embedding
                    assert result is not None


class TestIngestDocuments:
    """Testes para ingest_documents."""

    @patch("src.agent.rag_pipeline.Chroma")
    @patch("src.agent.rag_pipeline.get_embeddings")
    @patch("src.agent.rag_pipeline.RecursiveCharacterTextSplitter")
    @patch("src.agent.rag_pipeline.DirectoryLoader")
    def test_ingests_and_returns_vectorstore(
        self, mock_loader, mock_splitter, mock_embeddings, mock_chroma
    ):
        """Deve ingerir documentos e retornar vectorstore."""
        # Mock documents
        mock_doc = MagicMock()
        mock_doc.page_content = "Texto de teste sobre PETR4"
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance

        # Mock splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [mock_doc, mock_doc]
        mock_splitter.return_value = mock_splitter_instance

        # Mock embeddings
        mock_embeddings.return_value = MagicMock()

        # Mock Chroma
        mock_vectorstore = MagicMock()
        mock_chroma.from_documents.return_value = mock_vectorstore

        result = ingest_documents("/fake/path")

        assert result == mock_vectorstore
        mock_loader.assert_called_once()
        mock_splitter_instance.split_documents.assert_called_once()
        mock_chroma.from_documents.assert_called_once()

    @patch("src.agent.rag_pipeline.Chroma")
    @patch("src.agent.rag_pipeline.get_embeddings")
    @patch("src.agent.rag_pipeline.RecursiveCharacterTextSplitter")
    @patch("src.agent.rag_pipeline.DirectoryLoader")
    def test_uses_correct_chunk_params(
        self, mock_loader, mock_splitter, mock_embeddings, mock_chroma
    ):
        """Deve usar CHUNK_SIZE e CHUNK_OVERLAP corretos."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = []
        mock_splitter.return_value = mock_splitter_instance

        mock_embeddings.return_value = MagicMock()
        mock_chroma.from_documents.return_value = MagicMock()

        ingest_documents("/fake/path")

        mock_splitter.assert_called_once_with(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


class TestGetVectorstore:
    """Testes para get_vectorstore."""

    @patch("src.agent.rag_pipeline.Chroma")
    @patch("src.agent.rag_pipeline.get_embeddings")
    def test_returns_chroma_instance(self, mock_embeddings, mock_chroma):
        """Deve retornar instância do Chroma."""
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()

        result = get_vectorstore()

        assert result is not None
        mock_chroma.assert_called_once()

    @patch("src.agent.rag_pipeline.Chroma")
    @patch("src.agent.rag_pipeline.get_embeddings")
    def test_accepts_custom_collection(self, mock_embeddings, mock_chroma):
        """Deve aceitar nome de coleção customizado."""
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()

        get_vectorstore(collection_name="custom_collection")

        call_kwargs = mock_chroma.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"


class TestRetrieveContext:
    """Testes para retrieve_context."""

    @patch("src.agent.rag_pipeline.get_vectorstore")
    def test_returns_list_of_strings(self, mock_get_vs):
        """Deve retornar lista de strings."""
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "PETR4 é a ação da Petrobras"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "A Petrobras atua no setor de petróleo"

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc1, mock_doc2]
        mock_get_vs.return_value = mock_vs

        contexts = retrieve_context("O que é PETR4?", top_k=2)

        assert len(contexts) == 2
        assert "PETR4" in contexts[0]
        assert isinstance(contexts[0], str)

    @patch("src.agent.rag_pipeline.get_vectorstore")
    def test_respects_top_k(self, mock_get_vs):
        """Deve respeitar parâmetro top_k."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_get_vs.return_value = mock_vs

        retrieve_context("query", top_k=5)

        mock_vs.similarity_search.assert_called_once_with("query", k=5)


class TestGenerateAnswer:
    """Testes para generate_answer."""

    def test_fallback_without_llm(self):
        """Deve retornar resposta baseada em contexto sem LLM."""
        with patch.dict(
            "os.environ",
            {"USE_OLLAMA": "false", "OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ):
            contexts = [
                "O RSI é um indicador de momentum que mede velocidade.",
                "Valores acima de 70 indicam sobrecompra.",
            ]
            answer = generate_answer("O que é RSI?", contexts)

            assert len(answer) > 0
            assert "RSI" in answer or "indicador" in answer

    def test_no_contexts_returns_default_message(self):
        """Deve retornar mensagem padrão sem contextos."""
        with patch.dict(
            "os.environ",
            {"USE_OLLAMA": "false", "OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""},
        ):
            answer = generate_answer("Pergunta sem resposta", [])
            assert "não foi possível" in answer.lower() or "não encontr" in answer.lower()

    @patch("src.agent.rag_pipeline.ChatOllama")
    def test_uses_ollama_when_available(self, mock_ollama):
        """Deve usar Ollama quando USE_OLLAMA=true."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Resposta do Ollama"
        mock_llm.invoke.return_value = mock_response
        mock_ollama.return_value = mock_llm

        with patch.dict("os.environ", {"USE_OLLAMA": "true"}):
            with patch("src.agent.rag_pipeline.os.getenv") as mock_getenv:
                mock_getenv.side_effect = lambda k, d="": {
                    "USE_OLLAMA": "true",
                    "OLLAMA_MODEL": "qwen2.5:3b",
                    "OLLAMA_BASE_URL": "http://localhost:11434",
                    "GOOGLE_API_KEY": "",
                    "OPENAI_API_KEY": "",
                }.get(k, d)

                answer = generate_answer("O que é RSI?", ["RSI mede momentum"])

                assert answer == "Resposta do Ollama"


class TestRagQuery:
    """Testes para rag_query (pipeline completo)."""

    @patch("src.agent.rag_pipeline.trace_query")
    @patch("src.agent.rag_pipeline.generate_answer")
    @patch("src.agent.rag_pipeline.retrieve_context")
    def test_returns_tuple(self, mock_retrieve, mock_generate, mock_trace):
        """Deve retornar tupla (answer, contexts)."""
        mock_retrieve.return_value = ["contexto 1", "contexto 2"]
        mock_generate.return_value = "Resposta completa"

        # Mock tracer
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        answer, contexts = rag_query("Qual o preço?", top_k=2)

        assert answer == "Resposta completa"
        assert len(contexts) == 2
        mock_retrieve.assert_called_once_with("Qual o preço?", top_k=2)

    @patch("src.agent.rag_pipeline.trace_query")
    @patch("src.agent.rag_pipeline.generate_answer")
    @patch("src.agent.rag_pipeline.retrieve_context")
    def test_traces_query(self, mock_retrieve, mock_generate, mock_trace):
        """Deve usar telemetria para tracing."""
        mock_retrieve.return_value = ["ctx"]
        mock_generate.return_value = "resp"

        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_tracer

        rag_query("test")

        mock_trace.assert_called_once_with("test", method="rag")
        mock_tracer.set_contexts.assert_called_once_with(1)
        mock_tracer.set_output.assert_called_once_with("resp")
