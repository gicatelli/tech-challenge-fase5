"""RAG Pipeline — Embedding + Vector Store + Retriever + Generator.

Implementa o pipeline completo de Retrieval-Augmented Generation:
1. Ingestão de documentos
2. Chunking e embedding
3. Armazenamento em vector store (ChromaDB)
4. Retrieval por similaridade
5. Geração de resposta com contexto

Suporta dois modos de embedding:
- LOCAL (padrão): sentence-transformers (gratuito, offline)
- OPENAI: OpenAI text-embedding-3-small (requer API key com créditos)
"""

import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

# Configurações
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base")

# Modo de embedding: "local" (sentence-transformers) ou "openai"
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local").lower()


def get_embeddings():
    """Retorna o modelo de embeddings conforme configuração.

    Usa sentence-transformers local por padrão (gratuito, offline).
    Se EMBEDDING_MODE=openai e OPENAI_API_KEY válida, usa OpenAI.

    Returns:
        Instância de embeddings compatível com LangChain.

    """
    if EMBEDDING_MODE == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.info("Usando embeddings OpenAI: %s", EMBEDDING_MODEL)
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore[arg-type]
            )
        except Exception as e:
            logger.warning("OpenAI embeddings falhou (%s), usando local", e)

    # Default: embeddings locais gratuitos
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Usando embeddings locais: %s", LOCAL_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_documents(docs_path: str, collection_name: str = "datathon") -> Chroma:
    """Ingere documentos no vector store.

    Args:
        docs_path: Caminho para diretório com documentos.
        collection_name: Nome da coleção no ChromaDB.

    Returns:
        Vector store populado.

    """
    logger.info("Ingerindo documentos de: %s", docs_path)

    # Carregar documentos
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    logger.info("Documentos carregados: %d", len(documents))

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Chunks gerados: %d", len(chunks))

    # Embedding + Vector Store
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name,
    )

    logger.info("Vector store criado com %d chunks em %s", len(chunks), CHROMA_PERSIST_DIR)
    return vectorstore


def get_vectorstore(collection_name: str = "datathon") -> Chroma:
    """Carrega vector store existente.

    Args:
        collection_name: Nome da coleção.

    Returns:
        Vector store carregado.

    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectorstore


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    """Recupera contextos relevantes para uma query.

    Args:
        query: Pergunta do usuário.
        top_k: Número de documentos a retornar.

    Returns:
        Lista de textos relevantes.

    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)

    contexts = [doc.page_content for doc in results]
    logger.info("Retrieval: %d contextos para query '%s'", len(contexts), query[:50])

    return contexts


def generate_answer(query: str, contexts: list[str]) -> str:
    """Gera resposta usando LLM com contexto do RAG.

    Se OpenAI não estiver disponível, retorna resposta baseada nos contextos.

    Args:
        query: Pergunta do usuário.
        contexts: Contextos recuperados do vector store.

    Returns:
        Resposta gerada pelo LLM ou baseada em contexto.

    """
    context_text = "\n\n".join([f"Contexto {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

    # Tentar usar OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-proj-PLACEHOLDER"):
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
                temperature=0.0,
                api_key=api_key,  # type: ignore[arg-type]
            )

            prompt = f"""Com base nos contextos fornecidos, responda a pergunta do usuário.
Se a informação não estiver nos contextos, diga claramente que não encontrou.

{context_text}

Pergunta: {query}

Resposta:"""

            response = llm.invoke(prompt)
            return response.content  # type: ignore[return-value]
        except Exception as e:
            logger.warning("OpenAI LLM falhou (%s), usando resposta baseada em contexto", e)

    # Fallback: resposta baseada em contexto (sem LLM)
    if contexts:
        return (
            f"Com base nos documentos disponíveis sobre o tema:\n\n"
            f"{contexts[0]}\n\n"
            f"{'Informação adicional: ' + contexts[1] if len(contexts) > 1 else ''}"
        )
    return "Não foi possível encontrar informações relevantes para sua pergunta."


def rag_query(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Pipeline RAG completo: retrieve + generate.

    Args:
        query: Pergunta do usuário.
        top_k: Número de contextos a recuperar.

    Returns:
        Tupla (resposta, contextos utilizados).

    """
    from src.monitoring.telemetry import trace_query

    with trace_query(query, method="rag") as trace:
        contexts = retrieve_context(query, top_k=top_k)
        trace.set_contexts(len(contexts))

        answer = generate_answer(query, contexts)
        trace.set_output(answer)

    logger.info("RAG query completa: %d contextos usados", len(contexts))
    return answer, contexts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import sys

    print("=" * 60)
    print("  RAG PIPELINE — DATATHON FASE 05")
    print("=" * 60)

    action = sys.argv[1] if len(sys.argv) > 1 else "ingest"

    if action == "ingest":
        print(f"\n[1] Ingerindo documentos de: {KNOWLEDGE_BASE_DIR}")
        vectorstore = ingest_documents(KNOWLEDGE_BASE_DIR)
        print("    Documentos ingeridos com sucesso!")
        print(f"    Persist dir: {CHROMA_PERSIST_DIR}")

        # Teste rápido de retrieval
        print("\n[2] Testando retrieval...")
        test_queries = [
            "O que é RSI?",
            "Quais são os riscos da Petrobras?",
            "Como calcular o VaR?",
            "O que é o pré-sal?",
            "Qual a política de dividendos da Petrobras?",
        ]

        for query in test_queries:
            contexts = retrieve_context(query, top_k=2)
            print(f"\n  Q: {query}")
            print(f"  A: {contexts[0][:120]}..." if contexts else "  A: Nenhum contexto encontrado")

    elif action == "query":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "O que é RSI?"
        print(f"\nQuery: {query}")
        answer, contexts = rag_query(query)
        print(f"\nResposta: {answer}")
        print(f"\nContextos utilizados: {len(contexts)}")
        for i, ctx in enumerate(contexts):
            print(f"  [{i+1}] {ctx[:100]}...")

    else:
        print("Uso:")
        print("  python -m src.agent.rag_pipeline ingest   # Ingerir documentos")
        print("  python -m src.agent.rag_pipeline query <pergunta>  # Consultar RAG")
