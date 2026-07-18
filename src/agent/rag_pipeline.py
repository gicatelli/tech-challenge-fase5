"""RAG Pipeline — Embedding + Vector Store + Retriever + Generator.

Implementa o pipeline completo de Retrieval-Augmented Generation:
1. Ingestão de documentos
2. Chunking e embedding
3. Armazenamento em vector store (ChromaDB)
4. Retrieval por similaridade
5. Geração de resposta com contexto
"""

import logging
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# Configurações
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")


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
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

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
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

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

    Args:
        query: Pergunta do usuário.
        contexts: Contextos recuperados do vector store.

    Returns:
        Resposta gerada pelo LLM.

    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    context_text = "\n\n".join([f"Contexto {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""Com base nos contextos fornecidos, responda a pergunta do usuário.
Se a informação não estiver nos contextos, diga claramente que não encontrou.

{context_text}

Pergunta: {query}

Resposta:"""

    response = llm.invoke(prompt)
    return response.content


def rag_query(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Pipeline RAG completo: retrieve + generate.

    Args:
        query: Pergunta do usuário.
        top_k: Número de contextos a recuperar.

    Returns:
        Tupla (resposta, contextos utilizados).

    """
    contexts = retrieve_context(query, top_k=top_k)
    answer = generate_answer(query, contexts)

    logger.info("RAG query completa: %d contextos usados", len(contexts))
    return answer, contexts
