"""Agente ReAct com tools customizadas para o domínio do Datathon.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting
             in Language Models. https://arxiv.org/abs/2210.03629
"""

import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from src.agent.tools import get_available_tools

load_dotenv()

logger = logging.getLogger(__name__)

REACT_PROMPT = PromptTemplate.from_template(
    """Você é um assistente especializado no domínio financeiro.
Use as ferramentas disponíveis para responder perguntas com precisão.
Sempre busque informações antes de responder — não invente dados.

Ferramentas disponíveis:
{tools}

Nomes das ferramentas: {tool_names}

Use o formato:
Thought: pensar sobre o que fazer
Action: nome_da_ferramenta
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita Thought/Action/Observation quantas vezes necessário)
Thought: Agora sei a resposta final
Final Answer: resposta para o usuário

Importante:
- Sempre cite as fontes quando usar a base de conhecimento
- Se não encontrar informação suficiente, diga claramente
- Nunca invente dados ou estatísticas

Pergunta: {input}
{agent_scratchpad}"""
)


def create_datathon_agent(
    tools: list[Tool] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    max_iterations: int = 10,
) -> AgentExecutor:
    """Cria agente ReAct para o Datathon.

    Args:
        tools: Lista de ferramentas (≥ 3 obrigatório). Se None, usa tools padrão.
        model_name: Modelo LLM a utilizar. Se None, usa env var.
        temperature: Temperatura de geração.
        max_iterations: Máximo de iterações do agente.

    Returns:
        AgentExecutor configurado.
    """
    if tools is None:
        tools = get_available_tools()

    if len(tools) < 3:
        logger.warning("Datathon exige ≥ 3 tools. Fornecidas: %d", len(tools))

    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(
        "Agente ReAct criado: model=%s, tools=%d, max_iter=%d",
        model_name,
        len(tools),
        max_iterations,
    )

    return executor


def run_agent(query: str, agent: AgentExecutor | None = None) -> dict:
    """Executa o agente com uma query.

    Args:
        query: Pergunta do usuário.
        agent: AgentExecutor pré-configurado. Se None, cria um novo.

    Returns:
        Dicionário com output e steps intermediários.
    """
    if agent is None:
        agent = create_datathon_agent()

    result = agent.invoke({"input": query})

    logger.info("Agente respondeu em %d steps", len(result.get("intermediate_steps", [])))

    return {
        "answer": result["output"],
        "steps": len(result.get("intermediate_steps", [])),
    }
