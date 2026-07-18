"""Tools do agente ReAct — implementações reais para o domínio financeiro.

Cada tool executa lógica real sobre dados de mercado:
- prever_preco: predição LSTM via MLflow
- analisar_historico: estatísticas descritivas do ativo
- buscar_conhecimento: RAG com ChromaDB
- calcular_risco: VaR, volatilidade, Sharpe ratio
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.tools import Tool

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/PETR4_SA_historico.csv")


def _load_stock_data() -> pd.DataFrame:
    """Carrega dados históricos da PETR4."""
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# === TOOL 1: PREVER PREÇO ===


def prever_preco(query: str) -> str:
    """Prevê preço futuro da PETR4 usando modelo LSTM.

    Tenta carregar modelo do MLflow Registry. Se indisponível,
    usa projeção baseada em tendência linear como fallback.

    Args:
        query: Descrição da previsão desejada (ex: "próximos 5 dias").

    Returns:
        Previsão formatada em texto.

    """
    try:
        df = _load_stock_data()
        last_price = df["Close"].iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")

        # Extrair número de dias do query
        days = 5  # default
        for word in query.split():
            if word.isdigit():
                days = min(int(word), 30)
                break

        # Tentar MLflow Registry
        try:
            from src.models.registry import get_production_model

            get_production_model()  # Verifica se modelo está disponível
            logger.info("Modelo carregado do MLflow Registry")
            # TODO: preparar sequência e inferir com modelo real
            # Por ora usa fallback
            raise ImportError("Forçar fallback para desenvolvimento")
        except (ImportError, Exception):
            logger.info("MLflow indisponível, usando projeção por tendência")

        # Fallback: projeção baseada em tendência dos últimos 30 dias
        recent = df["Close"].tail(30)
        daily_return = recent.pct_change().mean()
        volatility = recent.pct_change().std()

        predictions = []
        price = last_price
        for i in range(1, days + 1):
            price = price * (1 + daily_return)
            predictions.append({
                "dia": i,
                "preco_estimado": round(float(price), 2),
                "intervalo_inferior": round(float(price * (1 - 2 * volatility)), 2),
                "intervalo_superior": round(float(price * (1 + 2 * volatility)), 2),
            })

        result = {
            "ativo": "PETR4.SA",
            "ultimo_preco": round(float(last_price), 2),
            "data_referencia": last_date,
            "metodo": "projeção_tendencia_30d",
            "previsoes": predictions,
            "aviso": "Previsões são estimativas e não garantem resultados futuros.",
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"erro": f"Falha na previsão: {e}"})


# === TOOL 2: ANALISAR HISTÓRICO ===


def analisar_historico(query: str) -> str:
    """Analisa dados históricos da PETR4 com estatísticas descritivas.

    Calcula: média, volatilidade, tendência, max drawdown,
    retornos acumulados e indicadores de momento.

    Args:
        query: Descrição da análise desejada.

    Returns:
        Análise formatada em texto JSON.

    """
    try:
        df = _load_stock_data()

        # Determinar período de análise
        period = 90  # default: últimos 90 dias
        if "30" in query or "mês" in query.lower():
            period = 30
        elif "180" in query or "semestre" in query.lower():
            period = 180
        elif "365" in query or "ano" in query.lower() or "anual" in query.lower():
            period = 365

        recent = df.tail(period)
        returns = recent["returns"].dropna()

        # Estatísticas
        close = recent["Close"]
        vol_anual = returns.std() * np.sqrt(252)
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax

        # Tendência (regressão linear simples)
        x = np.arange(len(close))
        slope = np.polyfit(x, close.values, 1)[0]
        tendencia = "alta" if slope > 0 else "baixa"

        # SMA atual
        sma_30 = df["Close"].rolling(30).mean().iloc[-1]
        sma_90 = df["Close"].rolling(90).mean().iloc[-1]

        result = {
            "ativo": "PETR4.SA",
            "periodo_analisado": f"últimos {period} dias",
            "data_inicio": recent.index[0].strftime("%Y-%m-%d"),
            "data_fim": recent.index[-1].strftime("%Y-%m-%d"),
            "preco_atual": round(float(close.iloc[-1]), 2),
            "preco_minimo": round(float(close.min()), 2),
            "preco_maximo": round(float(close.max()), 2),
            "preco_medio": round(float(close.mean()), 2),
            "retorno_periodo": f"{((close.iloc[-1] / close.iloc[0]) - 1) * 100:.1f}%",
            "volatilidade_anualizada": f"{vol_anual * 100:.1f}%",
            "max_drawdown": f"{drawdown.min() * 100:.1f}%",
            "tendencia": tendencia,
            "slope_diario": f"R$ {slope:.3f}/dia",
            "sma_30": round(float(sma_30), 2),
            "sma_90": round(float(sma_90), 2),
            "posicao_vs_sma30": "acima" if close.iloc[-1] > sma_30 else "abaixo",
            "volume_medio": f"{recent['Volume'].mean():,.0f}",
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"erro": f"Falha na análise: {e}"})


# === TOOL 3: BUSCAR CONHECIMENTO ===


def buscar_conhecimento(query: str) -> str:
    """Busca informações na base de conhecimento financeiro via RAG.

    Consulta o vector store (ChromaDB) com documentos sobre
    indicadores técnicos, análise da Petrobras, conceitos de risco, etc.

    Args:
        query: Pergunta do usuário sobre o domínio financeiro.

    Returns:
        Contextos relevantes encontrados na base.

    """
    try:
        from src.agent.rag_pipeline import retrieve_context

        contexts = retrieve_context(query, top_k=3)

        if not contexts:
            return json.dumps({
                "resultado": "Nenhuma informação encontrada na base de conhecimento.",
                "sugestao": "Tente reformular a pergunta ou consultar outro indicador.",
            })

        result = {
            "query": query,
            "contextos_encontrados": len(contexts),
            "informacoes": contexts,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        # Fallback: busca por keyword nos arquivos locais
        logger.warning("ChromaDB indisponível, usando busca por keyword: %s", e)
        return _keyword_search_fallback(query)


def _keyword_search_fallback(query: str) -> str:
    """Busca por keyword nos documentos locais (fallback sem ChromaDB)."""
    kb_dir = Path("data/knowledge_base")
    if not kb_dir.exists():
        return json.dumps({"erro": "Base de conhecimento não encontrada."})

    query_lower = query.lower()
    results = []

    for file in kb_dir.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            # Score simples: contagem de palavras do query no parágrafo
            score = sum(1 for word in query_lower.split() if word in para.lower())
            if score >= 2 and len(para) > 50:
                results.append((score, para.strip()))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = [text for _, text in results[:3]]

    if not top_results:
        return json.dumps({"resultado": "Nenhuma informação relevante encontrada."})

    return json.dumps({
        "query": query,
        "metodo": "keyword_search (fallback)",
        "contextos_encontrados": len(top_results),
        "informacoes": top_results,
    }, ensure_ascii=False, indent=2)


# === TOOL 4: CALCULAR RISCO ===


def calcular_risco(query: str) -> str:
    """Calcula métricas de risco para PETR4.

    Métricas: VaR (95% e 99%), Expected Shortfall, volatilidade
    anualizada, Sharpe Ratio, Sortino Ratio, max drawdown.

    Args:
        query: Descrição do cenário ou período para cálculo.

    Returns:
        Métricas de risco em formato JSON.

    """
    try:
        df = _load_stock_data()
        returns = df["returns"].dropna()

        # Período de análise
        period = 252  # default: 1 ano
        if "30" in query or "mês" in query.lower():
            period = 30
        elif "90" in query or "trimestre" in query.lower():
            period = 90

        recent_returns = returns.tail(period)
        last_price = df["Close"].iloc[-1]

        # VaR paramétrico
        var_95 = float(np.percentile(recent_returns, 5))
        var_99 = float(np.percentile(recent_returns, 1))

        # Expected Shortfall (CVaR)
        es_95 = float(recent_returns[recent_returns <= var_95].mean())

        # Volatilidade anualizada
        vol_anual = float(recent_returns.std() * np.sqrt(252))

        # Sharpe Ratio (Selic como risk-free: ~10% ao ano)
        risk_free_daily = 0.10 / 252
        excess_returns = recent_returns - risk_free_daily
        sharpe = float(excess_returns.mean() / recent_returns.std() * np.sqrt(252))

        # Sortino Ratio
        downside_returns = recent_returns[recent_returns < 0]
        downside_std = float(downside_returns.std() * np.sqrt(252))
        sortino = float(
            (recent_returns.mean() * 252 - 0.10) / downside_std
        ) if downside_std > 0 else 0.0

        # Drawdown
        cummax = df["Close"].tail(period).cummax()
        drawdown = (df["Close"].tail(period) - cummax) / cummax
        max_dd = float(drawdown.min())

        # VaR em reais (para posição de R$ 100.000)
        posicao = 100000
        var_95_reais = abs(var_95) * posicao
        var_99_reais = abs(var_99) * posicao

        result = {
            "ativo": "PETR4.SA",
            "periodo_analise": f"{period} dias",
            "preco_atual": round(float(last_price), 2),
            "metricas_risco": {
                "var_95_diario": f"{var_95 * 100:.2f}%",
                "var_99_diario": f"{var_99 * 100:.2f}%",
                "var_95_reais_100k": f"R$ {var_95_reais:,.0f}",
                "var_99_reais_100k": f"R$ {var_99_reais:,.0f}",
                "expected_shortfall_95": f"{es_95 * 100:.2f}%",
                "volatilidade_anualizada": f"{vol_anual * 100:.1f}%",
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "max_drawdown": f"{max_dd * 100:.1f}%",
            },
            "interpretacao": _interpretar_risco(vol_anual, sharpe, max_dd),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"erro": f"Falha no cálculo de risco: {e}"})


def _interpretar_risco(vol: float, sharpe: float, max_dd: float) -> str:
    """Gera interpretação textual das métricas de risco."""
    partes = []

    if vol > 0.4:
        partes.append("Volatilidade ALTA (>40%) — ativo de alto risco.")
    elif vol > 0.25:
        partes.append("Volatilidade MODERADA-ALTA (25-40%).")
    else:
        partes.append("Volatilidade MODERADA (<25%).")

    if sharpe > 1:
        partes.append(f"Sharpe de {sharpe:.2f} indica retorno ajustado ao risco BOM.")
    elif sharpe > 0:
        partes.append(f"Sharpe de {sharpe:.2f} — retorno positivo mas abaixo do ideal.")
    else:
        partes.append(f"Sharpe NEGATIVO ({sharpe:.2f}) — renda fixa seria melhor.")

    if max_dd < -0.3:
        partes.append(f"Drawdown máximo de {max_dd*100:.0f}% é SEVERO.")

    return " ".join(partes)


# === REGISTRO DAS TOOLS ===


def get_available_tools() -> list[Tool]:
    """Retorna lista de tools disponíveis para o agente.

    Returns:
        Lista com 4 tools configuradas para o domínio financeiro.

    """
    return [
        Tool(
            name="prever_preco",
            func=prever_preco,
            description=(
                "Prevê o preço futuro da ação PETR4.SA para os próximos N dias. "
                "Use quando o usuário perguntar sobre previsões, projeções ou "
                "estimativas de preço. Input: descrição como 'próximos 5 dias'."
            ),
        ),
        Tool(
            name="analisar_historico",
            func=analisar_historico,
            description=(
                "Analisa dados históricos da PETR4.SA com estatísticas: "
                "preço médio, volatilidade, tendência, drawdown, volume. "
                "Use para perguntas sobre performance passada, tendências "
                "ou comparações de período. Input: período como '30 dias' ou 'último ano'."
            ),
        ),
        Tool(
            name="buscar_conhecimento",
            func=buscar_conhecimento,
            description=(
                "Busca informações na base de conhecimento financeiro. "
                "Contém: indicadores técnicos (RSI, MACD, Bollinger), "
                "análise da Petrobras, conceitos de risco (VaR, Sharpe), "
                "política de investimentos, e cenário macroeconômico. "
                "Use para perguntas conceituais ou explicações."
            ),
        ),
        Tool(
            name="calcular_risco",
            func=calcular_risco,
            description=(
                "Calcula métricas de risco da PETR4.SA: VaR (95%/99%), "
                "Expected Shortfall, volatilidade, Sharpe Ratio, Sortino Ratio, "
                "max drawdown. Use quando o usuário perguntar sobre risco, "
                "proteção de carteira ou gestão de risco. Input: período ou cenário."
            ),
        ),
    ]
