"""Módulo de Coleta de Dados Financeiros.

Responsável por coletar dados históricos de ações via Yahoo Finance
e persistir em formato CSV para versionamento com DVC.

Evolução da Fase 4: agora coleta múltiplas ações e adiciona metadata
para rastreabilidade e reprodutibilidade.
"""

import json
import logging
import os
import ssl
from datetime import datetime
from pathlib import Path

import pandas as pd
import urllib3
import yfinance as yf

# Desabilitar warnings de SSL para ambientes com certificados corporativos
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONHTTPSVERIFY"] = "0"

# Fix para SSL em ambientes com proxy/certificado auto-assinado
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Ações monitoradas no projeto
DEFAULT_SYMBOLS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
DEFAULT_START_DATE = "2018-01-01"
DATA_RAW_DIR = Path("data/raw")


def collect_stock_data(
    symbol: str = "PETR4.SA",
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Coleta dados históricos de preços de ações usando yfinance.

    Args:
        symbol: Símbolo da ação (ex: 'PETR4.SA' para Petrobras).
        start_date: Data de início no formato 'YYYY-MM-DD'.
        end_date: Data de fim no formato 'YYYY-MM-DD' (default: hoje).

    Returns:
        DataFrame com dados históricos (Open, High, Low, Close, Volume).

    Raises:
        ValueError: Se nenhum dado for encontrado para o símbolo.

    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("Coletando dados de %s de %s até %s...", symbol, start_date, end_date)

    # Configurar session sem verificação SSL (ambientes corporativos)
    import requests

    session = requests.Session()
    session.verify = False

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        session=session,
    )

    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para o símbolo {symbol}")

    # Remover MultiIndex se existir (yfinance retorna MultiIndex para single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Garantir que o index é DatetimeIndex nomeado
    df.index.name = "Date"

    logger.info(
        "Dados coletados: %s | Shape: %s | Período: %s até %s",
        symbol,
        df.shape,
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )

    return df


def get_stock_info(symbol: str = "PETR4.SA") -> dict:
    """Obtém informações gerais sobre a ação.

    Args:
        symbol: Símbolo da ação.

    Returns:
        Dicionário com informações da empresa.

    """
    import requests

    session = requests.Session()
    session.verify = False

    ticker = yf.Ticker(symbol, session=session)
    info = ticker.info

    return {
        "symbol": symbol,
        "name": info.get("longName", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "currency": info.get("currency", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
    }


def collect_multiple_stocks(
    symbols: list[str] = DEFAULT_SYMBOLS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    output_dir: Path = DATA_RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Coleta dados de múltiplas ações e salva em CSV.

    Args:
        symbols: Lista de símbolos a coletar.
        start_date: Data de início.
        end_date: Data de fim (default: hoje).
        output_dir: Diretório de saída para os CSVs.

    Returns:
        Dicionário {symbol: DataFrame} com os dados coletados.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for i, symbol in enumerate(symbols):
        try:
            # Delay entre requests para evitar rate limit
            if i > 0:
                import time
                time.sleep(5)

            df = collect_stock_data(symbol, start_date, end_date)
            results[symbol] = df

            # Salvar CSV
            filename = f"{symbol.replace('.', '_')}_historico.csv"
            filepath = output_dir / filename
            df.to_csv(filepath)
            logger.info("Salvo: %s (%d registros)", filepath, len(df))

        except ValueError as e:
            logger.error("Falha ao coletar %s: %s", symbol, e)
            continue

    # Salvar metadata da coleta
    metadata = {
        "collected_at": datetime.now().isoformat(),
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
        "records": {s: len(df) for s, df in results.items()},
    }


    metadata_path = output_dir / "collection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata salvo: %s", metadata_path)

    return results


def collect_combined_dataset(
    symbols: list[str] = DEFAULT_SYMBOLS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    output_dir: Path = DATA_RAW_DIR,
) -> pd.DataFrame:
    """Coleta dados de múltiplas ações e retorna DataFrame combinado.

    Útil para análise comparativa e correlações entre ações.

    Args:
        symbols: Lista de símbolos.
        start_date: Data de início.
        end_date: Data de fim.
        output_dir: Diretório de saída.

    Returns:
        DataFrame com preços de fechamento de todas as ações (colunas = símbolos).

    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("Coletando dataset combinado: %s", symbols)

    # Configurar session sem verificação SSL
    import requests

    session = requests.Session()
    session.verify = False

    # Download em batch (mais eficiente)
    df_all = yf.download(symbols, start=start_date, end=end_date, progress=False, session=session)

    if df_all.empty:
        raise ValueError(f"Nenhum dado encontrado para os símbolos: {symbols}")

    # Extrair apenas Close prices
    if isinstance(df_all.columns, pd.MultiIndex):
        close_prices = df_all["Close"]
    else:
        close_prices = df_all[["Close"]]

    # Salvar
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "combined_close_prices.csv"
    close_prices.to_csv(filepath)
    logger.info("Dataset combinado salvo: %s | Shape: %s", filepath, close_prices.shape)

    return close_prices


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import time

    print("=" * 60)
    print("  COLETA DE DADOS — DATATHON FASE 05")
    print("=" * 60)

    # Verificar se já existem dados locais (evitar rate limit desnecessário)
    existing_files = list(DATA_RAW_DIR.glob("*_historico.csv"))
    if existing_files:
        print(f"\n[INFO] Dados locais encontrados: {[f.name for f in existing_files]}")
        print("[INFO] Para forçar re-coleta, delete os CSVs em data/raw/")

        # Mostrar resumo dos dados existentes
        for f in existing_files:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            start = df.index[0].strftime('%Y-%m-%d')
            end = df.index[-1].strftime('%Y-%m-%d')
            print(f"  {f.name}: {len(df)} registros | {start} → {end}")
    else:
        # Coletar ações individuais com retry
        print("\n[1/2] Coletando ações individuais (com retry)...")
        max_retries = 3
        retry_delay = 30  # segundos

        for attempt in range(max_retries):
            results = collect_multiple_stocks(
                symbols=DEFAULT_SYMBOLS,
                start_date=DEFAULT_START_DATE,
            )
            if results:
                break
            print(
                f"\n[RETRY] Tentativa {attempt + 1}/{max_retries} falhou. "
                f"Aguardando {retry_delay}s..."
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # backoff exponencial

        if results:
            # Coletar dataset combinado
            print("\n[2/2] Gerando dataset combinado (Close prices)...")
            time.sleep(5)
            try:
                combined = collect_combined_dataset(
                    symbols=DEFAULT_SYMBOLS,
                    start_date=DEFAULT_START_DATE,
                )
                print(f"\n  Dataset combinado: {combined.shape}")
            except ValueError as e:
                print(f"\n[WARN] Dataset combinado falhou: {e}")
                print("[INFO] Será gerado a partir dos CSVs individuais.")

    # Resumo final
    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    for f in DATA_RAW_DIR.glob("*_historico.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        print(f"  {f.name}: {len(df)} registros")
    print(f"\n  Diretório: {DATA_RAW_DIR.resolve()}")
