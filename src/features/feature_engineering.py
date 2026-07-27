"""Feature engineering para séries temporais financeiras.

Implementa indicadores técnicos padronizados com validação de schema (Pandera)
para garantir qualidade e reprodutibilidade do pipeline de dados.

Indicadores implementados:
- Médias Móveis (SMA 7, 30, 90)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Retornos logarítmicos
- Volatilidade rolling
- Volume normalizado
- Features exógenas (Brent, USD/BRL, Ibovespa)
- Features de correlação (VALE3, ITUB4)
"""

import logging

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

logger = logging.getLogger(__name__)


# === SCHEMAS DE VALIDAÇÃO ===

# Schema de entrada — dados brutos OHLCV do yfinance
INPUT_SCHEMA = DataFrameSchema(
    {
        "Close": Column(float, pa.Check.gt(0), nullable=False),
        "High": Column(float, pa.Check.gt(0), nullable=False),
        "Low": Column(float, pa.Check.gt(0), nullable=False),
        "Open": Column(float, pa.Check.gt(0), nullable=False),
        "Volume": Column(int, pa.Check.ge(0), nullable=False),
    },
    strict=False,
)

# Schema de saída — features computadas
OUTPUT_SCHEMA = DataFrameSchema(
    {
        "close": Column(float, pa.Check.gt(0)),
        "sma_7": Column(float, nullable=True),
        "sma_30": Column(float, nullable=True),
        "sma_90": Column(float, nullable=True),
        "rsi_14": Column(float, pa.Check.between(0, 100), nullable=True),
        "macd": Column(float, nullable=True),
        "macd_signal": Column(float, nullable=True),
        "bb_upper": Column(float, nullable=True),
        "bb_lower": Column(float, nullable=True),
        "log_return": Column(float, nullable=True),
        "volatility_30": Column(float, pa.Check.ge(0), nullable=True),
        "volume_norm": Column(float, pa.Check.ge(0), nullable=True),
    },
    strict=False,
)


# === FUNÇÕES DE INDICADORES TÉCNICOS ===


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Calcula Simple Moving Average (SMA).

    Args:
        series: Série de preços.
        window: Janela de cálculo.

    Returns:
        Série com SMA calculada.

    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcula Relative Strength Index (RSI).

    RSI mede a magnitude de mudanças recentes de preço para avaliar
    condições de sobrecompra (>70) ou sobrevenda (<30).

    Args:
        series: Série de preços de fechamento.
        period: Período para cálculo (padrão: 14 dias).

    Returns:
        Série com RSI (0-100).

    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula MACD (Moving Average Convergence Divergence).

    MACD detecta mudanças na força, direção, momentum e duração
    de uma tendência.

    Args:
        series: Série de preços de fechamento.
        fast: Período da EMA rápida (padrão: 12).
        slow: Período da EMA lenta (padrão: 26).
        signal: Período da linha de sinal (padrão: 9).

    Returns:
        Tupla (macd_line, signal_line, histogram).

    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula Bollinger Bands.

    Bandas de Bollinger medem volatilidade e identificam
    condições de sobrecompra/sobrevenda relativas.

    Args:
        series: Série de preços de fechamento.
        window: Janela da média móvel (padrão: 20).
        num_std: Número de desvios padrão (padrão: 2.0).

    Returns:
        Tupla (upper_band, middle_band, lower_band).

    """
    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return upper, middle, lower


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Calcula retornos logarítmicos.

    Log returns são preferidos em finanças por serem aditivos no tempo
    e terem distribuição mais simétrica.

    Args:
        series: Série de preços.

    Returns:
        Série com retornos logarítmicos.

    """
    return np.log(series / series.shift(1))


def compute_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calcula volatilidade rolling (desvio padrão anualizado).

    Args:
        returns: Série de retornos.
        window: Janela de cálculo (padrão: 30 dias).

    Returns:
        Série com volatilidade anualizada.

    """
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def normalize_volume(volume: pd.Series, window: int = 30) -> pd.Series:
    """Normaliza volume pela média móvel (ratio).

    Volume acima de 1.0 indica atividade acima da média,
    abaixo de 1.0 indica atividade abaixo da média.

    Args:
        volume: Série de volume.
        window: Janela para média móvel (padrão: 30).

    Returns:
        Série com volume normalizado (ratio vs média).

    """
    vol_mean = volume.rolling(window=window, min_periods=window).mean()
    return volume / vol_mean.replace(0, np.nan)


# === PIPELINE PRINCIPAL ===


def compute_features(
    df: pd.DataFrame,
    validate: bool = True,
    include_exogenous: bool = True,
    include_correlations: bool = True,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Computa features técnicas a partir de dados OHLCV.

    Pipeline completo de feature engineering para séries temporais
    financeiras. Gera indicadores técnicos padronizados, features
    exógenas (Brent, USD/BRL, Ibovespa) e features de correlação
    (VALE3, ITUB4). Remove linhas iniciais com NaN (warm-up period).

    Args:
        df: DataFrame com colunas OHLCV (Open, High, Low, Close, Volume).
        validate: Se True, valida schemas de entrada e saída.
        include_exogenous: Se True, adiciona Brent, USD/BRL, Ibovespa.
        include_correlations: Se True, adiciona VALE3, ITUB4 como features.
        data_dir: Diretório com dados brutos para features exógenas.

    Returns:
        DataFrame com features técnicas computadas (sem NaN).

    Raises:
        pandera.errors.SchemaError: Se validação de schema falhar.

    """
    logger.info("Computando features para %d registros", len(df))

    if validate:
        INPUT_SCHEMA.validate(df, lazy=True)

    result = pd.DataFrame(index=df.index)

    # Preço base
    result["close"] = df["Close"].astype(float)
    result["high"] = df["High"].astype(float)
    result["low"] = df["Low"].astype(float)
    result["open"] = df["Open"].astype(float)

    # Médias Móveis (SMA)
    result["sma_7"] = compute_sma(df["Close"], 7)
    result["sma_30"] = compute_sma(df["Close"], 30)
    result["sma_90"] = compute_sma(df["Close"], 90)

    # Ratios preço/SMA (normalização relativa)
    result["price_sma7_ratio"] = df["Close"] / result["sma_7"]
    result["price_sma30_ratio"] = df["Close"] / result["sma_30"]
    result["price_sma90_ratio"] = df["Close"] / result["sma_90"]

    # RSI
    result["rsi_14"] = compute_rsi(df["Close"], period=14)

    # MACD
    macd_line, signal_line, histogram = compute_macd(df["Close"])
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df["Close"])
    result["bb_upper"] = bb_upper
    result["bb_middle"] = bb_middle
    result["bb_lower"] = bb_lower
    result["bb_width"] = (bb_upper - bb_lower) / bb_middle  # Largura normalizada
    result["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)  # Posição 0-1

    # Retornos logarítmicos
    result["log_return"] = compute_log_returns(df["Close"])
    result["log_return_2d"] = compute_log_returns(df["Close"]).rolling(2).sum()
    result["log_return_5d"] = compute_log_returns(df["Close"]).rolling(5).sum()

    # Volatilidade
    log_ret = compute_log_returns(df["Close"])
    result["volatility_7"] = compute_volatility(log_ret, window=7)
    result["volatility_30"] = compute_volatility(log_ret, window=30)

    # Volume normalizado
    result["volume_norm"] = normalize_volume(df["Volume"].astype(float))

    # Features de range (High-Low)
    result["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    result["daily_range_ma"] = result["daily_range"].rolling(14).mean()

    # === FEATURES EXÓGENAS (Brent, USD/BRL, Ibovespa) ===
    if include_exogenous:
        try:
            exog = load_exogenous_features(df.index, data_dir=data_dir)
            if not exog.empty:
                for col in exog.columns:
                    result[col] = exog[col]
                logger.info("Features exógenas adicionadas: %d colunas", len(exog.columns))
        except Exception as e:
            logger.warning("Features exógenas não disponíveis: %s", e)

    # === FEATURES DE CORRELAÇÃO (VALE3, ITUB4) ===
    if include_correlations:
        try:
            corr_features = load_correlation_features(df.index, data_dir=data_dir)
            if not corr_features.empty:
                for col in corr_features.columns:
                    result[col] = corr_features[col]
                logger.info("Features de correlação adicionadas: %d colunas", len(corr_features.columns))
        except Exception as e:
            logger.warning("Features de correlação não disponíveis: %s", e)

    # Remover linhas com NaN (warm-up period dos indicadores)
    initial_rows = len(result)
    result = result.dropna()
    dropped = initial_rows - len(result)
    logger.info(
        "Removidas %d linhas (warm-up). Features finais: %d linhas x %d colunas",
        dropped,
        len(result),
        len(result.columns),
    )

    if validate:
        OUTPUT_SCHEMA.validate(result, lazy=True)

    return result


def load_exogenous_features(
    petr4_index: pd.DatetimeIndex,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Carrega e prepara features exógenas (Brent, USD/BRL, Ibovespa).

    Features exógenas capturam fatores macroeconômicos que impactam
    PETR4 mas não estão nos dados OHLCV do ativo:
    - Brent: correlação direta com receita da Petrobras
    - USD/BRL: impacta dívida e receita de exportação
    - Ibovespa: sentimento geral do mercado brasileiro

    Args:
        petr4_index: DatetimeIndex do DataFrame principal (para alinhamento).
        data_dir: Diretório com dados brutos.

    Returns:
        DataFrame com features exógenas alinhadas ao index de PETR4.

    """
    from pathlib import Path

    logger.info("Carregando features exógenas...")

    exog = pd.DataFrame(index=petr4_index)

    # Tentar carregar dados exógenos (Brent, USD/BRL, Ibovespa)
    try:
        import yfinance as yf

        # Brent Crude Oil (BZ=F)
        brent_path = Path(data_dir) / "BZ_F_historico.csv"
        if brent_path.exists():
            brent = pd.read_csv(brent_path, index_col=0, parse_dates=True)
            brent_close = brent["Close"]
        else:
            brent_close = yf.download(
                "BZ=F", start=petr4_index[0], end=petr4_index[-1], progress=False
            )["Close"]
            if isinstance(brent_close, pd.DataFrame):
                brent_close = brent_close.iloc[:, 0]

        # USD/BRL (USDBRL=X)
        usdbrl_path = Path(data_dir) / "USDBRL_X_historico.csv"
        if usdbrl_path.exists():
            usdbrl = pd.read_csv(usdbrl_path, index_col=0, parse_dates=True)
            usdbrl_close = usdbrl["Close"]
        else:
            usdbrl_close = yf.download(
                "USDBRL=X", start=petr4_index[0], end=petr4_index[-1], progress=False
            )["Close"]
            if isinstance(usdbrl_close, pd.DataFrame):
                usdbrl_close = usdbrl_close.iloc[:, 0]

        # Ibovespa (^BVSP)
        ibov_path = Path(data_dir) / "BVSP_historico.csv"
        if ibov_path.exists():
            ibov = pd.read_csv(ibov_path, index_col=0, parse_dates=True)
            ibov_close = ibov["Close"]
        else:
            ibov_close = yf.download(
                "^BVSP", start=petr4_index[0], end=petr4_index[-1], progress=False
            )["Close"]
            if isinstance(ibov_close, pd.DataFrame):
                ibov_close = ibov_close.iloc[:, 0]

        # Alinhar ao index de PETR4 (forward fill para gaps de feriados)
        if len(brent_close) > 0:
            exog["brent_close"] = brent_close.reindex(petr4_index, method="ffill")
            exog["brent_return"] = compute_log_returns(exog["brent_close"])
            exog["brent_return_5d"] = exog["brent_return"].rolling(5).sum()
            logger.info("Brent adicionado: %d registros", exog["brent_close"].notna().sum())

        if len(usdbrl_close) > 0:
            exog["usdbrl_close"] = usdbrl_close.reindex(petr4_index, method="ffill")
            exog["usdbrl_return"] = compute_log_returns(exog["usdbrl_close"])
            exog["usdbrl_volatility"] = compute_volatility(exog["usdbrl_return"], window=14)
            logger.info("USD/BRL adicionado: %d registros", exog["usdbrl_close"].notna().sum())

        if len(ibov_close) > 0:
            exog["ibov_close"] = ibov_close.reindex(petr4_index, method="ffill")
            exog["ibov_return"] = compute_log_returns(exog["ibov_close"])
            exog["ibov_return_5d"] = exog["ibov_return"].rolling(5).sum()
            exog["ibov_rsi_14"] = compute_rsi(exog["ibov_close"], period=14)
            logger.info("Ibovespa adicionado: %d registros", exog["ibov_close"].notna().sum())

    except Exception as e:
        logger.warning("Falha ao carregar features exógenas: %s. Continuando sem elas.", e)

    # Remover colunas de preço bruto (manter apenas retornos e indicadores derivados)
    drop_cols = [c for c in exog.columns if c.endswith("_close")]
    exog = exog.drop(columns=drop_cols, errors="ignore")

    logger.info("Features exógenas: %d colunas", len(exog.columns))
    return exog


def load_correlation_features(
    petr4_index: pd.DatetimeIndex,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Carrega features de correlação cruzada (VALE3, ITUB4).

    VALE3 e ITUB4 são os ativos mais líquidos da B3 junto com PETR4.
    Seus movimentos relativos capturam rotação setorial e sentimento
    de mercado que impactam PETR4.

    Features geradas:
    - Retornos de VALE3/ITUB4 (momentum cross-asset)
    - Spread PETR4 vs VALE3/ITUB4 (mean reversion)
    - Correlação rolling (regime detection)

    Args:
        petr4_index: DatetimeIndex do DataFrame principal.
        data_dir: Diretório com dados brutos.

    Returns:
        DataFrame com features de correlação alinhadas.

    """
    from pathlib import Path

    logger.info("Carregando features de correlação (VALE3, ITUB4)...")

    corr = pd.DataFrame(index=petr4_index)

    # Carregar VALE3
    vale3_path = Path(data_dir) / "VALE3_SA_historico.csv"
    if vale3_path.exists():
        vale3 = pd.read_csv(vale3_path, index_col=0, parse_dates=True)
        vale3_close = vale3["Close"].reindex(petr4_index, method="ffill")

        corr["vale3_return"] = compute_log_returns(vale3_close)
        corr["vale3_return_5d"] = corr["vale3_return"].rolling(5).sum()
        corr["vale3_rsi_14"] = compute_rsi(vale3_close, period=14)
        logger.info("VALE3 adicionado como feature de correlação")
    else:
        logger.warning("VALE3 não encontrado em %s", vale3_path)

    # Carregar ITUB4
    itub4_path = Path(data_dir) / "ITUB4_SA_historico.csv"
    if itub4_path.exists():
        itub4 = pd.read_csv(itub4_path, index_col=0, parse_dates=True)
        itub4_close = itub4["Close"].reindex(petr4_index, method="ffill")

        corr["itub4_return"] = compute_log_returns(itub4_close)
        corr["itub4_return_5d"] = corr["itub4_return"].rolling(5).sum()
        corr["itub4_rsi_14"] = compute_rsi(itub4_close, period=14)
        logger.info("ITUB4 adicionado como feature de correlação")
    else:
        logger.warning("ITUB4 não encontrado em %s", itub4_path)

    logger.info("Features de correlação: %d colunas", len(corr.columns))
    return corr


def prepare_sequences(
    features_df: pd.DataFrame,
    target_col: str = "close",
    sequence_length: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara sequências para input do LSTM.

    Cria janelas deslizantes de tamanho fixo para alimentar
    o modelo LSTM. O target é o preço de fechamento do próximo dia.

    Args:
        features_df: DataFrame com features computadas.
        target_col: Coluna alvo para predição.
        sequence_length: Tamanho da janela de entrada (padrão: 90 dias).

    Returns:
        Tupla (X, y) onde X tem shape (n_samples, sequence_length, n_features)
        e y tem shape (n_samples,).

    """
    data = features_df.values
    target_idx = features_df.columns.get_loc(target_col)

    x_list: list = []
    y_list: list = []
    for i in range(sequence_length, len(data)):
        x_list.append(data[i - sequence_length:i])
        y_list.append(data[i, target_idx])

    X = np.array(x_list)
    y = np.array(y_list)

    logger.info(
        "Sequências criadas: X=%s, y=%s (seq_len=%d)",
        X.shape,
        y.shape,
        sequence_length,
    )

    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Carregar dados reais
    from pathlib import Path

    data_path = Path("data/raw/PETR4_SA_historico.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Converter Volume para int (pode vir como float do CSV)
    df["Volume"] = df["Volume"].astype(int)

    print(f"Dados de entrada: {df.shape}")
    print(f"Período: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    # Computar features
    features = compute_features(df)

    print(f"\nFeatures computadas: {features.shape}")
    print(f"Colunas: {features.columns.tolist()}")
    print("\nPrimeiras linhas:")
    print(features.head())
    print("\nEstatísticas:")
    print(features.describe().round(4))

    # Preparar sequências para LSTM
    X, y = prepare_sequences(features, target_col="close", sequence_length=90)
    print(f"\nSequências LSTM: X={X.shape}, y={y.shape}")

    # Salvar features processadas
    output_path = Path("data/processed/features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    print(f"\nFeatures salvas em: {output_path}")
