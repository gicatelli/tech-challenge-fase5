"""Feature engineering pipeline com validação de schema.

Este módulo implementa transformações de features com contratos
de schema (pandera) para garantir qualidade dos dados.
"""

import logging

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

logger = logging.getLogger(__name__)

# Schema de entrada — contrato de dados
INPUT_SCHEMA = DataFrameSchema(
    {
        "feature_1": Column(float, pa.Check.between(0, 1), nullable=False),
        "feature_2": Column(float, pa.Check.gt(0), nullable=False),
    },
    strict=False,  # Permite colunas extras
)

# Schema de saída — contrato de features
OUTPUT_SCHEMA = DataFrameSchema(
    {
        "feature_1": Column(float, pa.Check.between(0, 1)),
        "feature_2": Column(float, pa.Check.gt(0)),
        "feature_1_x_feature_2": Column(float),
        "feature_1_squared": Column(float, pa.Check.between(0, 1)),
        "feature_2_log": Column(float),
    },
    strict=False,
)


def compute_features(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Computa features derivadas a partir do DataFrame de entrada.

    Args:
        df: DataFrame com features brutas.
        validate: Se True, valida schemas de entrada e saída.

    Returns:
        DataFrame com features computadas.

    Raises:
        pandera.errors.SchemaError: Se validação de schema falhar.

    """
    logger.info("Computando features para %d registros", len(df))

    if validate:
        INPUT_SCHEMA.validate(df, lazy=True)

    result = df.copy()

    # Feature interactions
    result["feature_1_x_feature_2"] = result["feature_1"] * result["feature_2"]

    # Transformações não-lineares
    result["feature_1_squared"] = result["feature_1"] ** 2
    result["feature_2_log"] = np.log1p(result["feature_2"])

    # Preencher nulls (se houver) com mediana
    numeric_cols = result.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if result[col].isnull().any():
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)
            logger.warning("Coluna %s tinha nulls, preenchido com mediana=%.4f", col, median_val)

    if validate:
        OUTPUT_SCHEMA.validate(result, lazy=True)

    logger.info("Features computadas com sucesso: %d colunas", len(result.columns))
    return result


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Aplica one-hot encoding em colunas categóricas.

    Args:
        df: DataFrame de entrada.
        columns: Lista de colunas categóricas para encoding.

    Returns:
        DataFrame com colunas categóricas encodadas.

    """
    logger.info("Encoding categórico para colunas: %s", columns)
    return pd.get_dummies(df, columns=columns, drop_first=True, dtype=float)


if __name__ == "__main__":
    # Exemplo de uso
    import numpy as np

    np.random.seed(42)
    sample = pd.DataFrame(
        {
            "feature_1": np.random.uniform(0, 1, 100),
            "feature_2": np.random.uniform(1, 10, 100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    result = compute_features(sample)
    print(result.head())
    print(f"\nShape: {result.shape}")
