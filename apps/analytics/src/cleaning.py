"""Limpieza de las transacciones de crédito de la Unidad 2, Tema 3."""

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.db_connector import extract_raw_data, get_database_engine


def normalize_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza region y convierte edades fuera de 18-100 años en faltantes."""
    cleaned = df.copy()
    cleaned["region"] = cleaned["region"].map(
        lambda value: value.strip().title() if isinstance(value, str) else value
    )
    cleaned["region"] = cleaned["region"].replace("", np.nan)
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
    cleaned.loc[~cleaned["age"].between(18, 100), "age"] = np.nan
    return cleaned


def impute_column(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    """Imputa una columna por media, mediana o moda sin alterar el original."""
    result = df.copy()
    if strategy in ("mean", "median"):
        values = pd.to_numeric(result[column], errors="raise")
        replacement = values.mean() if strategy == "mean" else values.median()
        result[column] = values
    elif strategy == "mode":
        modes = result[column].mode(dropna=True)
        replacement = modes.iloc[0] if not modes.empty else np.nan
    else:
        raise ValueError(f"Estrategia de imputación no soportada: {strategy}")

    if pd.isna(replacement):
        raise ValueError(f"No hay valores válidos para imputar {column}")
    result[column] = result[column].fillna(replacement)
    return result


def cap_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Acota los valores superiores a Q3 + 1,5 IQR sin eliminar registros."""
    result = df.copy()
    values = pd.to_numeric(result[column], errors="raise")
    q1, q3 = values.quantile([0.25, 0.75])
    maximum = q3 + 1.5 * (q3 - q1)
    result[column] = values.clip(upper=maximum)
    return result


def run_cleaning_pipeline(engine: Engine | None = None) -> pd.DataFrame:
    """Extrae, limpia y guarda customer_credit_clean en PostgreSQL."""
    active_engine = engine if engine is not None else get_database_engine()
    raw = extract_raw_data(
        "SELECT * FROM customer_credit_transactions ORDER BY transaction_id",
        engine=active_engine,
    )
    if raw.empty:
        raise ValueError("customer_credit_transactions no contiene registros")

    variance_before = pd.to_numeric(raw["annual_income"]).var()
    cleaned = normalize_inconsistencies(raw)
    for column, strategy in (
        ("age", "median"),
        ("annual_income", "median"),
        ("credit_score", "mean"),
        ("loan_amount", "median"),
        ("region", "mode"),
    ):
        cleaned = impute_column(cleaned, column, strategy)
    for column in ("annual_income", "loan_amount"):
        cleaned = cap_outliers(cleaned, column)

    with active_engine.begin() as connection:
        cleaned.to_sql(
            "customer_credit_clean", connection, if_exists="replace", index=False
        )

    print(f"Varianza de annual_income antes: {variance_before:.2f}")
    print(f"Varianza de annual_income después: {cleaned['annual_income'].var():.2f}")
    print(f"Registros conservados en customer_credit_clean: {len(cleaned)}")
    return cleaned


if __name__ == "__main__":
    run_cleaning_pipeline()
