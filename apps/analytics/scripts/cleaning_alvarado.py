"""Parte A (U2-T3): reimplementación de la especificación básica de limpieza.

Implementa las firmas simples de la spec (todas devuelven solo pd.DataFrame,
salvo run_cleaning_pipeline que devuelve un dict) y escribe el resultado en la
tabla propia `customer_credit_clean_alvarado`, sin tocar `customer_credit_clean`.

Ejecución (desde apps/analytics, dentro del contenedor):
    python scripts/cleaning_alvarado.py
"""
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Permite importar `src.*` aunque el script se ejecute como `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db_connector import extract_raw_data, get_database_engine  # noqa: E402

RAW_QUERY = "SELECT * FROM customer_credit_transactions ORDER BY transaction_id;"
OUTPUT_TABLE = "customer_credit_clean_alvarado"

MIN_VALID_AGE = 18
MAX_VALID_AGE = 100
KNN_NEIGHBORS = 5
ZSCORE_THRESHOLD = 3.0
IQR_FACTOR = 1.5

COLUMNAS_NUMERICAS = ["age", "annual_income", "credit_score", "loan_amount"]
COLUMNAS_OUTLIERS = ["annual_income", "loan_amount"]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza `region` (strip + Title Case) y fuerza a NaN las edades
    fuera del rango de negocio [18, 100]."""
    df = df.copy()

    if "region" in df.columns:
        region = df["region"].str.strip().str.title()
        df["region"] = region.mask(region.isin(["", "Nan", "None"]), np.nan)

    if "age" in df.columns:
        df["age"] = df["age"].where(df["age"].between(MIN_VALID_AGE, MAX_VALID_AGE))

    return df


def impute_column(
    df: pd.DataFrame,
    column: str,
    strategy: Literal["mean", "median", "mode", "knn"],
) -> pd.DataFrame:
    """Imputa los valores faltantes de `column` según la estrategia.

    - "mean" / "median": estadísticos para variables numéricas continuas.
    - "mode": moda, para variables categóricas como `region`.
    - "knn": KNNImputer (n_neighbors=5) usando las demás columnas numéricas;
      solo se escribe de vuelta la columna indicada.
    """
    df = df.copy()

    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        moda = df[column].mode(dropna=True)
        if not moda.empty:
            df[column] = df[column].fillna(moda.iloc[0])
    elif strategy == "knn":
        numericas = df.select_dtypes(include=[np.number]).columns
        # KNNImputer descarta las columnas 100% nulas y cambia la forma de la salida
        usables = [c for c in numericas if df[c].notna().any()]
        if column not in usables:
            return df
        imputado = KNNImputer(n_neighbors=KNN_NEIGHBORS).fit_transform(df[usables])
        df[column] = imputado[:, usables.index(column)]
    else:
        raise ValueError(f"Estrategia de imputación desconocida: {strategy}")

    return df


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    method: Literal["zscore", "iqr"] = "iqr",
    action: Literal["cap"] = "cap",
) -> pd.DataFrame:
    """Detecta outliers en `column` con zscore (3σ) o IQR (1.5·IQR) y los
    acota (capping) a los límites, preservando la cantidad de registros."""
    if action != "cap":
        raise ValueError(f"Acción no soportada: {action}")

    df = df.copy()
    serie = df[column]

    if method == "iqr":
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        rango = q3 - q1
        inferior, superior = q1 - IQR_FACTOR * rango, q3 + IQR_FACTOR * rango
    elif method == "zscore":
        media, desviacion = serie.mean(), serie.std()
        inferior = media - ZSCORE_THRESHOLD * desviacion
        superior = media + ZSCORE_THRESHOLD * desviacion
    else:
        raise ValueError(f"Método de detección desconocido: {method}")

    df[column] = serie.clip(lower=inferior, upper=superior)
    return df


def run_cleaning_pipeline() -> dict:
    """Orquesta: normalizar -> imputar (mediana / moda) -> acotar outliers IQR
    y guarda el resultado en `customer_credit_clean_alvarado`."""
    crudo = extract_raw_data(RAW_QUERY)
    varianza_antes = float(crudo["annual_income"].var())

    limpio = normalize_frame(crudo)
    nulos_tras_normalizar = int(limpio.isna().sum().sum())

    for columna in COLUMNAS_NUMERICAS:
        if columna in limpio.columns:
            limpio = impute_column(limpio, columna, "median")
    if "region" in limpio.columns:
        limpio = impute_column(limpio, "region", "mode")

    for columna in COLUMNAS_OUTLIERS:
        limpio = treat_outliers(limpio, columna, method="iqr", action="cap")

    varianza_despues = float(limpio["annual_income"].var())
    limpio.to_sql(OUTPUT_TABLE, con=get_database_engine(), if_exists="replace", index=False)

    return {
        "tabla": OUTPUT_TABLE,
        "filas_crudas": len(crudo),
        "filas_limpias": len(limpio),
        "nulos_tras_normalizar": nulos_tras_normalizar,
        "nulos_finales": int(limpio.isna().sum().sum()),
        "varianza_income_antes": varianza_antes,
        "varianza_income_despues": varianza_despues,
    }


def main() -> None:
    """Ejecuta el pipeline y muestra la auditoría por consola."""
    r = run_cleaning_pipeline()
    print(f"Tabla '{r['tabla']}' guardada: {r['filas_crudas']} -> {r['filas_limpias']} filas")
    print(f"Nulos tras normalizar: {r['nulos_tras_normalizar']} | nulos finales: {r['nulos_finales']}")
    print(
        f"Varianza annual_income: {r['varianza_income_antes']:,.2f} -> "
        f"{r['varianza_income_despues']:,.2f}"
    )


if __name__ == "__main__":
    main()