"""Parte B2 (bug hunting): casos borde de `src.cleaning` que los tests originales no cubren.

Cada test se escribió ANTES del arreglo y falla con el código original:
  1. `impute_column(..., "mode")` sobre una columna 100% nula lanza IndexError.
  2. `impute_column(..., "knn")` con OTRA columna numérica 100% nula lanza ValueError
     (KNNImputer descarta esa columna y la forma de la salida ya no coincide).
  3. `impute_column(..., "knn")` sobre la propia columna 100% nula tampoco debe fallar.
"""
import numpy as np
import pandas as pd

from src.cleaning import impute_column


def _frame_numerico() -> pd.DataFrame:
    """Frame pequeño con nulos en `age` y las 4 columnas numéricas del pipeline."""
    return pd.DataFrame(
        {
            "age": [30.0, 40.0, np.nan, 25.0, 50.0, 35.0, 28.0, 42.0],
            "annual_income": [30_000.0, 45_000, 38_000, 22_000, 60_000, 38_500, 25_000, 52_000],
            "credit_score": [700.0, 650, 640, 720, 600, 690, 710, 630],
            "loan_amount": [10_000.0, 12_000, 9_000, 8_000, 15_000, 11_000, 9_500, 13_000],
        }
    )


def test_mode_con_columna_100_nula_no_lanza_error():
    """Una `region` toda vacía debe quedar igual (sin moda posible), no romper el pipeline."""
    df = pd.DataFrame({"region": [None, None, None], "age": [30.0, 40.0, 50.0]})
    out, rep = impute_column(df, "region", "mode")
    assert out["region"].isna().all()
    assert rep["imputed"] == 0  # no se imputó nada: el reporte no debe mentir


def test_knn_con_otra_columna_100_nula_imputa_la_pedida():
    """Si `credit_score` llega 100% nula, KNN debe seguir imputando `age`."""
    df = _frame_numerico()
    df["credit_score"] = np.nan
    out, rep = impute_column(df, "age", "knn")
    assert out["age"].isna().sum() == 0
    assert rep["imputed"] == 1
    assert out["credit_score"].isna().all()  # la columna vacía se conserva


def test_knn_sobre_la_propia_columna_100_nula_no_falla():
    """Pedir KNN sobre una columna sin ningún dato no debe lanzar excepción."""
    df = _frame_numerico()
    df["credit_score"] = np.nan
    out, rep = impute_column(df, "credit_score", "knn")
    assert out["credit_score"].isna().all()
    assert rep["imputed"] == 0
