import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.cleaning import (
    cap_outliers,
    impute_column,
    normalize_inconsistencies,
    run_cleaning_pipeline,
)


def test_normalization_and_imputation_keep_rows():
    raw = pd.DataFrame(
        {
            "region": ["  cOSTA  ", None, "SIERRA", "   "],
            "age": [17, 34, 101, None],
            "annual_income": [100.0, None, 200.0, 300.0],
        }
    )
    normalized = normalize_inconsistencies(raw)
    assert normalized["region"].iloc[0] == "Costa"
    assert normalized["region"].iloc[2] == "Sierra"
    assert normalized["age"].isna().sum() == 3
    assert raw["age"].iloc[0] == 17

    filled = impute_column(normalized, "region", "mode")
    filled = impute_column(filled, "age", "median")
    filled = impute_column(filled, "annual_income", "mean")
    assert len(filled) == len(raw)
    assert filled[["region", "age", "annual_income"]].isna().sum().sum() == 0
    assert filled["annual_income"].iloc[1] == pytest.approx(200.0)


def test_capping_reduces_extreme_value_without_dropping_rows():
    raw = pd.DataFrame({"annual_income": [10, 11, 12, 13, 14, 15, 1000]})
    cleaned = cap_outliers(raw, "annual_income")
    assert len(cleaned) == len(raw)
    assert cleaned["annual_income"].max() < 1000
    assert cleaned["annual_income"].var() < raw["annual_income"].var()
    assert raw["annual_income"].iloc[-1] == 1000


def test_pipeline_persists_clean_table_and_reports_variance(capsys):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    raw = pd.DataFrame(
        {
            "transaction_id": range(1, 10),
            "customer_id": [f"CUST-{i}" for i in range(1, 10)],
            "age": [20, 30, 40, 50, 60, 70, 120, None, 35],
            "annual_income": [10, 11, 12, 13, 14, 15, 16, None, 1000],
            "credit_score": [600, 610, None, 630, 640, 650, 660, 670, 680],
            "loan_amount": [1, 2, 3, 4, 5, 6, 7, 500, None],
            "region": [" costa ", "SIERRA", "Costa", "Costa", None,
                       "Sierra", "Costa", "Sierra", "Costa"],
        }
    )
    raw.to_sql("customer_credit_transactions", engine, index=False)

    cleaned = run_cleaning_pipeline(engine)
    with engine.connect() as connection:
        stored = pd.read_sql_query(text("SELECT * FROM customer_credit_clean"), connection)
        original = pd.read_sql_query(
            text("SELECT * FROM customer_credit_transactions"), connection
        )

    assert len(stored) == len(original) == len(raw)
    assert stored.isna().sum().sum() == 0
    assert cleaned["annual_income"].var() < raw["annual_income"].var()
    assert stored["region"].iloc[0] == "Costa"
    assert original["annual_income"].iloc[-1] == 1000
    output = capsys.readouterr().out
    assert "Varianza de annual_income antes:" in output
    assert "Varianza de annual_income después:" in output
