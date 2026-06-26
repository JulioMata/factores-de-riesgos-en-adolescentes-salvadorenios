"""Tests de carga, preprocesamiento y feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import MISSING_SENTINEL, TARGET_IMC, TARGET_MENTAL_HEALTH
from data.load import replace_missing_sentinel
from data.preprocess import build_mental_health_target, compute_bmi, preprocess_data
from features.build import get_feature_columns


@pytest.fixture
def sample_gshs_df() -> pd.DataFrame:
    """Mini dataset con estructura GSHS para pruebas unitarias."""
    return pd.DataFrame(
        {
            "Q1": [4, 5, 3],
            "Q2": [1, 2, 1],
            "Q4": [1.68, 1.55, MISSING_SENTINEL],
            "Q5": [43.0, 36.0, 40.0],
            "QN6": [1, 2, 1],
            "QN25": [2, 1, 2],
            "QN21": [2, 1, 1],
            "QN34": [1, 2, 1],
            "QN48": [2, 1, 2],
            "qnpa7g": [2, 1, 2],
            "weight": [100.0, 200.0, 150.0],
            "stratum": [1, 2, 3],
            "psu": [10, 20, 30],
        }
    )


def test_replace_missing_sentinel() -> None:
    df = pd.DataFrame({"a": [1.0, MISSING_SENTINEL, 3.0]})
    cleaned = replace_missing_sentinel(df)
    assert pd.isna(cleaned.loc[1, "a"])
    assert cleaned.loc[0, "a"] == 1.0


def test_compute_bmi() -> None:
    weight = pd.Series([43.0, 36.0])
    height = pd.Series([1.68, 1.55])
    bmi = compute_bmi(weight, height)
    assert pytest.approx(bmi.iloc[0], rel=1e-3) == 43.0 / (1.68**2)
    assert pytest.approx(bmi.iloc[1], rel=1e-3) == 36.0 / (1.55**2)


def test_build_mental_health_target() -> None:
    df = pd.DataFrame({"QN25": [1, 2, np.nan, 3]})
    target = build_mental_health_target(df)
    assert target.iloc[0] == 1.0
    assert target.iloc[1] == 0.0
    assert pd.isna(target.iloc[2])
    assert pd.isna(target.iloc[3])


def test_preprocess_data_creates_targets(sample_gshs_df: pd.DataFrame) -> None:
    processed = preprocess_data(sample_gshs_df)
    assert TARGET_IMC in processed.columns
    assert TARGET_MENTAL_HEALTH in processed.columns
    assert pd.isna(processed.loc[2, TARGET_IMC])  # altura centinela


def test_regression_features_exclude_leakage(sample_gshs_df: pd.DataFrame) -> None:
    processed = preprocess_data(sample_gshs_df)
    cols = get_feature_columns(processed, "regression")
    assert "Q4" not in cols
    assert "Q5" not in cols
    assert "weight" not in cols
    assert TARGET_IMC not in cols


def test_classification_features_exclude_mental_health_leakage(
    sample_gshs_df: pd.DataFrame,
) -> None:
    processed = preprocess_data(sample_gshs_df)
    cols = get_feature_columns(processed, "classification")
    assert "QN25" not in cols
    assert "QN21" not in cols
    assert TARGET_MENTAL_HEALTH not in cols
