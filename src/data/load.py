"""Carga de datasets desde data/raw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_RAW_FILENAME, MISSING_SENTINEL, get_project_paths


def replace_missing_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza el valor centinela de SPSS/Stata por np.nan.

    El valor 1.79769313486232e+308 es el máximo de float64 y representa
    datos faltantes en encuestas procesadas con SPSS/Stata.
    """
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned.loc[cleaned[col] >= MISSING_SENTINEL * 0.99, col] = np.nan
    return cleaned


def load_csv(path: Path) -> pd.DataFrame:
    """Carga un archivo CSV desde la ruta indicada."""
    return pd.read_csv(path)


def load_raw_data(filename: str | None = None) -> pd.DataFrame:
    """
    Carga el dataset GSHS desde data/raw y limpia el centinela de nulos.

    Args:
        filename: Nombre del archivo CSV. Por defecto SLV2013_Public_Use.csv.

    Returns:
        DataFrame con los datos cargados y nulos centinela reemplazados.
    """
    paths = get_project_paths()
    csv_name = filename or DEFAULT_RAW_FILENAME
    csv_path = paths.data_raw / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")

    df = load_csv(csv_path)
    return replace_missing_sentinel(df)
