"""Inferencia con modelos entrenados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(model: Any, path: Path) -> None:
    """Guarda un modelo entrenado en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Carga un modelo serializado desde disco."""
    return joblib.load(path)


def predict(model_path: Path, X: pd.DataFrame) -> pd.Series:
    """
    Genera predicciones con un modelo entrenado.

    Args:
        model_path: Ruta al artefacto del modelo (.joblib).
        X: Features para inferencia (mismas columnas que en entrenamiento).

    Returns:
        Serie con las predicciones.
    """
    pipeline = load_model(model_path)
    predictions = pipeline.predict(X)
    return pd.Series(predictions, index=X.index, name="prediction")
