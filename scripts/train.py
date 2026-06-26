#!/usr/bin/env python
"""CLI de entrada para el pipeline de entrenamiento GSHS 2013."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import TARGET_IMC, TARGET_MENTAL_HEALTH, get_project_paths, load_config
from data.load import load_raw_data
from data.preprocess import preprocess_data
from features.build import build_features, get_target
from models.predict import save_model
from models.train import (
    evaluate_classification,
    evaluate_regression,
    select_best_classification,
    select_best_regression,
    tune_classification_model,
    tune_regression_model,
)
from sklearn.model_selection import train_test_split
from visualization.plots import (
    plot_age_distribution,
    plot_bmi_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_mental_health_prevalence,
    plot_residuals,
    plot_roc_curve,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline GSHS 2013: regresión IMC + clasificación riesgo salud mental."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta opcional al archivo de configuración YAML.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Omitir ajuste de hiperparámetros (más rápido).",
    )
    return parser.parse_args()


def print_regression_summary(results: list) -> None:
    print("\n=== REGRESIÓN IMC ===")
    for r in results:
        print(
            f"  {r.model_name}: RMSE={r.rmse:.3f} | R²={r.r2:.3f} | "
            f"CV RMSE={r.cv_rmse_mean:.3f}±{r.cv_rmse_std:.3f} | "
            f"CV R²={r.cv_r2_mean:.3f}±{r.cv_r2_std:.3f}"
        )


def print_classification_summary(results: list) -> None:
    print("\n=== CLASIFICACIÓN RIESGO SALUD MENTAL ===")
    for r in results:
        print(
            f"  {r.model_name}: F1(minoritaria)={r.f1_minority:.3f} | "
            f"AUC-ROC={r.auc_roc:.3f} | Accuracy={r.accuracy:.3f} | "
            f"CV F1={r.cv_f1_mean:.3f}±{r.cv_f1_std:.3f}"
        )


def main() -> int:
    """Ejecuta el pipeline completo de entrenamiento y evaluación."""
    args = parse_args()
    paths = get_project_paths()
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    test_size = config.get("training", {}).get("test_size", 0.2)
    cv_folds = config.get("training", {}).get("cross_validation_folds", 5)

    set_plot_style()
    figures_dir = paths.reports_figures
    models_dir = paths.models
    metrics_path = paths.reports / "metrics.json"

    print(f"Proyecto: {paths.root}")
    print("Cargando datos GSHS 2013...")
    df_raw = load_raw_data()
    print(f"  Registros cargados: {len(df_raw):,}")

    print("Preprocesando...")
    df = preprocess_data(df_raw)
    processed_path = paths.data_processed / "gshs_processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Datos procesados guardados en: {processed_path}")

    # EDA visualizaciones
    print("Generando visualizaciones EDA...")
    plot_age_distribution(df, figures_dir / "eda_age_distribution.png")
    plot_bmi_distribution(df, figures_dir / "eda_bmi_distribution.png")
    plot_mental_health_prevalence(df, figures_dir / "eda_mental_health_prevalence.png")

    valid_imc = df[TARGET_IMC].notna().sum()
    valid_risk = df[TARGET_MENTAL_HEALTH].notna().sum()
    risk_rate = df[TARGET_MENTAL_HEALTH].mean()
    print(f"  IMC válidos: {valid_imc:,} | Riesgo mental válidos: {valid_risk:,}")
    print(f"  Prevalencia riesgo mental: {risk_rate:.1%}")

    # --- Regresión IMC ---
    print("\nEntrenando modelos de regresión (IMC)...")
    X_reg, reg_cols, reg_preprocessor = build_features(df, "regression")
    y_reg = get_target(df, "regression")
    reg_results = evaluate_regression(
        X_reg, y_reg, reg_preprocessor, test_size=test_size, cv_folds=cv_folds, random_state=seed
    )
    print_regression_summary(reg_results)
    best_reg = select_best_regression(reg_results)
    print(f"  Mejor modelo regresión: {best_reg.model_name}")

    save_model(best_reg.best_model, models_dir / "regression_imc.joblib")
    plot_residuals(
        best_reg.y_test,
        best_reg.y_pred,
        figures_dir / "regression_residuals.png",
    )

    if not args.skip_tuning:
        mask = y_reg.notna()
        X_tune = X_reg.loc[mask]
        y_tune = y_reg.loc[mask]
        X_tr, _, y_tr, _ = train_test_split(X_tune, y_tune, test_size=test_size, random_state=seed)
        tuned_reg, reg_params = tune_regression_model(X_tr, y_tr, reg_preprocessor, seed)
        save_model(tuned_reg, models_dir / "regression_imc_tuned.joblib")
        print(f"  Hiperparámetros regresión (RF): {reg_params}")

    # --- Clasificación ---
    print("\nEntrenando modelos de clasificación (riesgo salud mental)...")
    X_clf, clf_cols, clf_preprocessor = build_features(df, "classification")
    y_clf = get_target(df, "classification")
    clf_results = evaluate_classification(
        X_clf, y_clf, clf_preprocessor, test_size=test_size, cv_folds=cv_folds, random_state=seed
    )
    print_classification_summary(clf_results)
    best_clf = select_best_classification(clf_results)
    print(f"  Mejor modelo clasificación: {best_clf.model_name}")
    print(f"\n{best_clf.classification_report}")

    save_model(best_clf.best_model, models_dir / "classification_mental_health.joblib")
    plot_confusion_matrix(
        best_clf.confusion,
        figures_dir / "classification_confusion_matrix.png",
    )
    if best_clf.fpr is not None and best_clf.tpr is not None:
        plot_roc_curve(
            best_clf.fpr,
            best_clf.tpr,
            best_clf.auc_roc,
            figures_dir / "classification_roc_curve.png",
        )
    if best_clf.feature_importances:
        plot_feature_importance(
            best_clf.feature_importances,
            figures_dir / "classification_feature_importance.png",
            title="Predictores de riesgo en salud mental — Top 15",
        )

    if not args.skip_tuning:
        mask = y_clf.notna()
        X_tune = X_clf.loc[mask]
        y_tune = y_clf.loc[mask].astype(int)
        X_tr, _, y_tr, _ = train_test_split(
            X_tune, y_tune, test_size=test_size, random_state=seed, stratify=y_tune
        )
        tuned_clf, clf_params = tune_classification_model(X_tr, y_tr, clf_preprocessor, seed)
        save_model(tuned_clf, models_dir / "classification_mental_health_tuned.joblib")
        print(f"  Hiperparámetros clasificación (RF+SMOTE): {clf_params}")

    metrics = {
        "regression": {
            "best_model": best_reg.model_name,
            "rmse": best_reg.rmse,
            "r2": best_reg.r2,
            "cv_rmse_mean": best_reg.cv_rmse_mean,
            "cv_r2_mean": best_reg.cv_r2_mean,
            "feature_columns": reg_cols,
        },
        "classification": {
            "best_model": best_clf.model_name,
            "f1_minority": best_clf.f1_minority,
            "auc_roc": best_clf.auc_roc,
            "accuracy": best_clf.accuracy,
            "cv_f1_mean": best_clf.cv_f1_mean,
            "cv_auc_mean": best_clf.cv_auc_mean,
            "feature_columns": clf_cols,
            "top_features": best_clf.feature_importances,
        },
        "eda": {
            "n_records": len(df),
            "valid_imc": int(valid_imc),
            "valid_mental_health": int(valid_risk),
            "mental_health_prevalence": float(risk_rate) if not np.isnan(risk_rate) else None,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMétricas guardadas en: {metrics_path}")
    print(f"Modelos en: {models_dir}")
    print(f"Figuras en: {figures_dir}")
    print("\nPipeline completado exitosamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
