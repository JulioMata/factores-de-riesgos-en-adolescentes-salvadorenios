#!/usr/bin/env python
"""CLI de entrada para el pipeline de entrenamiento."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añadir src/ al path cuando se ejecuta directamente
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_project_paths, load_config


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de machine learning (stub)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta opcional al archivo de configuración YAML.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada del script de entrenamiento."""
    args = parse_args()
    paths = get_project_paths()
    config = load_config()

    print(f"Proyecto: {paths.root}")
    print(f"Configuración cargada: {config.get('project', {}).get('name', 'fras')}")

    if args.config:
        print(f"Config personalizada indicada: {args.config}")

    print("\nPipeline de entrenamiento no implementado aún.")
    print("Implementa la lógica en src/models/train.py y conecta aquí.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
