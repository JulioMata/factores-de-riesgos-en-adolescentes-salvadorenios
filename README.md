# Sistema de Predicción de Factores de Riesgo en Adolescentes Salvadoreños (GSHS 2013)


## Requisitos

- Python 3.13+
- PowerShell (Windows)

## Setup rápido

Desde la raíz del proyecto:

```powershell
.\scripts\setup.ps1
```

El script crea `.venv`, instala dependencias, registra el kernel de Jupyter **Python (fras)** y configura `PYTHONPATH` para importar módulos desde `src/`.

## Setup manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m ipykernel install --user --name fras --display-name "Python (fras)"
$env:PYTHONPATH = "$PWD\src"
```

Copia la configuración de ejemplo si aún no tienes una local:

```powershell
Copy-Item configs\config.example.yaml configs\config.yaml
Copy-Item .env.example .env
```

## Estructura del proyecto

```
fras/
├── configs/          # Configuración YAML
├── data/
│   ├── raw/          # Datos originales (no versionados)
│   ├── interim/      # Datos intermedios
│   ├── processed/    # Datos listos para modelado
│   └── external/     # Fuentes externas
├── notebooks/        # Libretas Jupyter
├── src/              # Código fuente
│   ├── config.py     # Configuración y rutas
│   ├── data/         # Carga y preprocesamiento
│   ├── features/     # Feature engineering
│   ├── models/       # Entrenamiento e inferencia
│   └── visualization/# Gráficos
├── models/           # Artefactos entrenados (.joblib, .pkl)
├── reports/figures/  # Figuras generadas
├── scripts/          # Scripts de utilidad y CLI
└── tests/            # Tests
```

## Flujo de trabajo sugerido

1. **Exploración** — Crea libretas en `notebooks/` para analizar datos.
2. **Código reutilizable** — Mueve la lógica estable a `src/`.
3. **Entrenamiento** — Ejecuta el pipeline desde `scripts/train.py`.
4. **Artefactos** — Guarda modelos en `models/` y figuras en `reports/figures/`.

## Importar el paquete

Con `PYTHONPATH` apuntando a `src/` (lo hace `setup.ps1`):

```python
from config import get_project_paths, load_config
from data.load import load_raw_data
```

## Comandos útiles

```powershell
# Activar entorno
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"

# JupyterLab
jupyter lab

# Tests
pytest

# Lint
ruff check src tests

# Entrenamiento (stub)
python scripts/train.py
```

## Licencia

MIT — ver [LICENSE](LICENSE).
