# Configura el entorno de desarrollo para el proyecto fras.
# Uso: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Proyecto: $ProjectRoot"

# Crear entorno virtual si no existe
if (-not (Test-Path ".venv")) {
    Write-Host "Creando entorno virtual .venv ..."
    python -m venv .venv
} else {
    Write-Host "Entorno virtual .venv ya existe."
}

# Activar entorno virtual
Write-Host "Activando entorno virtual ..."
& ".\.venv\Scripts\Activate.ps1"

# Actualizar pip e instalar dependencias
Write-Host "Instalando dependencias ..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Registrar kernel de Jupyter
Write-Host "Registrando kernel Jupyter 'fras' ..."
python -m ipykernel install --user --name fras --display-name "Python (fras)"

# Configurar PYTHONPATH para importar módulos desde src/
$SrcPath = Join-Path $ProjectRoot "src"
$env:PYTHONPATH = $SrcPath
Write-Host "PYTHONPATH configurado: $env:PYTHONPATH"

# Crear archivos de configuración local si no existen
if (-not (Test-Path "configs\config.yaml")) {
    Copy-Item "configs\config.example.yaml" "configs\config.yaml"
    Write-Host "Creado configs\config.yaml desde plantilla."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Creado .env desde plantilla."
}

Write-Host ""
Write-Host "Setup completado." -ForegroundColor Green
Write-Host "Para activar manualmente en una nueva terminal:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  `$env:PYTHONPATH = `"`$PWD\src`""
Write-Host ""
Write-Host "Comandos utiles:"
Write-Host "  jupyter lab"
Write-Host "  pytest"
Write-Host "  ruff check src tests"
