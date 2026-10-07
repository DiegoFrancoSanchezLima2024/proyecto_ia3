param(
    [switch]$ConVestidor,
    [switch]$VerificacionCompleta
)

$ErrorActionPreference = "Stop"
$RaizProyecto = Split-Path -Parent $PSScriptRoot
$Verificador = Join-Path $PSScriptRoot "verificar_demo_defensa.ps1"

if ($VerificacionCompleta) {
    & $Verificador -Completa
} else {
    & $Verificador
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:SATRE_ENABLE_EXPERIMENTAL_TRYON = if ($ConVestidor) { "1" } else { "0" }
$env:HF_HOME = Join-Path $RaizProyecto "pretrained/hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Iniciando SATRE-IA en http://127.0.0.1:8765/" -ForegroundColor Cyan
Write-Host "Vestidor generativo: $(if ($ConVestidor) { 'HABILITADO (experimental)' } else { 'deshabilitado' })"
Push-Location (Join-Path $RaizProyecto "pattern-engine")
try {
    & npm start
} finally {
    Pop-Location
}
