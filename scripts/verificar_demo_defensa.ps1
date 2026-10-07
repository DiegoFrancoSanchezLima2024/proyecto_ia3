param([switch]$Completa)

$ErrorActionPreference = "Stop"
$RaizProyecto = Split-Path -Parent $PSScriptRoot
$PythonMedicion = "C:\Users\diego\miniconda3\envs\sastre-ia\python.exe"
$PythonVestidor = "C:\Users\diego\miniconda3\envs\sastre-ia-fashn\python.exe"

function Confirmar-Archivo([string]$RutaRelativa, [string]$Nombre) {
    $RutaAbsoluta = Join-Path $RaizProyecto $RutaRelativa
    if (-not (Test-Path -LiteralPath $RutaAbsoluta -PathType Leaf)) {
        throw "Falta $Nombre`: $RutaRelativa"
    }
    $TamanoMB = [math]::Round((Get-Item -LiteralPath $RutaAbsoluta).Length / 1MB, 1)
    Write-Host "  OK  $Nombre ($TamanoMB MB)" -ForegroundColor Green
}

Write-Host "SATRE-IA | verificación previa de defensa" -ForegroundColor Cyan
Set-Location -LiteralPath $RaizProyecto

$Registro = Get-Content -LiteralPath "configs/modelos_runtime.json" -Raw -Encoding utf8 | ConvertFrom-Json
if ($Registro.medicion_oficial.regresion.nombre -ne "exp_008_perfiles_arboles") {
    throw "El registro no selecciona exp_008 como regresor oficial."
}

Confirmar-Archivo "pretrained/mediapipe/pose_landmarker_lite.task" "MediaPipe Pose Lite"
Confirmar-Archivo "pretrained/sam2/sam2.1_hiera_small.pt" "SAM 2.1 Small"
Confirmar-Archivo "experiments/exp_008_perfiles_arboles/model.joblib" "ExtraTrees exp_008"
Confirmar-Archivo "experiments/exp_008_perfiles_arboles/conformal_q_hat.npy" "calibración conformal"
Confirmar-Archivo "pretrained/schp-atr/onnx/schp-atr-18-int8-static.onnx" "SCHP-ATR"
Confirmar-Archivo "pretrained/fashn-vton-1.5/model.safetensors" "FASHN VTON 1.5"
Confirmar-Archivo "pretrained/fashn-vton-1.5/dwpose/yolox_l.onnx" "detector DWPose"
Confirmar-Archivo "pretrained/fashn-vton-1.5/dwpose/dw-ll_ucoco_384.onnx" "pose DWPose"
Confirmar-Archivo "pretrained/stable-diffusion-inpainting/unet/diffusion_pytorch_model.fp16.safetensors" "SD 1.5 Inpainting FP16"
Confirmar-Archivo "pattern-engine/node_modules/@freesewing/jaeger/package.json" "FreeSewing Jaeger"
Confirmar-Archivo "pattern-engine/node_modules/@freesewing/charlie/package.json" "FreeSewing Charlie"
Confirmar-Archivo "pattern-engine/node_modules/@freesewing/penelope/package.json" "FreeSewing Penelope"

foreach ($Python in @($PythonMedicion, $PythonVestidor)) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Falta el entorno Python requerido: $Python"
    }
}

& $PythonVestidor -c "import torch; assert torch.cuda.is_available(); print('  OK  CUDA:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA no está disponible en el entorno del vestidor." }

& node --check "pattern-engine/server.mjs"
if ($LASTEXITCODE -ne 0) { throw "server.mjs no supera la validación de sintaxis." }

& $PythonMedicion "scripts/auditar_modelos_runtime.py"
if ($LASTEXITCODE -ne 0) { throw "Falló la auditoría de modelos activos." }

if ($Completa) {
    Write-Host "Ejecutando pruebas Python..." -ForegroundColor Cyan
    & $PythonMedicion -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Fallaron las pruebas Python." }

    Write-Host "Ejecutando pruebas de patronaje/API..." -ForegroundColor Cyan
    Push-Location "pattern-engine"
    try { & npm test } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "Fallaron las pruebas Node/FreeSewing." }
}

Write-Host "VERIFICACIÓN APROBADA: medición, patronaje y vestidor tienen sus artefactos locales." -ForegroundColor Green
Write-Host "Los modelos históricos permanecen fuera del runtime oficial." -ForegroundColor DarkGray
