param(
    [Parameter(Mandatory = $true)][string]$Front,
    [Parameter(Mandatory = $true)][string]$Left,
    [Parameter(Mandatory = $true)][double]$HeightCm,
    [ValidateSet("male", "female")][string]$GarmentRoute = "male",
    [ValidateSet("tight", "loose", "unknown")][string]$ClothingFit = "unknown",
    [string]$ConfirmedMeasurements,
    [string]$OutputDir = "outputs/assisted_demo",
    [switch]$SkipBody3D
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ModelRegistryPath = Join-Path $ProjectRoot "configs/modelos_runtime.json"
if (-not (Test-Path -LiteralPath $ModelRegistryPath -PathType Leaf)) {
    throw "Falta el registro de modelos: $ModelRegistryPath"
}
$ModelRegistry = Get-Content -LiteralPath $ModelRegistryPath -Raw -Encoding utf8 | ConvertFrom-Json
$PoseModel = Join-Path $ProjectRoot $ModelRegistry.medicion_oficial.pose.artefacto
$SamCheckpoint = Join-Path $ProjectRoot $ModelRegistry.medicion_oficial.segmentacion.artefacto
$SamRoot = Join-Path $ProjectRoot $ModelRegistry.medicion_oficial.segmentacion.codigo
$MeasurementExperiment = [string]$ModelRegistry.medicion_oficial.regresion.directorio
$MeasurementBackend = [string]$ModelRegistry.medicion_oficial.regresion.backend
$MaskDir = Join-Path $ProjectRoot $OutputDir
$Prediction = Join-Path $MaskDir "prediction.json"
$BundleReport = Join-Path $MaskDir "best_configuration.json"
$CondaCandidates = @(
    $env:SATRE_CONDA_EXE,
    (Join-Path $env:USERPROFILE "miniconda3/Scripts/conda.exe"),
    (Join-Path $env:USERPROFILE "anaconda3/Scripts/conda.exe"),
    "C:/ProgramData/miniconda3/Scripts/conda.exe",
    "C:/ProgramData/anaconda3/Scripts/conda.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
$Conda = $CondaCandidates | Select-Object -First 1
if (-not $Conda) {
    $CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($CondaCommand) { $Conda = $CondaCommand.Source }
}
if (-not $Conda) { throw "No se encontró conda. Define SATRE_CONDA_EXE con la ruta de conda.exe." }

foreach ($PathToCheck in @($Front, $Left, $PoseModel, $SamCheckpoint)) {
    if (-not (Test-Path -LiteralPath $PathToCheck -PathType Leaf)) {
        throw "No existe el archivo requerido: $PathToCheck"
    }
}

$SegmentationArgs = @(
    "run", "-n", "sastre-ia-perception", "python", "src/perception/sam2_capture.py",
    "--front", (Resolve-Path -LiteralPath $Front).Path,
    "--left", (Resolve-Path -LiteralPath $Left).Path,
    "--checkpoint", $SamCheckpoint,
    "--sam2-root", $SamRoot,
    "--pose-model", $PoseModel,
    "--output-dir", $MaskDir
)

Push-Location $ProjectRoot
try {
    & $Conda @SegmentationArgs
    if ($LASTEXITCODE -ne 0) { throw "Falló la segmentación automática" }

    $PredictionArgs = @(
        "run", "-n", "sastre-ia", "python", "src/inference/predict_two_views.py",
        "--exp", $MeasurementExperiment,
        "--backend", $MeasurementBackend,
        "--front-mask", (Join-Path $MaskDir "front.png"),
        "--left-mask", (Join-Path $MaskDir "left.png"),
        "--capture-manifest", (Join-Path $MaskDir "capture_manifest.json"),
        "--clothing-fit", $ClothingFit,
        "--height-cm", $HeightCm,
        "--garment-route", $GarmentRoute,
        "--output", $Prediction
    )
    if ($ConfirmedMeasurements) {
        if (-not (Test-Path -LiteralPath $ConfirmedMeasurements -PathType Leaf)) {
            throw "No existe el archivo de medidas confirmadas: $ConfirmedMeasurements"
        }
        $PredictionArgs += @(
            "--confirmed-measurements",
            (Resolve-Path -LiteralPath $ConfirmedMeasurements).Path
        )
    }
    & $Conda @PredictionArgs
    if ($LASTEXITCODE -ne 0) { throw "Falló la estimación de medidas" }

    if (-not $SkipBody3D) {
        Write-Warning "Body3D se omite en la ruta oficial: SHAPY requiere peso y sexo, que ya no son entradas de captura."
    }

    $Bundle = [ordered]@{
        schema_version = "1.0"
        configuration = "sastre_dos_vistas_bodym_v4_registry"
        model_registry = $ModelRegistryPath
        measurements = $Prediction
        body3d = $null
        policy = "exp_008 ExtraTrees sobre perfiles fisicos BodyM; frontal+lateral+estatura; peso estimado; confeccion a medida sin tallas"
    }
    $Bundle | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $BundleReport -Encoding utf8
} finally {
    Pop-Location
}

Write-Host "Prototipo completado. Paquete: $BundleReport"
