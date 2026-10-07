param([switch]$MostrarTamano)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
$Registro = Get-Content -LiteralPath (Join-Path $Raiz "configs/modelos_runtime.json") -Raw -Encoding utf8 | ConvertFrom-Json

$Filas = foreach ($Grupo in $Registro.excluidos_del_runtime_oficial) {
    foreach ($RutaRelativa in $Grupo.rutas) {
        $Ruta = Join-Path $Raiz $RutaRelativa
        $Bytes = 0
        if ($MostrarTamano -and (Test-Path -LiteralPath $Ruta)) {
            if (Test-Path -LiteralPath $Ruta -PathType Container) {
                $Bytes = (Get-ChildItem -LiteralPath $Ruta -Recurse -File -ErrorAction SilentlyContinue |
                    Measure-Object Length -Sum).Sum
            } else {
                $Bytes = (Get-Item -LiteralPath $Ruta).Length
            }
        }
        [PSCustomObject]@{
            Componente = $Grupo.nombre
            Ruta = $RutaRelativa
            Presente = Test-Path -LiteralPath $Ruta
            GB = if ($MostrarTamano) { [math]::Round($Bytes / 1GB, 3) } else { $null }
            Motivo = $Grupo.motivo
        }
    }
}

$Filas | Format-Table -AutoSize -Wrap
Write-Host "No se borro ningun archivo. Esta lista separa runtime oficial de evidencia experimental."
