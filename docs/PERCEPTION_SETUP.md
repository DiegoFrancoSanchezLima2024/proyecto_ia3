# Percepción RGB con SAM 2.1

Este módulo convierte cuatro fotos reales en máscaras, sin entrenar ni etiquetar
un dataset propio. Se mantiene fuera del entorno `sastre-ia` porque SAM 2 exige
PyTorch 2.5.1 o superior; el campeón `exp_006` usa PyTorch 2.3.1.

## Instalación

El repositorio oficial recomienda WSL con Ubuntu para Windows. Si se usa Windows
nativo, la extensión CUDA opcional puede no compilar; la documentación oficial
indica que la inferencia normalmente sigue funcionando con postproceso limitado.

```powershell
conda env create -f envs/environment-perception.yml
conda activate sastre-ia-perception
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
python -m pip install -e vendor/sam2
```

MediaPipe se usa para localizar automáticamente a la persona y validar que las
articulaciones principales sean visibles. Descarga el modelo oficial Lite en:

```text
pretrained/mediapipe/pose_landmarker_lite.task
```

Descargar solamente el checkpoint Small oficial y guardarlo en:

```text
pretrained/sam2/sam2.1_hiera_small.pt
```

Enlace oficial:
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

## Cajas de persona

La ruta recomendada usa `--pose-model` para que MediaPipe produzca la caja. Ya no
es necesario escribir coordenadas manuales.

Para mayor precisión se puede crear `boxes.json` con coordenadas `x1,y1,x2,y2`:

```json
{
  "front": [100, 20, 900, 1900],
  "left": [120, 20, 850, 1900],
  "back": [100, 20, 900, 1900],
  "right": [120, 20, 850, 1900]
}
```

Sin este archivo se usa la fotografía completa como prompt y el resultado queda
marcado para revisión visual.

## Segmentación

```powershell
python src/perception/sam2_capture.py `
  --front fotos/front.jpg --left fotos/left.jpg `
  --back fotos/back.jpg --right fotos/right.jpg `
  --checkpoint pretrained/sam2/sam2.1_hiera_small.pt `
  --sam2-root vendor/sam2 `
  --pose-model pretrained/mediapipe/pose_landmarker_lite.task
```

El comando genera cuatro PNG y `capture_manifest.json` en
`outputs/capture_masks`.

## Predicción métrica

Volver al entorno estable:

```powershell
conda activate sastre-ia
python src/inference/predict_four_views.py `
  --front-mask outputs/capture_masks/front.png `
  --left-mask outputs/capture_masks/left.png `
  --back-mask outputs/capture_masks/back.png `
  --right-mask outputs/capture_masks/right.png `
  --capture-manifest outputs/capture_masks/capture_manifest.json `
  --clothing-fit tight `
  --height-cm 171 --weight-kg 70 --gender 1
```

Código usado por BodyM: `--gender 0` para mujer y `--gender 1` para hombre.

Las máscaras deben revisarse antes de aceptar centímetros. SAM 2 segmenta la
superficie visible, incluida la ropa; por ello la captura debe usar ropa ceñida.

## Un solo comando para el prototipo

La espalda es opcional. El comando ejecuta percepción y medición en sus entornos
aislados:

```powershell
.\scripts\run_assisted_demo.ps1 `
  -Front fotos\front.jpg -Left fotos\left.jpg -Right fotos\right.jpg `
  -HeightCm 171 -WeightKg 70 -Gender 1 -ClothingFit tight
```
