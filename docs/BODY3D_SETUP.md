# Preparación de Body3D (SMPL-X + SHAPY)

## Estado verificado

El proyecto valida frente/perfil izquierdo/perfil derecho, usa estatura como
ancla métrica y genera una forma SMPL-X experimental con SHAPY. El entorno
Windows aislado y el checkpoint oficial están operativos en GPU.

La instalación actual contiene correctamente:

- los modelos SMPL-X masculino, femenino y neutral;
- el checkpoint oficial `SHAPY_A/checkpoints/best_checkpoint`;
- el código fuente de SHAPY en `pretrained/body3d/shapy-master`.

Los tres modelos SMPL-X se instanciaron correctamente: cada uno produce 10.475
vértices y 20.908 caras. Esto valida el formato y la ubicación de esos archivos.

## Descargas que debe realizar el usuario

Los portales oficiales requieren leer y aceptar sus condiciones. Codex no debe
aceptarlas ni descargar los modelos en nombre del usuario.

1. SMPL-X: <https://smpl-x.is.tue.mpg.de/>
2. SHAPY: <https://shapy.is.tue.mpg.de/>

Para investigación, descargar los modelos masculino, femenino y neutral. Antes
de cualquier uso comercial, revisar o solicitar la licencia comercial indicada
por los autores.

## Ubicación esperada

Descomprimir los modelos de manera que existan:

```text
pretrained/body3d/models/smplx/SMPLX_MALE.npz
pretrained/body3d/models/smplx/SMPLX_FEMALE.npz
pretrained/body3d/models/smplx/SMPLX_NEUTRAL.npz
pretrained/body3d/shapy.ckpt
```

También se reconoce directamente la estructura original del paquete oficial:

```text
pretrained/body3d/models/smplx/models/smplx/SMPLX_*.npz
pretrained/body3d/trained_models/shapy/SHAPY_A/checkpoints/best_checkpoint
pretrained/body3d/shapy-master/regressor/human_shape/
```

No es necesario duplicar ni renombrar archivos si se conserva esa estructura.

Si el checkpoint oficial tiene otro nombre, conservar el original y actualizar
la ruta del adaptador cuando se implemente; no modificar el contenido binario.

## Comprobación

```powershell
conda activate sastre-ia
python scripts/check_body3d_readiness.py
```

El preflight separa tres estados:

- `assets_ready`: modelos, checkpoint y código fuente presentes;
- `runtime_ready`: dependencias importables en el Python que ejecutó el comando;
- `ready`: ambos estados anteriores son verdaderos.

Es importante ejecutar el script con el Python del entorno correcto. En esta
máquina puede hacerse directamente con:

```powershell
& 'C:\Users\diego\miniconda3\envs\sastre-ia\python.exe' scripts/check_body3d_readiness.py
```

No deben instalarse en `sastre-ia` las versiones antiguas fijadas en el
`requirements.txt` original de SHAPY: podrían romper MediaPipe, SAM2 o PyTorch.
La integración usa el entorno independiente `sastre-ia-body3d`.

Crear y verificar ese entorno:

```powershell
& 'C:\Users\diego\miniconda3\Scripts\conda.exe' env create `
  -f envs\environment-body3d.yml
& 'C:\Users\diego\miniconda3\envs\sastre-ia-body3d\python.exe' `
  scripts\verify_body3d_env.py
```

### Nota para Windows

El `regressor/demo.py` oficial importa el módulo Unix `resource`, que no existe
en Windows. Por eso el preflight indica `custom_windows_runner`: reutilizaremos
el modelo y checkpoint oficiales mediante un ejecutor propio, sin lanzar ese
archivo de demostración directamente.

Cuando `ready` sea verdadero, la etapa siguiente conectará:

1. inicialización SHAPY por vista;
2. parámetros de forma compartidos entre las vistas;
3. pose y cámara independientes por fotografía;
4. pérdidas de landmarks, silueta, estatura, peso y prior de forma;
5. extracción de las 13 medidas sobre la malla escalada;
6. fusión con `exp_006` y confianza de captura.

## Captura recomendada

- ropa ceñida y no reflectante;
- pies completos y brazos ligeramente separados;
- cámara inmóvil, persona girando;
- frente, izquierda, derecha y, si es posible, espalda;
- misma iluminación y distancia aproximada;
- estatura y peso reales introducidos en la interfaz.

El 3D no debe presentarse como capaz de ver el cuerpo oculto por ropa holgada.

## Inferencia validada con una captura

```powershell
& 'C:\Users\diego\miniconda3\Scripts\conda.exe' run --no-capture-output `
  -n sastre-ia-body3d python scripts\run_shapy_capture.py `
  --manifest outputs\demo_002_masks\capture_manifest.json `
  --height-cm 164 --weight-kg 69 --sex male `
  --output-dir outputs\body3d\demo_002
```

Produce `body_shape.ply`, `body_shape.npz`, una vista previa y un reporte de
consistencia entre vistas. La estatura escala la malla. Peso y sexo alimentan
`exp_006` y el filtro de talla; todavía no refinan directamente el SHAPY neutral.
