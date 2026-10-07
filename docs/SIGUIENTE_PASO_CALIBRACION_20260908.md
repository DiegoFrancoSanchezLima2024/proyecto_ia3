# Qué sigue — selección conservada y estación métrica

Actualizado el 8 de septiembre de 2026.

## Lo que ya está hecho

Se conservaron copias exactas de 30 JPG en `dataset/local/calibration/infinix_x6876_20260908`:

- `train`: las 13 candidatas del primer lote, seleccionadas antes de evaluar el segundo.
- `validation`: las 17 verticales del segundo lote, sin quitar vistas por su residuo.
- 28 son preferidas; las dos marcadas para revisión permanecen en validación y en el registro. No son 30 fotos aprobadas.
- `manifest.json`: registra las 48 fotos revisadas, selección, procedencia, hashes de archivo y píxeles y las dos vistas problemáticas. Las 18 no seleccionadas no se copiaron y no se borraron de su ubicación original.
- Las fotos de validación no se usaron para ajustar la lente. Si se incorporan a un ajuste nuevo, ya no serán su evaluación independiente.

La ejecución real sobre estas copias reprodujo el diagnóstico anterior: RMS de ajuste 0,828 px; máximo en validación 1,637 px frente al límite provisional 1,50 px. El perfil sigue en `needs_recapture`; no se modificó el umbral ni se instaló como cámara aprobada.

Perfil de diagnóstico: `outputs/calibration/infinix_x6876_20260908_diagnostic.json`. El comando métrico `floor` continúa rechazándolo. No hay todavía precisión en centímetros validada.

## Decisión sobre data augmentation

No se aplica a las imágenes que estiman o validan la lente. No se usa IA generativa, reenfoque, reescalado, recorte, rotación, corrección de perspectiva ni “aplanado” digital del tablero.

OpenCV calibra a partir de correspondencias entre puntos del tablero y su proyección en distintas vistas reales. [Documentación oficial de calibración ChArUco](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html).

Nuestra conclusión técnica para este conjunto: las variantes artificiales de una misma foto no añaden observaciones independientes de la cámara; una transformación geométrica además cambia sus coordenadas y el modelo que habría que usar. El aumento fotométrico puede servir en otro momento para ensayar robustez del detector, separado de calibración y validación, pero no soluciona este residuo ni certifica escala física.

No hay una red corporal que entrenar con estas fotos del tablero. No se relanza BodyM ni se altera exp_006.

## Próxima captura útil: una foto del suelo con los cuatro ArUco

Podemos preparar y comprobar la visibilidad del montaje mientras sigue pendiente cerrar la lente. No hace falta otra tanda grande de ChArUco ahora. La aprobación de la lente sigue siendo requisito para obtener pose/medidas.

Usar las otras cuatro páginas del kit, las de ArUco 30, 31, 32 y 33, no cuatro copias del tablero de ajedrez. Si ya están impresas, reutilizarlas. [Kit existente de cinco páginas](<C:/Users/diego/Desktop/proyecto de ia3/proyecto-sastre-ia/outputs/calibration_kit/imprimir.html>); las páginas 2 a 5 son las del suelo.

Montaje sobre un mismo suelo plano, visto desde la cámara:

| Posición | ID | Centro respecto del centro del montaje |
| --- | --- | --- |
| Fondo izquierdo | 30 | 42 cm izquierda, 32 cm al fondo |
| Fondo derecho | 31 | 42 cm derecha, 32 cm al fondo |
| Frente izquierdo | 32 | 42 cm izquierda, 32 cm hacia la cámara |
| Frente derecho | 33 | 42 cm derecha, 32 cm hacia la cámara |

Verificar con regla/cinta:

- Cada cuadrado negro exterior mide 16 × 16 cm, sin contar el blanco.
- Separación entre centros: 84 cm de izquierda a derecha y 64 cm de frente a fondo.
- Rectángulo exterior entre bordes negros: 100 × 80 cm. Estas NO son las distancias entre centros.
- Las cuatro flechas de las hojas apuntan al fondo, alejándose de la cámara. No girar hojas individualmente.
- Mantener hojas planas e inmóviles, margen blanco y esquinas libres, sin plástico brillante. Comprobar paralelismo y escuadra.
- No poner unas hojas elevadas y otras en el piso. El futuro objeto/persona apoyará sobre el mismo plano de los marcadores.

Enviar una sola foto original vertical, cámara principal 1×, desde donde se tomaría el cuerpo entero, con los cuatro marcadores visibles y sin persona por ahora. No tiene que ser una toma cenital.

Se comprobará primero lectura de IDs, tamaño proyectado y duplicados. Este control no requiere un perfil de lente, pero tampoco aprueba las posiciones físicas ni produce centímetros. No se usarán las líneas de altura de la pared.

## Nuevo comando de comprobación

Desde la raíz de proyecto-sastre-ia, con la ruta de la nueva foto:

```powershell
conda activate sastre-ia-perception
python -m src.calibration.cli inspect-floor --image RUTA_FOTO_ORIGINAL.jpg --output outputs/calibration/suelo_visibilidad_01.json
```

Resultados posibles:

- `marker_visibility_passed`: al menos 3 IDs esperados sin duplicados y tamaño suficiente. Intentar los cuatro. NO es `passed_reprojection_checks` ni una pose métrica.
- `needs_capture_adjustment`: faltan IDs, están repetidos o se ven pequeños/achatados. El diagnóstico se guarda y el comando indica fallo.
- Una foto del ChArUco normal solo contiene IDs 0–16 y no puede sustituir los ArUco 30–33 del suelo.

No se permite sobrescribir una salida existente sin `--force`.

## Lo que permanece pendiente

1. Cerrar la discrepancia de lente. La toma `IMG_20260908_121624_724.jpg` supera el límite; la `IMG_20260908_121232_143.jpg` está desenfocada. No se “reparan” digitalmente. Si se repiten, conservar el mismo tipo de inclinación y registrar la comprobación posterior sin borrar el fallo anterior.
2. Verificar el montaje del suelo y, con lente aprobada, estimar pose en cada foto.
3. Medir una varilla rígida vertical de longitud conocida, con extremos visibles y base en el suelo. Su longitud real se reserva para comprobar después, no para estimar la altura. Repetir posiciones; un resultado único no acredita precisión general.
4. Integrar percepción del cuerpo y luego frente/perfiles/espalda. Solo después evaluar medidas corporales y patronaje. El sistema actual aún no hace esa cadena automáticamente.

El papel muestra ondulaciones; no se demostró que sean la única causa del residuo. La recomendación de comprobar soporte plano y escala física sigue la [guía de Kalibr](https://github.com/ethz-asl/kalibr/wiki/Calibration-targets#tipsproblems).

## Reproducir la calibración conservada

Elegir un archivo de salida nuevo:

```powershell
python -m src.calibration.cli calibrate --train dataset/local/calibration/infinix_x6876_20260908/train --validation dataset/local/calibration/infinix_x6876_20260908/validation --camera-id infinix_x6876_20260908_diagnostic --output outputs/calibration/infinix_x6876_recheck.json
```

Resultado conocido de este conjunto completo: `needs_recapture`. El JSON es diagnóstico, no habilita mediciones.

## Verificación realizada

- Integridad SHA256 de los 30 JPG copiados y de sus fuentes: coincide; 126.404.918 bytes conservados, sin duplicados de píxeles.
- Entorno principal sastre-ia: 65 pruebas pasan.
- Entorno sastre-ia-perception: 32 pruebas de calibración/percepción pasan. Son pruebas solapadas, no se suman a las 65.
- Nuevo control probado con cuatro IDs correctos, IDs faltantes/desconocidos/repetidos, tamaño insuficiente y protección contra sobrescritura.
- Una foto real del ChArUco se rechaza correctamente como estación de suelo: lee IDs 0–16, no 30–33. Resultado conservado en `outputs/calibration/charuco_is_not_floor_preflight_20260908.json`.
- El nuevo control no modifica la exigencia de lente aprobada de `floor`. Ninguna de estas pruebas acredita precisión corporal real.
