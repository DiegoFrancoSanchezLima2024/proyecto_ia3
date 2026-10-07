# Revisión del primer montaje de suelo

Fotos originales: IMG_20260908_135850_346.jpg e IMG_20260908_135917_266.jpg.

Ambas fotos (3456 × 4608) permiten detectar los cuatro IDs 30, 31, 32 y 33, sin duplicados ni otros IDs. Lado proyectado mínimo: 290,56 y 290,45 px, respectivamente, por encima del umbral de visibilidad de 20 px.

Los resultados `marker_visibility_passed` guardados en `outputs/calibration/suelo_visibilidad_20260908_135850.json` y `suelo_visibilidad_20260908_135917.json` solo acreditan lectura y tamaño de los marcadores. No acreditan el montaje físico.

## Montaje no válido para el modelo de suelo actual

La inspección visual muestra las cuatro hojas de pie, apoyadas/inclinadas. El modelo del proyecto supone que todas las esquinas negras están en el mismo plano horizontal Z=0. Las hojas de estas fotos no satisfacen ese supuesto. No se calculó pose ni ninguna longitud con ellas.

Corrección: acostar las cuatro hojas completamente sobre el suelo, impresión mirando hacia arriba, sin soportes que las levanten. Mantener 30/31 al fondo y 32/33 delante, flechas hacia el fondo; después de acostarlas volver a comprobar 84 cm entre centros lateralmente y 64 cm en profundidad. Cada cuadrado negro debe medir 16 × 16 cm. Fijar por márgenes blancos, sin tapar códigos ni esquinas y sin arrugas.

Las distancias físicas no se verificaron a partir de estas fotos. Se necesita una nueva foto original vertical 1× con el montaje acostado y los cuatro códigos visibles, sin persona, para repetir el control de visibilidad.

La discrepancia de calibración de lente previamente registrada permanece pendiente. Corregir el suelo no la elimina ni habilita por sí solo medidas en centímetros. No se modificaron las fotos, los parámetros ni los controles de seguridad.
