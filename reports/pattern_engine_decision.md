# Motor de patronaje para SATRE-IA

## Decisión

Usar FreeSewing como motor paramétrico para el MVP:

- Jaeger para saco de hombre y mujer.
- Charlie para pantalón de hombre.
- Penelope para falda de mujer.

Los antiguos SVG/PDF de trapecios eran diagramas esquemáticos, no patrones de
confección. Su exportación normal queda desactivada.

## Por qué

FreeSewing genera piezas reales a medida en SVG y conserva elementos de
patronaje como curvas, mangas, cuello, pinzas, bolsillos y márgenes. Es preferible
a inventar geometría de patronaje con fórmulas sin validar.

Seamly2D queda como alternativa futura si un sastre entrega patrones maestros
en formato `.val`; es una herramienta CAD sólida, pero no aporta por sí sola el
patrón base que hoy falta.

## Condición de seguridad

El motor no completa medidas ausentes con promedios por altura, peso o sexo.
Primero se ejecuta `scripts/audit_pattern_engine.py`; solo si todas las entradas
requeridas están validadas se permite renderizar el SVG 1:1.

## Validación requerida

Antes de cortar tela final, cada diseño debe pasar por una toile o prototipo en
tela económica. Para evaluar SATRE-IA se necesitan medidas manuales de referencia
en desarrollo, aunque el producto final no obligue al usuario a usar cinta.

## Fuentes oficiales

- https://freesewing.org/docs/designs/
- https://freesewing.org/docs/designs/jaeger/
- https://freesewing.org/docs/designs/charlie/
- https://freesewing.org/docs/designs/penelope/
- https://freesewing.dev/reference/api/pattern/render/
- https://github.com/FashionFreedom/Seamly2D
