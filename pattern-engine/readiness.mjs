/**
 * Contrato de corte: un molde SOLO puede declararse listo para cortar tela
 * cuando cada medida que lo alimenta es trazable a una medida confirmada y la
 * sesion de medicion no quedo marcada.
 *
 * Falla cerrado a proposito. Si no llega procedencia, el molde se emite como
 * BORRADOR NO VALIDADO. Antes de este modulo, `status: 'draft_ready'` era una
 * constante escrita a mano que se emitia aunque la decision fuera
 * `manual_confirmation` y aunque 10 de 12 medidas fueran proporciones genericas.
 */

/** Decisiones de la etapa de medicion que impiden declarar un molde cortable. */
export const DECISIONES_BLOQUEANTES = new Set(['manual_confirmation', 'repeat_capture']);

export const ESTADO_LISTO = 'draft_ready';
export const ESTADO_NO_VALIDADO = 'draft_unvalidated';

/** Marca visible que debe aparecer en cualquier salida no validada. */
export const MARCA_NO_VALIDADO = 'BORRADOR NO VALIDADO - NO CORTAR TELA DEFINITIVA';

/** Origen de una medida. Solo `confirmed` transfiere la responsabilidad al humano. */
const DERIVADA = 'derivada';
const MEDIDA_IA = 'medida IA';

/**
 * @param {object}   input
 * @param {string?}  input.decision   decision de la sesion de medicion
 * @param {object}   input.sources    { nombreMedida: 'medida IA' | 'derivada' | 'guardada' }
 * @param {string[]} input.required   medidas que el patron exige
 * @param {object}   input.intervals  { nombreMedida: { requires_manual_confirmation, lower_cm, upper_cm } }
 */
export function evaluarListoParaCorte({ decision = null, sources = {}, required = [], intervals = {} } = {}) {
  const blocking_reasons = [];
  const derived_measurements = [];
  const unreliable_measurements = [];

  if (decision === null || decision === undefined || decision === '') {
    blocking_reasons.push(
      'No se recibio la procedencia de las medidas, asi que no se puede verificar su origen.',
    );
  } else if (DECISIONES_BLOQUEANTES.has(decision)) {
    blocking_reasons.push(
      `La sesion de medicion termino en "${decision}", que exige confirmacion humana antes de cortar.`,
    );
  }

  for (const name of required) {
    if (sources[name] === DERIVADA) derived_measurements.push(name);
    // Una medida del modelo cuyo intervalo conformal es demasiado ancho no
    // sirve para cortar tela, por mucho que el trazado sea coherente.
    if (sources[name] === MEDIDA_IA && intervals[name]?.requires_manual_confirmation) {
      unreliable_measurements.push(name);
    }
  }

  if (derived_measurements.length) {
    blocking_reasons.push(
      `${derived_measurements.length} de ${required.length} medidas son proporciones genericas, no medidas de esta persona: ${derived_measurements.join(', ')}.`,
    );
  }
  if (unreliable_measurements.length) {
    blocking_reasons.push(
      `Intervalo de confianza demasiado ancho para cortar en: ${unreliable_measurements.join(', ')}.`,
    );
  }

  const cut_ready = blocking_reasons.length === 0;
  return {
    cut_ready,
    status: cut_ready ? ESTADO_LISTO : ESTADO_NO_VALIDADO,
    blocking_reasons,
    derived_measurements,
    unreliable_measurements,
    measured_count: required.length - derived_measurements.length,
    required_count: required.length,
    decision,
    notice: cut_ready
      ? 'Todas las medidas requeridas son confirmadas. Validar el ajuste en tela de ensayo antes de cortar tela definitiva.'
      : `${MARCA_NO_VALIDADO}. ${blocking_reasons.join(' ')}`,
  };
}

/**
 * Estampa la marca de no validado sobre el SVG del molde.
 *
 * Se inyecta justo antes de `</svg>` en coordenadas de patron (mm), de modo que
 * sobrevive tanto al visor del navegador como a la conversion a PDF: el PDF se
 * construye desde este mismo SVG, asi que no puede salir sin la marca.
 */
export function estamparNoValidado(svg, assessment, widthMm, heightMm) {
  if (assessment.cut_ready) return svg;
  const w = Number.isFinite(widthMm) && widthMm > 0 ? widthMm : 200;
  const h = Number.isFinite(heightMm) && heightMm > 0 ? heightMm : 200;
  const band = Math.max(8, Math.min(w, h) * 0.035);
  const detail = [
    `${assessment.measured_count}/${assessment.required_count} medidas confirmadas`,
    assessment.decision ? `decision: ${assessment.decision}` : 'sin procedencia',
  ].join(' · ');
  // Diagonales repetidas: impiden que la marca se recorte al imprimir por piezas.
  const diagonals = [0.25, 0.5, 0.75]
    .map(
      (f) =>
        `<text x="${(w * f).toFixed(2)}" y="${(h * f).toFixed(2)}" font-family="Helvetica" font-size="${(band * 1.1).toFixed(2)}" fill="#c1121f" fill-opacity="0.22" text-anchor="middle" transform="rotate(-30 ${(w * f).toFixed(2)} ${(h * f).toFixed(2)})">${MARCA_NO_VALIDADO}</text>`,
    )
    .join('');
  const header =
    `<rect x="0" y="0" width="${w.toFixed(2)}" height="${band.toFixed(2)}" fill="#c1121f" fill-opacity="0.9"/>` +
    `<text x="${(w / 2).toFixed(2)}" y="${(band * 0.72).toFixed(2)}" font-family="Helvetica" font-size="${(band * 0.55).toFixed(2)}" fill="white" text-anchor="middle">${MARCA_NO_VALIDADO}</text>` +
    `<text x="${(w / 2).toFixed(2)}" y="${(band * 1.55).toFixed(2)}" font-family="Helvetica" font-size="${(band * 0.38).toFixed(2)}" fill="#c1121f" text-anchor="middle">${detail}</text>`;
  return svg.replace(/<\/svg>\s*$/, `<g id="sastre-ia-unvalidated">${header}${diagonals}</g></svg>`);
}
