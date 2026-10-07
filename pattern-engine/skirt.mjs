import { Penelope } from '@freesewing/penelope';
import { imprimirSvg } from './print-svg.mjs';
import { evaluarListoParaCorte, estamparNoValidado } from './readiness.mjs';

export const requeridas = ['waist', 'seat', 'waistToHips', 'waistToSeat', 'waistToKnee'];

export function generarFalda(input) {
  const medidas = {};
  for (const key of requeridas) {
    const value = input.measurements_cm?.[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 300)
      throw new Error(`Medida ausente o inválida: ${key}`);
    medidas[key] = value * 10;
  }
  if (!(medidas.waistToHips < medidas.waistToSeat && medidas.waistToSeat < medidas.waistToKnee))
    throw new Error('Las alturas deben cumplir: cadera alta < asiento < rodilla.');
  const margenCostura = input.seam_allowance_mm ?? 10;
  if (typeof margenCostura !== 'number' || !Number.isFinite(margenCostura) || margenCostura < 0 || margenCostura > 30)
    throw new Error('El margen debe estar entre 0 y 30 mm.');
  const patron = new Penelope({ measurements: medidas, sa: margenCostura, units: 'metric', scale: 1, layout: true });
  patron.draft();
  const errores = patron.getLogs().sets.flatMap(set => set.error);
  if (errores.length) throw new Error(`No se pudo trazar la falda: ${JSON.stringify(errores)}`);
  let svg = imprimirSvg(patron.render());
  if (!(patron.width > 0 && patron.height > 0) || /NaN|Infinity/.test(svg))
    throw new Error('El motor produjo geometría inválida.');
  const listoParaCorte = evaluarListoParaCorte({
    decision: input.decision ?? null,
    sources: input.measurement_sources || {},
    required: requeridas,
    intervals: input.measurement_intervals || {},
  });
  svg = estamparNoValidado(svg, listoParaCorte, patron.width, patron.height);
  return { svg, width_mm: patron.width, height_mm: patron.height,
    status: listoParaCorte.status, cut_ready: listoParaCorte.cut_ready, validation: listoParaCorte,
    source: input.source === 'reference' ? 'reference' : 'user_input',
    measurements_cm: input.measurements_cm, seam_allowance_mm: margenCostura,
    design: 'Penelope', engine_version: '4.10.1',
    notice: listoParaCorte.notice };
}
