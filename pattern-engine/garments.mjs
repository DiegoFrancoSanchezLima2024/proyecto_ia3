import { Jaeger } from '@freesewing/jaeger';
import { Charlie } from '@freesewing/charlie';
import { Penelope } from '@freesewing/penelope';
import { imprimirSvg } from './print-svg.mjs';
import { evaluarListoParaCorte, estamparNoValidado } from './readiness.mjs';

export const especificacionesPrenda = {
  jacket: { Design: Jaeger, label: 'Saco Jaeger', sex: ['male', 'female'] },
  trousers: { Design: Charlie, label: 'Pantalon Charlie', sex: ['male'] },
  skirt: { Design: Penelope, label: 'Falda Penelope', sex: ['female'] },
};

function positivo(name, raw, unit = 'cm') {
  const value = Number(raw);
  const max = unit === 'degree' ? 60 : 300;
  if (!Number.isFinite(value) || value <= 0 || value > max) {
    throw new Error(`Medida invalida: ${name}`);
  }
  return unit === 'degree' ? value : value * 10;
}

export function generarPrenda(input) {
  const especificacion = especificacionesPrenda[input.garment];
  if (!especificacion) throw new Error('Prenda no soportada.');
  if (!especificacion.sex.includes(input.sex)) throw new Error('La prenda no corresponde al perfil seleccionado.');
  const required = new especificacion.Design().getConfig().measurements;
  const suministradas = input.measurements || {};
  const faltantes = required.filter(name => suministradas[name] === '' || suministradas[name] == null);
  if (faltantes.length) throw new Error(`Faltan medidas: ${faltantes.join(', ')}`);
  const measurements = Object.fromEntries(required.map(name => [
    name,
    positivo(name, suministradas[name], name === 'shoulderSlope' ? 'degree' : 'cm'),
  ]));
  const sa = Number(input.seam_allowance_mm ?? 10);
  if (!Number.isFinite(sa) || sa < 0 || sa > 30) throw new Error('Margen de costura invalido.');
  const patron = new especificacion.Design({ measurements, sa, units: 'metric', scale: 1, layout: true }).draft();
  const errores = patron.getLogs().sets.flatMap(set => set.error);
  if (errores.length) throw new Error(`No se pudo trazar: ${JSON.stringify(errores)}`);
  let svg = imprimirSvg(patron.render());
  if (!(patron.width > 0 && patron.height > 0)) throw new Error('El patron no tiene dimensiones validas.');
  if (/NaN|Infinity/.test(svg)) throw new Error('El patron contiene geometria invalida.');
  // Un trazado sin NaN demuestra coherencia matematica, no ajuste corporal. El
  // estado de corte se deriva de la procedencia de las medidas, nunca se asume.
  const listoParaCorte = evaluarListoParaCorte({
    decision: input.decision ?? null,
    sources: input.measurement_sources || {},
    required,
    intervals: input.measurement_intervals || {},
  });
  svg = estamparNoValidado(svg, listoParaCorte, patron.width, patron.height);
  return {
    svg,
    width_mm: patron.width,
    height_mm: patron.height,
    garment: input.garment,
    garment_label: especificacion.label,
    sex: input.sex,
    person: input.person || null,
    measurements: suministradas,
    seam_allowance_mm: sa,
    status: listoParaCorte.status,
    cut_ready: listoParaCorte.cut_ready,
    validation: listoParaCorte,
    notice: listoParaCorte.notice,
  };
}
