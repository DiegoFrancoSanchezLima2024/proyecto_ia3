import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

function analizarArgumentos(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error('Uso: node generate.mjs --readiness <json> --output-dir <directorio>');
    }
    args[key.slice(2)] = value;
  }
  return args;
}

const CARGADORES_DISENO = {
  jaeger: async () => (await import('@freesewing/jaeger')).Jaeger,
  charlie: async () => (await import('@freesewing/charlie')).Charlie,
  penelope: async () => (await import('@freesewing/penelope')).Penelope,
};

async function principal() {
  const args = analizarArgumentos(process.argv.slice(2));
  if (!args.readiness || !args['output-dir']) {
    throw new Error('Faltan --readiness o --output-dir');
  }

  const informe = JSON.parse(fs.readFileSync(args.readiness, 'utf8'));
  const margenCostura = args['seam-allowance-mm'] === undefined
    ? 10
    : Number(args['seam-allowance-mm']);
  if (!Number.isFinite(margenCostura) || margenCostura < 0 || margenCostura > 50) {
    throw new Error('--seam-allowance-mm debe estar entre 0 y 50');
  }
  const bloqueados = Object.entries(informe.garments).filter(([, garment]) => !garment.ready);
  if (bloqueados.length) {
    const detail = bloqueados
      .map(([name, garment]) => `${name}: ${garment.missing_measurements.join(', ')}`)
      .join('\n');
    throw new Error(`No se generaron moldes: faltan medidas validadas.\n${detail}`);
  }

  fs.mkdirSync(args['output-dir'], { recursive: true });
  for (const [garmentName, garment] of Object.entries(informe.garments)) {
    const cargarDiseno = CARGADORES_DISENO[garment.design];
    if (!cargarDiseno) throw new Error(`Diseño FreeSewing no soportado: ${garment.design}`);
    const Design = await cargarDiseno();
    const requeridasPorMotor = new Set(new Design().getConfig().measurements);
    const suministradas = new Set(Object.keys(garment.measurements_mm));
    const incompatibles = [...requeridasPorMotor].filter((name) => !suministradas.has(name));
    if (incompatibles.length) {
      throw new Error(
        `El mapa de ${garment.design} no coincide con FreeSewing; faltan: ${incompatibles.join(', ')}`,
      );
    }
    const patron = new Design({
      measurements: garment.measurements_mm,
      sa: margenCostura,
      units: 'metric',
      layout: true,
    });
    patron.draft();
    const errores = patron.getLogs().sets.flatMap((set) => set.error);
    if (errores.length) {
      throw new Error(`FreeSewing no pudo trazar ${garment.design}: ${JSON.stringify(errores)}`);
    }
    const svg = patron.render();
    fs.writeFileSync(path.join(args['output-dir'], `${garmentName}_1to1.svg`), svg, 'utf8');
  }
}

principal().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
