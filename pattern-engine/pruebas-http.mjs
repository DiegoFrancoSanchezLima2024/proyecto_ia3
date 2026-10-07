import { strict as assert } from 'node:assert';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { Jaeger } from '@freesewing/jaeger';
import { cisMaleAdult40 } from '@freesewing/models';

const puerto = 18000 + (process.pid % 10000);
const origen = `http://127.0.0.1:${puerto}`;
const servidor = spawn(process.execPath, ['server.mjs'], {
  cwd: import.meta.dirname,
  env: {
    ...process.env,
    SATRE_PATTERN_PORT: String(puerto),
    SATRE_ENABLE_EXPERIMENTAL_TRYON: '0',
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});

async function esperarServidor() {
  let salida = '';
  const limite = Date.now() + 10_000;
  while (!salida.includes(`127.0.0.1:${puerto}`)) {
    if (Date.now() > limite) throw new Error(`El servidor no inició. Salida: ${salida}`);
    const resultado = await Promise.race([
      once(servidor.stdout, 'data').then(([datos]) => ({ datos })),
      once(servidor, 'exit').then(([codigo]) => ({ codigo })),
      new Promise(resolve => setTimeout(() => resolve({ espera: true }), 250)),
    ]);
    if ('codigo' in resultado) throw new Error(`El servidor terminó con código ${resultado.codigo}`);
    if (resultado.datos) salida += resultado.datos.toString();
  }
}

try {
  await esperarServidor();

  const requeridas = new Jaeger().getConfig().measurements;
  const medidas = Object.fromEntries(requeridas.map(nombre => [
    nombre,
    nombre === 'shoulderSlope' ? cisMaleAdult40[nombre] : cisMaleAdult40[nombre] / 10,
  ]));
  const fuentes = Object.fromEntries(requeridas.map(nombre => [nombre, 'manual']));
  const respuesta = await fetch(`${origen}/api/pattern`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      garment: 'jacket',
      sex: 'male',
      person: { name: 'prueba-integracion' },
      measurements: medidas,
      seam_allowance_mm: 10,
      measurement_sources: fuentes,
    }),
  });
  const patron = await respuesta.json();
  assert.equal(respuesta.status, 200);
  assert.equal(patron.garment, 'jacket');
  assert.match(patron.svg, /<svg/);
  assert.ok(patron.width_mm > 0 && patron.height_mm > 0);

  const asset = await fetch(`${origen}/assets/tryon-examples/male.png`);
  assert.equal(asset.status, 200);
  assert.equal(asset.headers.get('content-type'), 'image/png');
  assert.ok((await asset.arrayBuffer()).byteLength > 0);

  const estado = await fetch(`${origen}/api/tryon/status`);
  assert.equal(estado.status, 200);
  assert.equal((await estado.json()).ready, false);

  console.log('Integración HTTP interfaz/servidor/FreeSewing: válida');
} finally {
  servidor.kill();
  await Promise.race([
    once(servidor, 'exit'),
    new Promise(resolve => setTimeout(resolve, 2000)),
  ]);
}
