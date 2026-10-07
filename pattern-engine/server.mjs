import http from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { generarFalda } from './skirt.mjs';
import { pdfFalda } from './pdf.mjs';
import { pdfPatron } from './pdf.mjs';
import { generarPrenda } from './garments.mjs';

const ID_SESION = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const registroModelos = JSON.parse(
  await readFile(new URL('../configs/modelos_runtime.json', import.meta.url), 'utf8'),
);
const perfilVestidor = registroModelos.vestidor.perfil_demo_en_vivo;
const VERSION_CACHE_VESTIDOR = 2; // v2 elimina por completo el overlay 2D de calzado.

/**
 * Lee la procedencia real de las medidas desde el prediction.json de la sesion.
 *
 * El navegador no es una fuente confiable de su propia validacion: si el cliente
 * declarara la decision, bastaria con editarla para desbloquear el corte. La
 * decision y los intervalos conformales se leen aqui del disco y pisan lo que
 * haya mandado el cliente.
 */
async function leerProcedenciaSesion(sessionId) {
  if (!ID_SESION.test(String(sessionId || ''))) return null;
  const file = path.join(raizProyecto, 'outputs', 'ui_sessions', String(sessionId), 'prediction.json');
  let prediction;
  try {
    prediction = JSON.parse(await readFile(file, 'utf8'));
  } catch {
    return null;
  }
  const intervals = {};
  for (const [name, value] of Object.entries(prediction.measurements || {})) {
    intervals[name] = {
      requires_manual_confirmation: Boolean(value?.requires_manual_confirmation),
      lower_cm: value?.lower_cm ?? null,
      upper_cm: value?.upper_cm ?? null,
    };
  }
  return { decision: prediction.decision ?? null, intervals };
}

/**
 * Traduce los nombres de medida del patron (FreeSewing) a los del modelo, para
 * poder cruzar cada entrada del molde con su intervalo conformal.
 */
function mapearIntervalosAPatron(intervals, sources) {
  const mapped = {};
  for (const [field, origen] of Object.entries(sources || {})) {
    const key = PATRON_A_MODELO[field];
    if (key && intervals[key]) mapped[field] = intervals[key];
    else if (intervals[field]) mapped[field] = intervals[field];
  }
  return mapped;
}

// Correspondencia entre medidas exigidas por FreeSewing y las 13 del modelo.
const PATRON_A_MODELO = {
  chest: 'chest', waist: 'waist', hips: 'hip', seat: 'hip',
  shoulderToWrist: 'arm-length', shoulderToShoulder: 'shoulder-breadth',
  upperLeg: 'thigh', waistToFloor: 'leg-length', inseam: 'leg-length',
  waistToSeat: 'shoulder-to-crotch', biceps: 'bicep', wrist: 'wrist',
  ankle: 'ankle', calf: 'calf',
};

const puerto = Number(process.env.SATRE_PATTERN_PORT || 8765);
const origen = `http://127.0.0.1:${puerto}`;
const vestidorExperimentalHabilitado = process.env.SATRE_ENABLE_EXPERIMENTAL_TRYON === '1';
const pagina = await readFile(new URL('./ui.html', import.meta.url));
const ejemplosVestidor = new Map([
  ['/assets/tryon-examples/male.png', new URL('./assets/tryon-examples/male-navy-suit-target.png', import.meta.url)],
  ['/assets/tryon-examples/female.png', new URL('./assets/tryon-examples/female-navy-skirt-suit-target.png', import.meta.url)],
]);
const raizProyecto = fileURLToPath(new URL('../', import.meta.url));
const ejecutarArchivoAsync = promisify(execFile);

function bufferImagen(photo, label) {
  // Los tipos MIME son un contrato estandar del navegador y no se traducen.
  if (!photo || !['image/jpeg', 'image/png', 'image/webp'].includes(photo.type)) {
    throw new Error(`Formato de ${label} no soportado.`);
  }
  const buffer = Buffer.from(photo.base64 || '', 'base64');
  if (!buffer.length || buffer.length > 12 * 1024 * 1024) throw new Error(`${label} vacia o mayor a 12 MB.`);
  return { buffer, ext: photo.type === 'image/png' ? '.png' : photo.type === 'image/webp' ? '.webp' : '.jpg' };
}

async function medirFotos(cargaUtil) {
  let altura = Number(String(cargaUtil.height_cm ?? '').replace(',', '.'));
  if (altura >= 1 && altura <= 2.3) altura *= 100;
  altura = Math.round(altura * 10) / 10;
  if (!Number.isFinite(altura) || altura < 100 || altura > 230) throw new Error('Estatura inválida. Escribe 164 cm o 1.64 m.');
  if (!['male', 'female'].includes(cargaUtil.sex)) throw new Error('Selecciona el conjunto de sastrería.');
  const id = randomUUID();
  const salidaRelativa = path.join('outputs', 'ui_sessions', id);
  const salidaAbsoluta = path.join(raizProyecto, salidaRelativa);
  const dirFotos = path.join(salidaAbsoluta, 'photos');
  await mkdir(dirFotos, { recursive: true });
  const argumentosArchivo = {};
  for (const view of ['front', 'left']) {
    const { buffer, ext } = bufferImagen(cargaUtil.photos?.[view], `foto ${view}`);
    const target = path.join(dirFotos, `${view}${ext}`);
    await writeFile(target, buffer);
    argumentosArchivo[view] = target;
  }
  const args = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(raizProyecto, 'scripts', 'run_assisted_demo.ps1'),
    '-Front', argumentosArchivo.front, '-Left', argumentosArchivo.left,
    '-HeightCm', String(altura), '-GarmentRoute', cargaUtil.sex,
    '-ClothingFit', cargaUtil.clothing_fit || 'unknown', '-OutputDir', salidaRelativa, '-SkipBody3D'];
  try {
    await ejecutarArchivoAsync('powershell.exe', args, { cwd: raizProyecto, timeout: 10 * 60 * 1000, maxBuffer: 10 * 1024 * 1024 });
  } catch (error) {
    const detail = String(error.stderr || error.stdout || error.message).slice(-1400);
    throw new Error(`La medicion local fallo: ${detail}`);
  }
  const prediction = JSON.parse(await readFile(path.join(salidaAbsoluta, 'prediction.json'), 'utf8'));
  const manifest = JSON.parse(await readFile(path.join(salidaAbsoluta, 'capture_manifest.json'), 'utf8'));
  const frontPose = manifest.views?.front?.pose;
  prediction.preview_2d = {
    source: frontPose ? 'mediapipe_pose' : 'fallback',
    front_landmarks: frontPose?.landmarks || null,
    pose_quality: frontPose?.quality || null,
  };
  prediction.local_session = { id, directory: salidaRelativa, photos_stored_locally: true };
  return prediction;
}

async function generarVestidor(cargaUtil) {
  if (!vestidorExperimentalHabilitado) {
    throw new Error('La prueba virtual personalizada permanece deshabilitada: el motor local experimental todavia no supera el control de calidad para sastreria formal.');
  }
  const id = String(cargaUtil.session_id || '');
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) throw new Error('Sesion local de fotos invalida. Primero saca las medidas.');
  if (!['male', 'female'].includes(cargaUtil.sex)) throw new Error('Sexo invalido.');
  const dirSesion = path.join(raizProyecto, 'outputs', 'ui_sessions', id);
  const dirSalida = path.join(dirSesion, 'tryon-fashn');
  const salida = path.join(dirSalida, 'result.png');
  const salidaCruda = path.join(dirSalida, 'result-raw.png');
  const reporteRuntime = path.join(dirSalida, 'runtime_report.json');
  await readFile(path.join(dirSesion, 'capture_manifest.json'));
  await mkdir(dirSalida, { recursive: true });
  // Una sesión de fotos es inmutable. Si el mismo sexo y perfil de inferencia
  // ya terminaron correctamente, se sirve el resultado auditado sin consumir
  // otros dos minutos de GPU durante la presentación.
  try {
    const cache = JSON.parse(await readFile(reporteRuntime, 'utf8'));
    if (cache.version === VERSION_CACHE_VESTIDOR
      && cache.sex === cargaUtil.sex
      && JSON.stringify(cache.profile) === JSON.stringify(perfilVestidor)) {
      const imagenCache = await readFile(path.join(raizProyecto, cache.stored_at));
      return {
        ...cache.response,
        image_data_url: `data:image/png;base64,${imagenCache.toString('base64')}`,
        cached: true,
      };
    }
  } catch {
    // Sin cache válida: continúa con inferencia real.
  }
  const conda = 'C:\\Users\\diego\\miniconda3\\Scripts\\conda.exe';
  const referenciaPrenda = path.join(
    raizProyecto, 'pattern-engine', 'assets', 'garments',
    cargaUtil.sex === 'female'
      ? 'female-navy-skirt-suit-model-reference-v1.png'
      : 'male-navy-suit-model-reference-v1.png',
  );
  const entornoFashn = {
    ...process.env,
    HF_HOME: path.join(raizProyecto, 'pretrained', 'hf-cache'),
    HF_HUB_OFFLINE: '1',
    TRANSFORMERS_OFFLINE: '1',
    // Sin esto, la salida de Python queda bufferizada y, si el proceso se
    // mata por timeout a medio generar, no se captura nada (ni siquiera el
    // progreso de tqdm) y el unico error visible es "Command failed: <cmd>".
    PYTHONUNBUFFERED: '1',
  };
  const argumentosGeneracion = ['run', '-n', 'sastre-ia-fashn', 'python', path.join(raizProyecto, 'scripts', 'run_fashn_vton_local.py'),
    '--session-dir', dirSesion,
    '--garment-image', referenciaPrenda,
    '--weights-dir', path.join(raizProyecto, 'pretrained', 'fashn-vton-1.5'),
    '--output', salidaCruda,
    '--category', 'one-pieces', '--garment-photo-type', 'model',
    '--steps', String(perfilVestidor.pasos),
    '--seed', String(perfilVestidor.semilla),
    '--num-candidates', String(perfilVestidor.candidatos)];
  let informeSeleccion;
  let informeComposicion;
  try {
    // Paso 1: generar varios candidatos independientes (distintas seeds).
    await ejecutarArchivoAsync(conda, argumentosGeneracion, {
      cwd: raizProyecto,
      // El perfil de demo se centraliza en configs/modelos_runtime.json. En la
      // RTX 3050, un candidato a 30 pasos tarda alrededor de 90 s y evita el
      // fallo observado al reutilizar el pipeline para tres candidatos.
      timeout: Number(perfilVestidor.timeout_minutos) * 60 * 1000,
      maxBuffer: 10 * 1024 * 1024,
      env: entornoFashn,
    });
    // Paso 2: puntuar cada candidato (color de prenda + fuga de piel) y
    // promover el mejor a result-raw.png. Nunca decide silenciosamente:
    // deja selection_report.json con el detalle de todos los candidatos.
    const { stdout: selectionStdout } = await ejecutarArchivoAsync(conda, [
      'run', '-n', 'sastre-ia-fashn', 'python',
      path.join(raizProyecto, 'scripts', 'select_best_tryon_candidate.py'),
      '--session-dir', dirSesion,
      '--candidates-manifest', path.join(dirSalida, 'candidates.json'),
      '--sex', cargaUtil.sex,
      '--garment-image', referenciaPrenda,
      '--output', salidaCruda,
    ], {
      cwd: raizProyecto,
      timeout: 2 * 60 * 1000,
      maxBuffer: 4 * 1024 * 1024,
      env: entornoFashn,
    });
    informeSeleccion = JSON.parse(selectionStdout);
    // Paso 3: restaurar fondo, cabeza y manos originales sobre el ganador.
    const { stdout: compositorStdout } = await ejecutarArchivoAsync(conda, [
      'run', '-n', 'sastre-ia-fashn', 'python',
      path.join(raizProyecto, 'scripts', 'composite_tryon_identity.py'),
      '--session-dir', dirSesion,
      '--generated', salidaCruda,
      '--output', salida,
      '--dilation', '3',
    ], {
      cwd: raizProyecto,
      timeout: 2 * 60 * 1000,
      maxBuffer: 2 * 1024 * 1024,
      env: entornoFashn,
    });
    informeComposicion = JSON.parse(compositorStdout);
    if (informeComposicion.restore_mask?.rectangular_restore_detected) {
      throw new Error('La restauración de identidad produjo una franja rectangular; resultado rechazado por QA.');
    }
  } catch (error) {
    const capturedOutput = String(error.stderr || error.stdout || '').trim();
    let detail = capturedOutput.slice(-1800);
    if (!detail) {
      // Sin salida capturada: casi siempre es timeout (el proceso murio a
      // medio generar antes de imprimir nada) o el ejecutable/entorno conda
      // no existe. error.killed/signal lo distingue de un crash real.
      detail = error.killed || error.signal
        ? `El proceso se detuvo sin generar nada (timeout o memoria insuficiente, señal ${error.signal || 'desconocida'}). Si tu GPU es modesta, intenta de nuevo o reduce --num-candidates/--steps en scripts/run_fashn_vton_local.py.`
        : `Sin salida capturada. Verifica que el entorno conda "sastre-ia-fashn" exista y que python funcione ahi. Detalle tecnico: ${error.message}`;
    }
    throw new Error(`La prueba virtual local fallo: ${detail}`);
  }
  // FASHN no soporta calzado. No se agrega ningun overlay 2D: ese metodo
  // producia un efecto de pegatina y falseaba el origen generativo de la
  // vista. El calzado se mantiene como limitacion explicita hasta que el
  // postproceso de inpainting localizado supere su QA independiente.
  const imagen = await readFile(salida);
  const resolucionGanadora = informeSeleccion?.winner?.output_resolution;
  const respuesta = {
    engine: 'FASHN VTON v1.5 + compositor semántico de identidad',
    mode: 'experimental',
    resolution: Array.isArray(resolucionGanadora) ? resolucionGanadora.join('x') : 'desconocida',
    image_data_url: `data:image/png;base64,${imagen.toString('base64')}`,
    stored_at: path.relative(raizProyecto, salida),
    selection: informeSeleccion && {
      candidates_evaluated: informeSeleccion.candidates_evaluated,
      winner_seed: informeSeleccion.winner?.seed,
      winner_verdict: informeSeleccion.winner?.verdict,
      all_candidates_suspicious: informeSeleccion.all_candidates_suspicious,
    },
    identity_qa: informeComposicion && {
      source: informeComposicion.identity_restore_source,
      background_preserved: informeComposicion.background?.preserved,
      rectangular_restore_detected: informeComposicion.restore_mask?.rectangular_restore_detected,
      visible_band_detected: informeComposicion.restore_boundary?.visible_band_detected,
      warnings: informeComposicion.warnings || [],
    },
    notice: 'MODO EXPERIMENTAL: generación FASHN local restringida a la silueta; el perfil en vivo usa una semilla reproducible y control técnico; restaura fondo, cara, pelo y manos desde la foto original mediante parsing semántico. Es una visualización de apariencia y no valida ajuste, holgura, caída ni sustituye el molde métrico.',
  };
  await writeFile(reporteRuntime, JSON.stringify({
    version: VERSION_CACHE_VESTIDOR,
    sex: cargaUtil.sex,
    profile: perfilVestidor,
    stored_at: respuesta.stored_at,
    response: respuesta,
  }, null, 2), 'utf8');
  return { ...respuesta, cached: false };
}

async function generarVistaCompuesta(cargaUtil) {
  const id = String(cargaUtil.session_id || '');
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) throw new Error('Sesion local de fotos invalida. Primero saca las medidas.');
  if (!['male', 'female'].includes(cargaUtil.sex)) throw new Error('Sexo invalido.');
  const dirSesion = path.join(raizProyecto, 'outputs', 'ui_sessions', id);
  const dirSalida = path.join(dirSesion, 'preview-2d');
  const salida = path.join(dirSalida, 'result.png');
  await readFile(path.join(dirSesion, 'capture_manifest.json'));
  await mkdir(dirSalida, { recursive: true });
  const conda = 'C:\\Users\\diego\\miniconda3\\Scripts\\conda.exe';
  const args = ['run', '-n', 'sastre-ia', 'python', path.join(raizProyecto, 'scripts', 'build_layered_tryon_preview.py'),
    '--session-dir', dirSesion, '--sex', cargaUtil.sex, '--output', salida, '--width', '768', '--height', '1024'];
  try {
    await ejecutarArchivoAsync(conda, args, { cwd: raizProyecto, timeout: 3 * 60 * 1000, maxBuffer: 4 * 1024 * 1024 });
  } catch (error) {
    const detail = String(error.stderr || error.stdout || error.message).slice(-1800);
    throw new Error(`La composicion 2D local fallo: ${detail}`);
  }
  const [imagen, textoMetadatos] = await Promise.all([
    readFile(salida), readFile(path.join(dirSalida, 'preview_2d_metadata.json'), 'utf8'),
  ]);
  const metadatos = JSON.parse(textoMetadatos);
  return {
    mode: metadatos.mode,
    resolution: metadatos.resolution.join('x'),
    image_data_url: `data:image/png;base64,${imagen.toString('base64')}`,
    stored_at: path.relative(raizProyecto, salida),
    qa: metadatos.qa,
    parsing: metadatos.parsing,
    semantic_human_parsing: metadatos.semantic_human_parsing,
    notice: metadatos.notice,
  };
}
http.createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Cache-Control', 'no-store');
  if (req.headers.host !== `127.0.0.1:${puerto}` || (req.headers.origen && req.headers.origen !== origen)) {
    res.writeHead(403); res.end(); return;
  }
  if (req.method === 'GET' && req.url === '/') {
    res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(pagina); return;
  }
  if (req.method === 'GET' && ejemplosVestidor.has(req.url)) {
    res.setHeader('Content-Type', 'image/png');
    res.end(await readFile(ejemplosVestidor.get(req.url))); return;
  }
  if (req.method === 'GET' && req.url === '/api/tryon/status') {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({
      ready: vestidorExperimentalHabilitado,
      engine: 'FASHN VTON v1.5 local + preservacion de identidad',
      profile: perfilVestidor,
      reason: vestidorExperimentalHabilitado
        ? 'Motor experimental habilitado: referencia vestida, inferencia local y composición restringida a la persona.'
        : 'Instalado pero deshabilitado hasta habilitar explícitamente el modo experimental.',
    }));
    return;
  }
  if (req.method !== 'POST' || !['/api/skirt', '/api/skirt/pdf', '/api/pattern', '/api/pattern/pdf', '/api/measure', '/api/tryon', '/api/tryon/layered'].includes(req.url)) {
    res.writeHead(404); res.end(); return;
  }
  try {
    let body = '';
    for await (const chunk of req) {
      body += chunk;
      const limit = req.url === '/api/measure' ? 50 * 1024 * 1024 : 32768;
      if (Buffer.byteLength(body) > limit) throw new Error('Entrada demasiado grande.');
    }
    const cargaUtil = JSON.parse(body);
    if (req.url === '/api/measure') {
      const result = await medirFotos(cargaUtil);
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result)); return;
    }
    if (req.url === '/api/tryon') {
      const result = await generarVestidor(cargaUtil);
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result)); return;
    }
    if (req.url === '/api/tryon/layered') {
      const result = await generarVistaCompuesta(cargaUtil);
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result)); return;
    }
    // La procedencia mandan el disco, no el cliente.
    const provenance = await leerProcedenciaSesion(cargaUtil.session_id);
    if (provenance) {
      cargaUtil.decision = provenance.decision;
      cargaUtil.measurement_intervals = mapearIntervalosAPatron(provenance.intervals, cargaUtil.measurement_sources);
    } else {
      // Sin sesion verificable no se puede declarar un molde cortable.
      cargaUtil.decision = null;
      cargaUtil.measurement_intervals = {};
    }
    const result = req.url.startsWith('/api/pattern') ? generarPrenda(cargaUtil) : generarFalda(cargaUtil);
    if (req.url === '/api/pattern/pdf') {
      const pdf = await pdfPatron(result);
      res.setHeader('Content-Type', 'application/pdf');
      const tag = result.cut_ready ? 'borrador' : 'BORRADOR-NO-VALIDADO';
      res.setHeader('Content-Disposition', `attachment; filename="${result.garment}_${tag}_A4.pdf"`);
      res.end(pdf); return;
    }
    if (req.url === '/api/skirt/pdf') {
      const pdf = await pdfFalda(result);
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="penelope_${result.cut_ready ? 'borrador' : 'BORRADOR-NO-VALIDADO'}_A4.pdf"`);
      res.end(pdf); return;
    }
    res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result));
  } catch (error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: error.message }));
  }
}).listen(puerto, '127.0.0.1', () => console.log(`SATRE-IA local: ${origen}`));
