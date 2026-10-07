"""Pruebas sintéticas: verifican código, NO precisión en personas reales."""
from __future__ import annotations

import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
    pytest.skip("Requiere entorno sastre-ia-perception con OpenCV ArUco moderno", allow_module_level=True)

from src.calibration.common import DEFAULT_CONFIG, matrices_camara, tablero_charuco, dictionary, leer_json
from src.calibration.intrinsics import calibrate, recolectar_vistas, ajustar_camara
from src.calibration.kit import generar_kit
from src.calibration.metric import detectar_piso, esquinas_objeto_piso, inspeccionar_marcadores_piso, resolver_pose_piso, altura_segmento_vertical


@pytest.fixture
def config():
    return leer_json(DEFAULT_CONFIG)


def scene(center=(0, -3, 1.15), target=(0, 0, 0.9)):
    center = np.array(center, dtype=float)
    forward = np.asarray(target) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    translation = -rotation @ center
    camera = {"camera_matrix": [[2500., 0., 1200.], [0., 2500., 1600.], [0., 0., 1.]],
              "dist_coeffs": [-0.08, 0.02, 0.001, -0.001, 0.], "image_size": [2400, 3200],
              "status": "passed_reprojection_checks"}
    pose = {"rotation_world_to_camera": rotation.tolist(), "translation_world_to_camera_m": translation.tolist()}
    return camera, pose


def project(objects, camera, pose):
    matrix, distortion = matrices_camara(camera)
    rvec = cv2.Rodrigues(np.array(pose["rotation_world_to_camera"]))[0]
    return cv2.projectPoints(np.array(objects, dtype=float), rvec,
                            np.array(pose["translation_world_to_camera_m"]), matrix, distortion)[0].reshape(-1, 2)


@pytest.mark.parametrize("height", [1.20, 1.53, 1.71, 1.93])
@pytest.mark.parametrize("center", [(0, -3, 1.15), (0.3, -2.7, 1.3), (-0.4, -3.5, 1.1)])
def test_height_with_distortion_perspective_and_floor_scale(config, height, center):
    camera, actual_pose = scene(center)
    objects = np.concatenate(list(esquinas_objeto_piso(config).values()))
    pose = resolver_pose_piso(objects, project(objects, camera, actual_pose), camera, config)
    pixels = project([[0.1, 0.05, 0], [0.1, 0.05, height]], camera, actual_pose)
    result = altura_segmento_vertical(*pixels, camera, pose)
    assert result["height_m"] == pytest.approx(height, abs=2e-5)
    assert result["metric_accuracy_validated"] is False
    assert pose["camera_center_world_m"] == pytest.approx(center, abs=2e-5)


def test_global_wrong_print_scale_is_not_detectable_from_reprojection(config):
    camera, pose = scene()
    objects = np.concatenate(list(esquinas_objeto_piso(config).values()))
    wrong = resolver_pose_piso(objects * 0.95, project(objects, camera, pose), camera, config)
    pixels = project([[0, 0, 0], [0, 0, 1.71]], camera, pose)
    result = altura_segmento_vertical(*pixels, camera, wrong)
    assert wrong["rms_px"] < 0.001
    assert result["height_m"] == pytest.approx(1.71 * 0.95, abs=2e-5)


def test_corrupted_floor_correspondence_is_rejected(config):
    camera, pose = scene()
    objects = np.concatenate(list(esquinas_objeto_piso(config).values()))
    pixels = project(objects, camera, pose)
    pixels[0] += [40, -30]
    with pytest.raises(ValueError):
        resolver_pose_piso(objects, pixels, camera, config)


def test_invalid_camera_and_missing_markers(config):
    camera, _ = scene()
    with pytest.raises(ValueError, match="Resolución"):
        matrices_camara(camera, (1200, 1600))
    invalid = dict(camera, camera_matrix=[[float("nan"), 0, 1], [0, 1, 1], [0, 0, 1]])
    with pytest.raises(ValueError):
        matrices_camara(invalid)
    blank = np.full((3200, 2400), 255, np.uint8)
    with pytest.raises(ValueError, match="marcadores"):
        detectar_piso(blank, camera, config)


def test_manual_endpoints_must_fit_vertical_model():
    camera, pose = scene()
    points = project([[0, 0, 0], [0.2, 0, 1.7]], camera, pose)
    with pytest.raises(ValueError, match="vertical"):
        altura_segmento_vertical(*points, camera, pose)
    with pytest.raises(ValueError, match="fuera"):
        altura_segmento_vertical([-1, 0], [10, 20], camera, pose)


def test_print_kit_charuco_detection_and_duplicate_rejection(tmp_path, config):
    result = generar_kit(config, tmp_path / "kit")
    assert result["pages"] == 5
    assert 'width="160.0mm"' in (tmp_path / "kit/aruco_30.svg").read_text()
    assert 'width="150.0mm"' in (tmp_path / "kit/charuco.svg").read_text()
    with pytest.raises(ValueError, match="existe"):
        generar_kit(config, tmp_path / "kit")
    board = tablero_charuco(config)
    raster = board.generateImage((750, 1050), marginSize=0)
    raster = cv2.copyMakeBorder(raster, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    folder = tmp_path / "photos"
    folder.mkdir()
    assert cv2.imwrite(str(folder / "one.png"), raster)
    views, rejected, _, _ = recolectar_vistas(folder, config)
    assert len(views) == 1 and views[0]["corners"] == 24 and not rejected
    assert cv2.imwrite(str(folder / "duplicate.png"), raster)
    with pytest.raises(ValueError, match="duplicada"):
        recolectar_vistas(folder, config)
    with pytest.raises(ValueError, match="distintas"):
        calibrate(folder, folder, config, "test")


def test_intrinsics_fit_and_heldout_error_gate(config):
    rng = np.random.default_rng(26)
    objects = tablero_charuco(config).getChessboardCorners().reshape(-1, 1, 3)
    matrix = np.array([[1600., 0., 1200.], [0., 1600., 1600.], [0., 0., 1.]])
    distortion = np.array([-0.05, 0.01, 0.001, -0.001, 0.])
    views = []
    for i in range(30):
        rvec = rng.uniform(-0.6, 0.6, 3)
        tvec = np.array([rng.uniform(-0.42, 0.28), rng.uniform(-0.58, 0.4), rng.uniform(0.65, 0.85)])
        pixels = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0]
        views.append({"object_points": objects.copy(), "image_points": pixels.astype(np.float32),
                      "file": f"synthetic_{i}", "corners": len(objects)})
    result = ajustar_camara(views[:24], views[24:], (2400, 3200), config, "synthetic")
    assert np.array(result["camera_matrix"]) == pytest.approx(matrix, abs=0.05)
    assert result["status"] == "passed_reprojection_checks", result["warnings"]
    assert len(result["views"]["validation"]) == 6
    assert result["metric_accuracy_validated"] is False
    views[-1]["image_points"][0] += [30, -20]
    failed = ajustar_camara(views[:24], views[24:], (2400, 3200), config, "synthetic")
    assert failed["status"] == "needs_recapture"
    with pytest.raises(ValueError, match="insuficientes"):
        ajustar_camara(views[:3], views[24:], (2400, 3200), config, "synthetic")


def test_floor_detector_on_rendered_scene(config):
    camera, pose = scene()
    camera["dist_coeffs"] = [0.] * 5  # Homografías de raster representan lente ideal.
    raster = np.full((3200, 2400), 255, dtype=np.uint8)
    source = np.array([[0, 0], [599, 0], [599, 599], [0, 599]], dtype=np.float32)
    for marker_id, objects in esquinas_objeto_piso(config).items():
        destination = project(objects, camera, pose).astype(np.float32)
        marker = cv2.aruco.generateImageMarker(dictionary(config), marker_id, 600)
        homography = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(marker, homography, (2400, 3200), borderValue=255)
        raster = np.minimum(raster, warped)
    detected = detectar_piso(raster, camera, config)
    assert sorted(detected["marker_ids"]) == [30, 31, 32, 33]
    assert detected["camera_center_world_m"] == pytest.approx([0., -3., 1.15], abs=0.02)


def test_printed_svg_symbols_are_lossless_and_decodable(tmp_path, config):
    generar_kit(config, tmp_path)
    detector = cv2.aruco.ArucoDetector(dictionary(config))
    for marker_id in [30, 31, 32, 33]:
        svg = ET.parse(tmp_path / f"aruco_{marker_id}.svg").getroot()
        shape = [int(n) for n in svg.attrib["viewBox"].split()][2:]
        raster = np.full((shape[1], shape[0]), 255, np.uint8)
        path = svg.find("{http://www.w3.org/2000/svg}path").attrib["d"]
        segments = re.findall(r"M(\d+) (\d+)h(\d+)v(\d+)h(-\d+)z", path)
        assert segments
        for x, y, width, height, _ in segments:
            x, y, width, height = map(int, (x, y, width, height))
            raster[y:y+height, x:x+width] = 0
        assert np.array_equal(raster, cv2.aruco.generateImageMarker(dictionary(config), marker_id, 600))
        padded = cv2.copyMakeBorder(raster, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        _, ids, _ = detector.detectMarkers(padded)
        assert ids.ravel().tolist() == [marker_id]


def test_same_photo_not_allowed_across_calibration_and_validation(tmp_path, config):
    train, validation = tmp_path / "train", tmp_path / "validation"
    train.mkdir()
    validation.mkdir()
    image = tablero_charuco(config).generateImage((750, 1050), marginSize=0)
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    assert cv2.imwrite(str(train / "a.png"), image)
    assert cv2.imwrite(str(validation / "renamed.png"), image)
    with pytest.raises(ValueError, match="duplicada"):
        calibrate(train, validation, config, "test")


def marker_sheet(config, ids):
    """Vista plana sintética: demuestra lectura, NO pose de una estación real."""
    raster = np.full((500, 500), 255, np.uint8)
    for marker_id, (x, y) in zip(ids, [(35, 35), (285, 35), (35, 285), (285, 285)]):
        raster[y:y+180, x:x+180] = cv2.aruco.generateImageMarker(dictionary(config), marker_id, 180)
    return raster


def test_floor_preflight_does_not_claim_metric_pose(config):
    result = inspeccionar_marcadores_piso(marker_sheet(config, [30, 31, 32, 33]), config)
    assert result['status'] == 'marker_visibility_passed'
    assert result['marker_ids'] == [30, 31, 32, 33]
    assert not result['physical_layout_verified']
    assert not result['metric_accuracy_validated']
    assert 'camera_center_world_m' not in result
    camera, _ = scene()
    camera.update(image_size=[500, 500], camera_matrix=[[400, 0, 250], [0, 400, 250], [0, 0, 1]],
                  status='needs_recapture')
    with pytest.raises(ValueError, match='perfil'):
        detectar_piso(marker_sheet(config, [30, 31, 32, 33]), camera, config)


@pytest.mark.parametrize('ids', [[], [0, 1, 2, 3], [30, 31], [30, 31, 32, 30]])
def test_floor_preflight_rejects_missing_or_duplicate_ids(config, ids):
    result = inspeccionar_marcadores_piso(marker_sheet(config, ids), config)
    assert result['status'] == 'needs_capture_adjustment'
    assert result['warnings']


def test_floor_preflight_size_gate(config):
    config['quality']['min_marker_edge_px'] = 190
    result = inspeccionar_marcadores_piso(marker_sheet(config, [30, 31, 32, 33]), config)
    assert result['status'] == 'needs_capture_adjustment'
    assert result['min_marker_edge_px'] < 190


def test_floor_preflight_cli_saves_report_and_refuses_overwrite(tmp_path, config, monkeypatch, capsys):
    from src.calibration.cli import main
    photo = tmp_path / 'station.png'
    output = tmp_path / 'visibility.json'
    assert cv2.imwrite(str(photo), marker_sheet(config, [30, 31, 32, 33]))
    monkeypatch.setattr(sys, 'argv', ['calibration', 'inspect-floor', '--image', str(photo), '--output', str(output)])
    assert main() == 0
    report = leer_json(output)
    assert report['status'] == 'marker_visibility_passed'
    assert report['image_sha256'] and report['station_config_sha256']
    assert 'camera_sha256' not in report
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == original
    capsys.readouterr()


def test_floor_preflight_cli_failure_is_saved(tmp_path, config, monkeypatch, capsys):
    from src.calibration.cli import main
    photo, output = tmp_path / 'board.png', tmp_path / 'visibility.json'
    assert cv2.imwrite(str(photo), marker_sheet(config, [0, 1, 2, 3]))
    monkeypatch.setattr(sys, 'argv', ['calibration', 'inspect-floor', '--image', str(photo), '--output', str(output)])
    assert main() == 2
    report = leer_json(output)
    assert report['status'] == 'needs_capture_adjustment'
    assert report['marker_ids'] == []
    assert report['other_marker_ids'] == [0, 1, 2, 3]
    capsys.readouterr()
