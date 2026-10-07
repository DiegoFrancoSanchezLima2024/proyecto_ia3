"""Regresiones para fallos que invalidaban entrenamiento/evaluacion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bodym_dataset import ConjuntoDatosBodyM
from src.models.regressor import RegresorMedidasCorporales
from src.training.conformal import CalibradorConforme
from src.training.runtime import geometria_normalizada, resolver_raiz_imagenes_crudas
from src.training.train import (
    construir_optimizador,
    pesos_balanceo_sujetos,
    resolver_mae_seleccion,
)
from src.inference.predict_measurements import calidad_mascara
from src.inference.predict_four_views import (
    aplicar_politica_captura,
    aplicar_confirmaciones_manuales,
    fusionar_predicciones_par,
)
from src.utils.config import cargar_config
from src.patterns.bespoke import aplicar_plan_a_medida
from src.anthropometry.ansur_filter import aplicar_filtro_antropometrico
from src.features.silhouette_profiles import caracteristicas_dos_vistas
from PIL import Image, ImageDraw


def test_missing_image_is_not_silently_replaced():
    csv_path = ROOT / "dataset" / "bodym_clean" / "train" / "data.csv"
    missing_image_root = ROOT / "tests" / "fixtures" / "images_do_not_exist"
    with pytest.raises(FileNotFoundError):
        ConjuntoDatosBodyM(
            str(csv_path),
            str(missing_image_root),
            ["chest"],
            "height_cm",
            "gender",
            strict_images=True,
        )


def test_geometry_is_normalized_before_residual_addition():
    class DummyGeometry:
        def __call__(self, front, left, heights):
            assert heights[0] == pytest.approx(175.0)
            return np.array([[110.0, 80.0]], dtype=np.float32)

    output = geometria_normalizada(
        DummyGeometry(),
        torch.zeros(1, 1, 8, 8),
        torch.zeros(1, 1, 8, 8),
        torch.tensor([[0.5, 1.0]]),
        np.array([100.0, 70.0], dtype=np.float32),
        np.array([10.0, 5.0], dtype=np.float32),
        170.0,
        10.0,
        torch.device("cpu"),
    )
    assert torch.allclose(output, torch.tensor([[1.0, 2.0]]))


def test_test_split_resolves_to_testa():
    cfg = {
        "paths": {"bodym_root": "dataset/bodym"},
        "dataset": {"raw_split_directories": {"test": "testA", "test_wild": "testB"}},
    }
    synthetic_root = Path("C:/sastre-ia-test-root")
    assert resolver_raiz_imagenes_crudas(synthetic_root, cfg, "test").name == "testA"
    assert resolver_raiz_imagenes_crudas(synthetic_root, cfg, "test_wild").name == "testB"


def test_optimizer_keeps_frozen_backbone_parameters():
    model = RegresorMedidasCorporales(
        n_measures=3,
        embed_dim=32,
        n_heads=4,
        n_layers=1,
        pretrained_encoder=False,
        use_geometry=False,
    )
    cfg = {
        "training": {
            "learning_rate": 2e-4,
            "encoder_learning_rate": 3e-5,
            "weight_decay": 1e-4,
        }
    }
    optimizer = construir_optimizador(model, cfg)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}


def test_conformal_uses_finite_sample_order_statistic():
    calibrator = CalibradorConforme(coverage=0.9)
    predictions = np.zeros((9, 1), dtype=np.float32)
    targets = np.arange(1, 10, dtype=np.float32)[:, None]
    calibrator.calibrate(predictions, targets)
    assert calibrator.q_hat[0] == pytest.approx(9.0)


def test_experiment_config_inherits_validated_defaults():
    cfg = cargar_config(ROOT / "configs" / "exp_003_weighted_huber.yaml")
    assert cfg["training"]["batch_size"] == 16
    assert cfg["training"]["loss"] == "huber"
    assert cfg["training"]["measurement_weights"]["waist"] == pytest.approx(2.0)


def test_profile_experiment_changes_only_intended_model_branch():
    cfg = cargar_config(ROOT / "configs" / "exp_004_profiles_huber.yaml")
    assert cfg["training"]["batch_size"] == 16
    assert cfg["training"]["loss"] == "huber"
    assert cfg["model"]["use_geometry"] is False
    assert cfg["model"]["use_profile_features"] is True
    assert cfg["model"]["profile_bins"] == 32


def test_subject_balancing_gives_each_person_equal_probability_mass():
    frame = pd.DataFrame({"subject_id": ["a", "a", "a", "b"]})
    weights = pesos_balanceo_sujetos(frame).numpy()
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_subject_balanced_experiment_keeps_winning_architecture():
    cfg = cargar_config(ROOT / "configs" / "exp_005_subject_balanced_huber.yaml")
    assert cfg["training"]["subject_balanced_sampling"] is True
    assert cfg["training"]["loss"] == "huber"
    assert cfg["model"]["use_geometry"] is False
    assert cfg["model"]["use_profile_features"] is False


def test_weight_experiment_is_explicit_and_keeps_huber_baseline():
    cfg = cargar_config(ROOT / "configs" / "exp_006_weight_huber.yaml")
    assert cfg["dataset"]["use_weight_meta"] is True
    assert cfg["dataset"]["weight_col"] == "weight_kg"
    assert cfg["training"]["loss"] == "huber"
    assert cfg["model"]["use_geometry"] is False


def test_two_view_multitask_experiment_has_no_weight_or_gender_input():
    cfg = cargar_config(ROOT / "configs" / "exp_007_multitarea_sin_peso.yaml")
    assert cfg["dataset"]["views"] == ["front", "left"]
    assert cfg["dataset"]["img_size"] == [320, 240]
    assert cfg["dataset"]["use_weight_meta"] is False
    assert cfg["dataset"]["use_gender_meta"] is False
    assert "weight_kg" in cfg["dataset"]["measurements"]
    assert "weight_kg" not in cfg["training"]["selection_measurements"]


def test_checkpoint_selection_excludes_auxiliary_weight():
    metrics = {
        "chest": {"mae": 1.0},
        "waist": {"mae": 2.0},
        "weight_kg": {"mae": 99.0},
    }
    cfg = {"training": {"selection_measurements": ["chest", "waist"]}}
    assert resolver_mae_seleccion(
        metrics, cfg, ["chest", "waist", "weight_kg"]
    ) == pytest.approx(1.5)


def test_physical_profiles_are_invariant_to_uniform_image_scale(tmp_path):
    paths = []
    for size, box in [((100, 200), (30, 10, 69, 189)), ((200, 400), (60, 20, 139, 379))]:
        image = Image.new("L", size, 0)
        ImageDraw.Draw(image).rectangle(box, fill=255)
        path = tmp_path / f"mask_{size[0]}.png"
        image.save(path)
        paths.append(path)
    small = caracteristicas_dos_vistas(paths[0], paths[0], 180.0, 16)
    large = caracteristicas_dos_vistas(paths[1], paths[1], 180.0, 16)
    assert np.allclose(small, large, atol=0.6)


def test_mask_quality_detects_empty_and_full_body_masks():
    empty = torch.full((1, 64, 64), -1.0)
    empty_quality = calidad_mascara(empty)
    assert empty_quality["warnings"]

    body = torch.full((1, 64, 64), -1.0)
    body[:, 4:62, 20:44] = 1.0
    body_quality = calidad_mascara(body)
    assert body_quality["vertical_coverage"] > 0.85
    assert not body_quality["warnings"]


def _pair_result(chest: float, waist: float, hip: float, warnings=None):
    measurements = {}
    for name, value in {"chest": chest, "waist": waist, "hip": hip}.items():
        measurements[name] = {
            "estimate_cm": value,
            "lower_cm": value - 2.0,
            "upper_cm": value + 2.0,
            "requires_manual_confirmation": False,
        }
    return {
        "measurements": measurements,
        "quality": {"warnings": list(warnings or [])},
    }


def test_four_view_fusion_averages_pairs_and_keeps_conservative_interval():
    left = _pair_result(100.0, 80.0, 95.0)
    right = _pair_result(102.0, 81.0, 94.0)
    result = fusionar_predicciones_par(left, right, max_critical_disagreement_cm=3.0)

    assert result["decision"] == "accepted"
    assert result["measurements"]["chest"]["estimate_cm"] == pytest.approx(101.0)
    assert result["measurements"]["chest"]["lower_cm"] == pytest.approx(98.0)
    assert result["measurements"]["chest"]["upper_cm"] == pytest.approx(104.0)
    assert result["measurements"]["chest"]["side_disagreement_cm"] == pytest.approx(2.0)


def test_four_view_fusion_rejects_critical_disagreement_or_bad_capture():
    left = _pair_result(100.0, 80.0, 95.0)
    right = _pair_result(104.0, 80.0, 95.0)
    result = fusionar_predicciones_par(left, right, max_critical_disagreement_cm=3.0)
    assert result["decision"] == "repeat_capture"
    assert result["critical_disagreements"] == ["chest"]
    assert result["measurements"]["chest"]["requires_manual_confirmation"]

    right = _pair_result(100.0, 80.0, 95.0, warnings=["cuerpo cortado"])
    result = fusionar_predicciones_par(left, right, max_critical_disagreement_cm=3.0)
    assert result["decision"] == "repeat_capture"


def test_capture_policy_rejects_loose_clothing_and_propagates_warnings():
    decision, warnings = aplicar_politica_captura("accepted", [], "loose")
    assert decision == "repeat_capture"
    assert any("ropa holgada" in warning for warning in warnings)

    decision, warnings = aplicar_politica_captura(
        "accepted", ["left: persona toca el borde"], "tight"
    )
    assert decision == "repeat_capture"
    assert warnings == ["left: persona toca el borde"]

    decision, warnings = aplicar_politica_captura("accepted", [], "unknown")
    assert decision == "manual_confirmation"
    assert warnings


def test_manual_confirmations_preserve_model_and_define_assisted_final_values():
    result = {
        "decision": "manual_confirmation",
        "measurements": {
            "chest": {"estimate_cm": 102.16, "requires_manual_confirmation": True},
            "waist": {"estimate_cm": 92.24, "requires_manual_confirmation": True},
        },
    }
    aplicar_confirmaciones_manuales(result, {"chest": 98.0})

    chest = result["measurements"]["chest"]
    assert chest["estimate_cm"] == pytest.approx(102.16)
    assert chest["final_cm"] == pytest.approx(98.0)
    assert chest["model_error_cm"] == pytest.approx(4.16)
    assert chest["source"] == "manual_tape"
    assert result["measurements"]["waist"]["source"] == "model_estimate"
    assert result["confirmation"]["status"] == "partial"


def test_manual_confirmations_reject_unknown_or_impossible_values():
    result = {
        "decision": "accepted",
        "measurements": {
            "chest": {"estimate_cm": 100.0, "requires_manual_confirmation": False}
        },
    }
    with pytest.raises(ValueError, match="desconocidas"):
        aplicar_confirmaciones_manuales(result, {"neck": 40.0})
    with pytest.raises(ValueError, match="inválido"):
        aplicar_confirmaciones_manuales(result, {"chest": -1.0})


def test_bespoke_plan_uses_confirmed_centimeters_without_sizes():
    result = {
        "measurements": {
            "chest": {
                "estimate_cm": 102.16,
                "final_cm": 98.0,
                "source": "manual_tape",
            },
            "waist": {
                "estimate_cm": 92.25,
                "final_cm": 87.0,
                "source": "manual_tape",
            },
            "hip": {
                "estimate_cm": 98.99,
                "lower_cm": 93.0,
                "upper_cm": 105.0,
                "source": "model_estimate",
            },
        }
    }
    config = {
        "config_id": "test",
        "sex_profiles": {
            "male": {"garments": {"jacket": {
                "style_profile": "test",
                "required_measurements": ["chest", "waist", "hip"],
                "ease_cm": {"chest": 10, "waist": 8, "hip": 8},
                "seam_allowance_cm": 1.5,
                "hem_allowance_cm": 4,
                "pattern_pieces": ["front", "back"],
            }}},
        },
    }
    aplicar_plan_a_medida(result, gender_code=1, height_cm=164, config=config)

    jacket = result["bespoke"]["garments"]["jacket"]
    assert result["measurements"]["chest"]["final_cm"] == pytest.approx(98.0)
    assert jacket["body_measurements_cm"]["chest"] == pytest.approx(98.0)
    assert jacket["finished_garment_measurements_cm"]["chest"] == pytest.approx(108.0)
    assert jacket["pattern_status"] == "draft_ready"
    assert "sizing" not in result


def test_bespoke_plan_selects_only_requested_sex_garments():
    result = {
        "decision": "accepted",
        "measurements": {
            "waist": {"estimate_cm": 70, "source": "calibrated_3d"},
            "hip": {"estimate_cm": 94, "source": "calibrated_3d"},
        },
    }
    config = {
        "config_id": "test",
        "sex_profiles": {
            "female": {"garments": {"skirt": {
                "style_profile": "test",
                "required_measurements": ["waist", "hip"],
                "ease_cm": {"waist": 2, "hip": 4},
                "seam_allowance_cm": 1.5,
                "hem_allowance_cm": 4,
                "pattern_pieces": ["front_skirt", "back_skirt"],
            }}},
        },
    }
    aplicar_plan_a_medida(result, gender_code=0, height_cm=149, config=config)
    assert set(result["bespoke"]["garments"]) == {"skirt"}
    assert result["bespoke"]["overall_pattern_status"] == "cut_ready"


def test_anthropometric_filter_flags_without_changing_centimeters():
    result = {
        "measurements": {
            "chest": {
                "estimate_cm": 130.0,
                "final_cm": 130.0,
                "source": "model_estimate",
                "requires_manual_confirmation": False,
            }
        }
    }
    reference = {
        "reference_id": "test",
        "sex_models": {
            "male": {
                "input_domain_p01_p99": {
                    "height_cm": [150, 200],
                    "weight_kg": [45, 140],
                },
                "measurements": {
                    "chest": {
                        "ansur_column": "chestcircumference",
                        "coefficients": {
                            "intercept": 50,
                            "height_cm": 0.1,
                            "weight_kg": 0.4,
                        },
                        "residual_interval_cm": [-5, 5],
                    }
                },
            }
        },
    }
    aplicar_filtro_antropometrico(result, 1, 170, 70, reference)

    assert result["measurements"]["chest"]["final_cm"] == pytest.approx(130.0)
    assert result["measurements"]["chest"]["anthropometric_status"] == "outlier"
    assert result["measurements"]["chest"]["requires_manual_confirmation"]
    assert result["anthropometric_check"]["flagged_measurements"] == ["chest"]


def test_anthropometric_filter_disables_outside_reference_domain():
    result = {"measurements": {"chest": {"estimate_cm": 100.0}}}
    reference = {
        "reference_id": "test",
        "sex_models": {
            "male": {
                "input_domain_p01_p99": {
                    "height_cm": [160, 190],
                    "weight_kg": [50, 120],
                },
                "measurements": {},
            }
        },
    }
    aplicar_filtro_antropometrico(result, 1, 150, None, reference)
    assert not result["anthropometric_check"]["active"]
    assert "weight_kg_missing" in result["anthropometric_check"]["inactive_reasons"]


def test_anthropometric_filter_accepts_overlapping_model_interval():
    result = {
        "measurements": {
            "chest": {
                "estimate_cm": 89.0,
                "lower_cm": 86.0,
                "upper_cm": 94.0,
                "source": "model_estimate",
                "requires_manual_confirmation": False,
            }
        }
    }
    reference = {
        "reference_id": "test",
        "sex_models": {
            "male": {
                "input_domain_p01_p99": {
                    "height_cm": [150, 200],
                    "weight_kg": [45, 140],
                },
                "measurements": {
                    "chest": {
                        "ansur_column": "chestcircumference",
                        "coefficients": {
                            "intercept": 50,
                            "height_cm": 0.1,
                            "weight_kg": 0.4,
                        },
                        "residual_interval_cm": [-5, 5],
                    }
                },
            }
        },
    }
    aplicar_filtro_antropometrico(result, 1, 170, 70, reference)
    assert result["measurements"]["chest"]["anthropometric_status"] == "plausible"
    assert not result["anthropometric_check"]["flagged_measurements"]
