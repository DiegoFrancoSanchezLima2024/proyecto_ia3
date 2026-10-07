"""
tests/test_model.py
====================
Tests de forward pass para todos los modulos del modelo.
Ejecutar: pytest tests/ -v
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─── Encoder ─────────────────────────────────────────────────────────────────
def test_silhouette_encoder_forward():
    from src.models.encoder import SilhouetteEncoder
    enc = SilhouetteEncoder(embed_dim=64, pretrained=False)
    x   = torch.randn(2, 1, 128, 128)
    out = enc(x)
    assert out.shape == (2, 64), f"Encoder output shape: {out.shape}"


def test_encoder_freeze_unfreeze():
    from src.models.encoder import SilhouetteEncoder
    enc = SilhouetteEncoder(embed_dim=64, pretrained=False, frozen_epochs=2)
    # Antes del epoch 2: backbone congelado
    frozen_params = [p for p in enc.backbone.parameters() if p.requires_grad]
    assert len(frozen_params) == 0, "Backbone debe estar congelado inicialmente"
    # frozen_epochs=2 significa dos epocas completas congeladas.
    enc.on_epoch_start(2)
    assert not [p for p in enc.backbone.parameters() if p.requires_grad]
    # En epoch 3 se descongela el ultimo bloque.
    enc.on_epoch_start(3)
    unfrozen = [p for p in enc.backbone.parameters() if p.requires_grad]
    assert len(unfrozen) > 0, "Backbone debe descongelarse en epoch frozen_epochs"


def test_batchnorm_adapts_only_for_configured_epochs():
    from src.models.encoder import SilhouetteEncoder
    enc = SilhouetteEncoder(
        embed_dim=32,
        pretrained=False,
        frozen_epochs=10,
        adapt_batchnorm_epochs=1,
    )
    batchnorms = [
        module
        for module in enc.backbone.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    enc.on_epoch_start(1)
    enc.train()
    assert any(module.training for module in batchnorms)
    enc.on_epoch_start(2)
    enc.train()
    assert not any(module.training for module in batchnorms)


# ─── Aggregator ──────────────────────────────────────────────────────────────
def test_multiview_aggregator_forward():
    from src.models.attention import MultiViewAggregator
    agg  = MultiViewAggregator(embed_dim=64, n_heads=4, n_layers=2, n_views=2)
    v1   = torch.randn(3, 64)
    v2   = torch.randn(3, 64)
    out  = agg([v1, v2])
    assert out.shape == (3, 64), f"Aggregator output shape: {out.shape}"


# ─── Regressor ───────────────────────────────────────────────────────────────
def test_regressor_forward_with_geometry():
    from src.models.regressor import RegresorMedidasCorporales
    model = RegresorMedidasCorporales(
        n_measures=13, embed_dim=64, n_heads=4, n_layers=2,
        pretrained_encoder=False, use_geometry=True
    )
    B = 2
    front   = torch.randn(B, 1, 128, 128)
    left    = torch.randn(B, 1, 128, 128)
    meta    = torch.randn(B, 2)
    geo_est = torch.randn(B, 13)
    delta, log_var = model([front, left], meta, geo_est)
    assert delta.shape   == (B, 13)
    assert log_var.shape == (B, 13)


def test_regressor_forward_without_geometry():
    from src.models.regressor import RegresorMedidasCorporales
    model = RegresorMedidasCorporales(
        n_measures=13, embed_dim=64, n_heads=4, n_layers=2,
        pretrained_encoder=False, use_geometry=False
    )
    B = 2
    front = torch.randn(B, 1, 128, 128)
    left  = torch.randn(B, 1, 128, 128)
    meta  = torch.randn(B, 2)
    delta, log_var = model([front, left], meta, geo_est=None)
    assert delta.shape == (B, 13)


def test_regressor_forward_with_weight_metadata():
    from src.models.regressor import RegresorMedidasCorporales
    model = RegresorMedidasCorporales(
        n_measures=13,
        embed_dim=64,
        n_heads=4,
        n_layers=1,
        pretrained_encoder=False,
        use_geometry=False,
        meta_dim=3,
    )
    views = [torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128)]
    delta, log_var = model(views, torch.randn(2, 3))
    assert delta.shape == (2, 13)
    assert log_var.shape == (2, 13)

    with pytest.raises(ValueError, match="meta"):
        model(views, torch.randn(2, 2))


def test_regressor_accepts_height_as_only_metadata():
    from src.models.regressor import RegresorMedidasCorporales
    model = RegresorMedidasCorporales(
        n_measures=14,
        embed_dim=64,
        n_heads=4,
        n_layers=1,
        pretrained_encoder=False,
        use_geometry=False,
        meta_dim=1,
    )
    views = [torch.randn(2, 1, 128, 96), torch.randn(2, 1, 128, 96)]
    prediction, uncertainty = model(views, torch.randn(2, 1))
    assert prediction.shape == (2, 14)
    assert uncertainty.shape == (2, 14)


def test_regressor_forward_with_spatial_profiles():
    from src.models.regressor import RegresorMedidasCorporales
    model = RegresorMedidasCorporales(
        n_measures=13,
        embed_dim=64,
        n_heads=4,
        n_layers=1,
        n_views=2,
        pretrained_encoder=False,
        use_geometry=False,
        use_profile_features=True,
        profile_bins=16,
    )
    front = torch.full((2, 1, 128, 128), -1.0)
    left = torch.full((2, 1, 128, 128), -1.0)
    front[:, :, 20:110, 40:88] = 1.0
    left[:, :, 20:110, 50:78] = 1.0
    profiles = model._profile_features([front, left])
    assert profiles.shape == (2, 32)
    assert torch.all((profiles >= 0.0) & (profiles <= 1.0))
    delta, log_var = model([front, left], torch.zeros(2, 2))
    assert delta.shape == (2, 13)
    assert log_var.shape == (2, 13)


# ─── Loss ─────────────────────────────────────────────────────────────────────
def test_total_loss():
    from src.training.losses import PerdidaTotal
    criterion = PerdidaTotal(huber_delta=2.0, lambda_coherence=0.1)
    pred    = torch.randn(4, 13)
    target  = torch.randn(4, 13)
    log_var = torch.zeros(4, 13)
    result  = criterion(pred, target, log_var)
    assert "total" in result
    assert result["total"].item() > 0


def test_measurement_weights_prioritize_selected_errors():
    from src.training.losses import PerdidaTotal
    pred = torch.tensor([[2.0, 0.5]])
    target = torch.zeros_like(pred)
    log_var = torch.zeros_like(pred)
    unweighted = PerdidaTotal(mode="huber")(pred, target, log_var)["main"]
    weighted = PerdidaTotal(
        mode="huber", measurement_weights=[2.0, 1.0]
    )(pred, target, log_var)["main"]
    assert weighted > unweighted


# ─── Geometric estimator ──────────────────────────────────────────────────────
def test_geometric_estimator():
    from src.geometry.geometric_est import EstimadorGeometrico
    measures = ["chest", "waist", "hip", "arm-length", "calf"]
    geo_est  = EstimadorGeometrico(measures)
    # Crear siluetas dummy (HxW uint8)
    front = np.zeros((128, 128), dtype=np.float32)
    front[20:110, 40:88] = 1.0    # cuerpo simplificado
    left  = np.zeros((128, 128), dtype=np.float32)
    left[20:110, 45:83]  = 1.0
    # Pasar como batch [B, 1, H, W] en formato normalizado [-1, 1]
    front_b = front[None, None] * 2 - 1   # [1, 1, H, W]
    left_b  = left[None,  None] * 2 - 1
    heights = np.array([175.0])
    out = geo_est(front_b, left_b, heights)
    assert out.shape == (1, 5), f"EstimadorGeometrico output shape: {out.shape}"


if __name__ == "__main__":
    test_silhouette_encoder_forward()
    test_encoder_freeze_unfreeze()
    test_multiview_aggregator_forward()
    test_regressor_forward_with_geometry()
    test_regressor_forward_without_geometry()
    test_total_loss()
    test_geometric_estimator()
    print("\n[OK] Todos los tests de modelo pasaron.")
