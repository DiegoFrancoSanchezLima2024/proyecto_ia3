"""
src/training/losses.py
=======================
Funciones de perdida para el regresor antropometrico.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def huber_heteroscedastic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    delta: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Perdida Huber heterocedástica (NLL gaussiana + robustez).
    Combina la incertidumbre aleatoria con robustez ante outliers.

    L = 0.5 * exp(-log_var) * huber(pred, target, delta) + 0.5 * log_var

    Args:
        pred:    [B, N]  predicciones
        target:  [B, N]  valores reales
        log_var: [B, N]  log-varianza predicha
        delta:   umbral Huber en unidades normalizadas
    """
    huber = F.huber_loss(pred, target, delta=delta, reduction="none")  # [B, N]
    loss  = 0.5 * torch.exp(-log_var) * huber + 0.5 * log_var
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Reduccion no soportada: {reduction}")


def coherence_loss(
    pred_views: list,  # lista de [B, N] predicciones por vista individual
) -> torch.Tensor:
    """
    Penaliza inconsistencias entre las predicciones de diferentes vistas.
    Cada vista deberia predecir medidas similares.
    """
    if len(pred_views) < 2:
        device = pred_views[0].device if pred_views else "cpu"
        return torch.tensor(0.0, device=device)
    mean_pred = torch.stack(pred_views, dim=0).mean(0)
    loss = sum(F.mse_loss(p, mean_pred.detach()) for p in pred_views)
    return loss / len(pred_views)


class PerdidaTotal(nn.Module):
    """Loss total = Huber heterocedástico + lambda * coherencia."""

    def __init__(
        self,
        huber_delta: float = 2.0,
        lambda_coherence: float = 0.1,
        mode: str = "heteroscedastic_huber",
        measurement_weights: list[float] | None = None,
    ):
        super().__init__()
        self.delta    = huber_delta
        self.lam_coh  = lambda_coherence
        if mode not in {"heteroscedastic_huber", "huber"}:
            raise ValueError(f"Loss no soportada: {mode}")
        self.mode = mode
        weights = torch.as_tensor(
            measurement_weights if measurement_weights is not None else [1.0],
            dtype=torch.float32,
        )
        if weights.ndim != 1 or torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0):
            raise ValueError("Los pesos de medidas deben ser finitos y mayores que cero")
        # Mantener media=1 evita que el cambio de ponderacion altere la escala
        # global de la perdida y, por tanto, el learning rate efectivo.
        self.register_buffer("measurement_weights", weights / weights.mean())

    def forward(
        self,
        pred:    torch.Tensor,
        target:  torch.Tensor,
        log_var: torch.Tensor,
        pred_views: list = None,
    ) -> dict:
        if self.measurement_weights.numel() not in {1, pred.shape[1]}:
            raise ValueError(
                "La cantidad de pesos no coincide con las medidas predichas: "
                f"{self.measurement_weights.numel()} != {pred.shape[1]}"
            )
        if self.mode == "heteroscedastic_huber":
            elementwise = huber_heteroscedastic_loss(
                pred, target, log_var, self.delta, reduction="none"
            )
        else:
            elementwise = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        main = (elementwise * self.measurement_weights).mean()
        coh  = coherence_loss(pred_views) if pred_views else torch.tensor(0.0, device=pred.device)
        total = main + self.lam_coh * coh
        return {"total": total, "main": main, "coherence": coh}
