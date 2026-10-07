"""
src/models/regressor.py
========================
Regresor residual completo: fusion 2.5D + CNN multivista.
Predice correccion residual Δm sobre la estimacion geometrica.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SilhouetteEncoder
from .attention import MultiViewAggregator


class RegresorMedidasCorporales(nn.Module):
    """
    Pipeline completo:
      1. Encoder CNN por vista (compartido)
      2. Agregador de atencion multivista
      3. MLP con estatura + estimaciones geometricas
      4. Salida: Δm (correccion residual) + log_var (incertidumbre)

    La prediccion final es: m_hat = m_geo + Δm
    """

    def __init__(
        self,
        n_measures: int,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_views: int = 2,
        dropout: float = 0.25,
        pretrained_encoder: bool = True,
        frozen_epochs: int = 10,
        adapt_batchnorm_epochs: int = 0,
        use_geometry: bool = True,
        use_profile_features: bool = False,
        profile_bins: int = 32,
        meta_dim: int = 2,
    ):
        super().__init__()
        self.n_measures   = n_measures
        self.use_geometry = use_geometry
        self.use_profile_features = bool(use_profile_features)
        self.profile_bins = int(profile_bins)
        self.n_views = int(n_views)
        self.meta_dim = int(meta_dim)
        if self.profile_bins <= 0:
            raise ValueError("profile_bins debe ser mayor que cero")
        if self.meta_dim < 1:
            raise ValueError("meta_dim debe incluir al menos la estatura")

        # Encoder compartido (mismos pesos para todas las vistas)
        self.encoder = SilhouetteEncoder(
            embed_dim=embed_dim,
            pretrained=pretrained_encoder,
            frozen_epochs=frozen_epochs,
            adapt_batchnorm_epochs=adapt_batchnorm_epochs,
        )

        # Agregador multivista
        self.aggregator = MultiViewAggregator(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_views=n_views,
            dropout=dropout,
        )

        # Dimension de entrada al MLP
        # = embed_dim + metadatos + n_measures (geo_est si use_geometry)
        mlp_in = embed_dim + self.meta_dim
        if use_geometry:
            mlp_in += n_measures
        if self.use_profile_features:
            mlp_in += self.n_views * self.profile_bins

        # MLP de prediccion
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )

        # Cabeza de prediccion de correccion residual
        self.head_delta = nn.Linear(256, n_measures)

        # Cabeza de incertidumbre (log-varianza)
        self.head_logvar = nn.Linear(256, n_measures)

        self._init_weights()

    def _init_weights(self):
        # Una salida normalizada cercana a cero equivale a la media de train.
        # Inicializar pequeno evita predicciones antropometricas extremas antes
        # de que la cabeza aprenda, sin bloquear el gradiente como un cero exacto.
        nn.init.normal_(self.head_delta.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head_delta.bias)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.zeros_(self.head_logvar.bias)

    def on_epoch_start(self, epoch: int):
        self.encoder.on_epoch_start(epoch)

    def _profile_features(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Resume el ancho de la silueta en bandas verticales por cada vista.

        Las entradas estan normalizadas en [-1, 1]. El promedio horizontal de
        primer plano produce un perfil de ancho relativo y adaptive pooling lo
        hace independiente de la resolucion de entrada.
        """
        if len(views) != self.n_views:
            raise ValueError(
                f"Se esperaban {self.n_views} vistas para perfiles; llegaron {len(views)}"
            )
        profiles = []
        for view in views:
            foreground = (view * 0.5 + 0.5).clamp(0.0, 1.0)
            row_width = foreground.mean(dim=-1)
            profiles.append(
                F.adaptive_avg_pool1d(row_width, self.profile_bins).flatten(1)
            )
        return torch.cat(profiles, dim=1)

    def forward(
        self,
        views: list,              # lista de tensores [B, 1, H, W]
        meta: torch.Tensor,       # [B, meta_dim]: altura y metadatos opcionales
        geo_est: torch.Tensor = None,  # [B, n_measures] estimacion geometrica normalizada
    ) -> tuple:
        """
        Returns:
            delta:   [B, n_measures]  correccion residual
            log_var: [B, n_measures]  log-varianza (incertidumbre)
        """
        if not views:
            raise ValueError("Se requiere al menos una vista")
        batch_size = views[0].shape[0]
        if any(view.shape[0] != batch_size for view in views):
            raise ValueError("Todas las vistas deben tener el mismo batch size")
        if meta.ndim != 2 or meta.shape != (batch_size, self.meta_dim):
            raise ValueError(
                f"Se esperaba meta [B, {self.meta_dim}]; llego {tuple(meta.shape)}"
            )

        # En la RTX 3050 6 GB, dos llamadas secuenciales fueron mas rapidas y
        # reservaron menos memoria que concatenar las vistas en un batch 2B.
        embs = [self.encoder(view) for view in views]

        # Fusionar vistas
        fused = self.aggregator(embs)   # [B, embed_dim]

        # Concatenar contexto
        context = [fused, meta]
        if self.use_geometry and geo_est is not None:
            context.append(geo_est)
        if self.use_profile_features:
            context.append(self._profile_features(views))
        x = torch.cat(context, dim=1)

        # MLP
        feat = self.mlp(x)

        # Predicciones
        delta   = self.head_delta(feat)    # [B, n_measures]
        # Limitar la varianza evita overflow de exp(-log_var) al inicio.
        log_var = self.head_logvar(feat).clamp(-6.0, 4.0)  # [B, n_measures]

        return delta, log_var
