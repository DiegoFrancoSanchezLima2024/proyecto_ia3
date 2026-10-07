"""
src/models/encoder.py
=====================
Encoder CNN para siluetas en escala de grises.
Usa EfficientNet-B0 preentrenado de timm, adaptado a 1 canal.
"""
import torch
import torch.nn as nn
import timm
from pathlib import Path


MODEL_CACHE = Path(__file__).resolve().parents[2] / "pretrained" / "cache"


class SilhouetteEncoder(nn.Module):
    """
    EfficientNet-B0 modificado para entrada 1 canal (siluetas).
    Salida: embedding de dimension `embed_dim` por imagen.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        frozen_epochs: int = 10,
        adapt_batchnorm_epochs: int = 0,
    ):
        super().__init__()
        self.embed_dim     = embed_dim
        self.frozen_epochs = frozen_epochs
        self.adapt_batchnorm_epochs = max(0, int(adapt_batchnorm_epochs))
        self._current_epoch= 0
        self._last_block_unfrozen = False

        # Cargar EfficientNet-B0 preentrenado en ImageNet
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # sin cabeza de clasificacion
            global_pool="avg",      # global average pooling
            in_chans=1,             # 1 canal (silueta gris)
            cache_dir=str(MODEL_CACHE),
        )

        backbone_dim = self.backbone.num_features  # 1280 para EfficientNet-B0

        # Proyeccion al espacio de embedding
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Inicialmente congelar el backbone
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_block(self):
        """Descongela el ultimo bloque del backbone."""
        # EfficientNet-B0: descongelar blocks[-1] y conv_head
        for name, p in self.backbone.named_parameters():
            if "blocks.6" in name or "conv_head" in name or "bn2" in name:
                p.requires_grad = True
        self._last_block_unfrozen = True

    def train(self, mode: bool = True):
        """Mantiene BatchNorm congelado excepto en el bloque habilitado.

        ``requires_grad=False`` no congela por si solo las medias moviles de
        BatchNorm. Sin este metodo el supuesto periodo congelado cambiaba el
        backbone y hacia menos reproducible el fine-tuning.
        """
        super().train(mode)
        if mode:
            self.backbone.eval()
            # Las estadisticas ImageNet no describen mascaras binarias. Durante
            # pocas epocas se actualizan sin entrenar sus pesos (AdaBN).
            if 0 < self._current_epoch <= self.adapt_batchnorm_epochs:
                for module in self.backbone.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.train()
            if self._last_block_unfrozen:
                self.backbone.blocks[-1].train()
                self.backbone.conv_head.train()
                self.backbone.bn2.train()
        return self

    def on_epoch_start(self, epoch: int):
        self._current_epoch = epoch
        if epoch > self.frozen_epochs and not self._last_block_unfrozen:
            self._unfreeze_last_block()
            print(f"[Encoder] Epoch {epoch}: Ultimo bloque descongelado.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        returns: [B, embed_dim]
        """
        feat = self.backbone(x)   # [B, backbone_dim]
        emb  = self.proj(feat)    # [B, embed_dim]
        return emb
