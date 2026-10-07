"""
src/models/attention.py
========================
Agregador de atencion multivista ligero.
Combina los embeddings de 2-4 vistas en un vector global.
"""
import torch
import torch.nn as nn
import math


class ViewEmbedding(nn.Module):
    """Embedding aprendido por posicion/vista (0=front,1=left,2=back,3=right)."""
    def __init__(self, n_views: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_views, embed_dim)

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(view_ids)


class MultiViewAggregator(nn.Module):
    """
    Toma una lista de embeddings de vistas y los fusiona con atencion cruzada.
    Entrada: list de tensores [B, embed_dim], uno por vista
    Salida:  tensor [B, embed_dim] fusionado
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_views: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_views   = n_views
        self.view_emb  = ViewEmbedding(n_views, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN: mas estable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Token CLS para extraer representacion global
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, view_embeddings: list, view_ids: list = None) -> torch.Tensor:
        """
        view_embeddings: lista de [B, embed_dim], longitud n_actual_views
        view_ids: lista de ints (indices de vista), default [0,1,...,n-1]
        """
        B = view_embeddings[0].shape[0]
        n = len(view_embeddings)
        if view_ids is None:
            view_ids = list(range(n))

        # Stack vistas: [B, n, embed_dim]
        tokens = torch.stack(view_embeddings, dim=1)

        # Sumar embedding de vista
        ids_t = torch.tensor(view_ids, dtype=torch.long, device=tokens.device)
        v_emb = self.view_emb(ids_t).unsqueeze(0)  # [1, n, embed_dim]
        tokens = tokens + v_emb

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, embed_dim]
        tokens = torch.cat([cls, tokens], dim=1)    # [B, n+1, embed_dim]

        # Transformer
        out = self.transformer(tokens)              # [B, n+1, embed_dim]

        # Extraer CLS
        return out[:, 0, :]                         # [B, embed_dim]
