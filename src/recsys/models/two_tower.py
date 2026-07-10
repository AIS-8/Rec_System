"""
Two-Tower retrieval model — from scratch in PyTorch.

Two independent encoders ("towers"):
  * user tower: user_id embedding → MLP → user vector u
  * item tower: [item_id embedding ; category embedding] → MLP → item vector v
Both outputs are **L2-normalized**, so the score is a cosine similarity:

    score(u, i) = û · v̂        (û, v̂ unit vectors)

Because the towers never interact until the final dot product, all item vectors
can be pre-computed offline and indexed with FAISS — retrieval is then a fast
nearest-neighbour search. (This is the generalization of MF: instead of a bare ID
lookup, each side is an MLP that can also ingest features like item category,
which helps cold-start items.)

**Training — in-batch negatives + softmax cross-entropy.** For a batch of B
positive (user, item) pairs we build the B×B score matrix û_a · v̂_b. The diagonal
holds the true pairs; every off-diagonal item is a "free" negative for that user.
We then apply softmax cross-entropy so each user's own item is the most probable
class:

    logits = (U V^T) / temperature
    loss   = cross_entropy(logits, [0, 1, ..., B-1])

The temperature sharpens the softmax (needed because cosine scores live in [-1, 1]).

Only the model lives here (imported by the notebook AND the webapp); the training
loop is written out in the notebook.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """Small feed-forward tower: in_dim → hidden... → out_dim, ReLU between layers."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_cats: int, emb: int = 64,
                 hidden: tuple[int, ...] = (128,), out_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.user_id_emb = nn.Embedding(n_users, emb)
        self.item_id_emb = nn.Embedding(n_items, emb)
        self.cat_emb = nn.Embedding(n_cats, emb)
        self.user_tower = _MLP(emb, hidden, out_dim)
        self.item_tower = _MLP(2 * emb, hidden, out_dim)   # id + category
        for e in (self.user_id_emb, self.item_id_emb, self.cat_emb):
            nn.init.normal_(e.weight, std=0.01)
        # category id per item (filled by set_item_categories); moves with .to(device)
        self.register_buffer("item_cat", torch.zeros(n_items, dtype=torch.long))

    def set_item_categories(self, cat_ids: np.ndarray) -> None:
        self.item_cat.copy_(torch.as_tensor(cat_ids, dtype=torch.long))

    def user_forward(self, u: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(self.user_id_emb(u)), dim=-1)

    def item_forward(self, i: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_id_emb(i), self.cat_emb(self.item_cat[i])], dim=-1)
        return F.normalize(self.item_tower(x), dim=-1)

    def loss(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """In-batch-negative softmax cross-entropy for a batch of positive pairs."""
        U = self.user_forward(u)                 # [B, d]
        V = self.item_forward(i)                 # [B, d]
        logits = (U @ V.t()) / self.temperature  # [B, B]; diagonal = positives
        labels = torch.arange(U.size(0), device=U.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def all_item_vectors(self, batch: int = 8192) -> np.ndarray:
        """Encode every item with the item tower → [n_items, out_dim] (L2-normalized)."""
        self.eval()
        device = self.item_cat.device
        n = self.item_id_emb.num_embeddings
        out = []
        for start in range(0, n, batch):
            idx = torch.arange(start, min(start + batch, n), device=device)
            out.append(self.item_forward(idx).cpu().numpy())
        return np.vstack(out).astype("float32")

    @torch.no_grad()
    def user_vectors(self, u: torch.Tensor) -> np.ndarray:
        """Encode users with the user tower → [len(u), out_dim] (L2-normalized)."""
        self.eval()
        return self.user_forward(u).cpu().numpy().astype("float32")
