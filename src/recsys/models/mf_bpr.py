"""
Matrix Factorization with BPR (Bayesian Personalized Ranking) — from scratch in PyTorch.

The model learns a k-dim embedding per user (P) and per item (Q), plus a per-item
bias. The predicted preference score is the dot product:

    r_hat(u, i) = p_u · q_i + b_i

BPR trains on triples (u, i, j): user u, a positive item i (interacted), and a
sampled negative item j (not interacted). It maximises the probability that the
user prefers i over j — a *pairwise ranking* objective, which is what we care
about for top-K retrieval (rather than predicting an exact rating):

    L = -mean( ln σ( r_hat(u,i) - r_hat(u,j) ) ) + λ · (‖p_u‖² + ‖q_i‖² + ‖q_j‖²)

Negatives are drawn with **popularity-based sampling** (proportional to how often
an item appears in training): popular items are ones the user very likely saw but
didn't engage with, so they are informative "hard-ish" negatives — better signal
than uniform-random items the model already ranks low.

Only the model + samplers live here (imported by the training notebook AND the
webapp API, so the served model matches what we trained). The explicit training
loop is written out in the notebook for readability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFBPR(nn.Module):
    """Matrix-factorization scorer trained with the BPR pairwise loss."""

    def __init__(self, n_users: int, n_items: int, k: int = 64):
        super().__init__()
        self.n_users, self.n_items, self.k = n_users, n_items, k
        self.user_emb = nn.Embedding(n_users, k)   # P: one row per user
        self.item_emb = nn.Embedding(n_items, k)    # Q: one row per item
        self.item_bias = nn.Embedding(n_items, 1)   # b_i: item popularity/offset
        # small random init keeps early scores near 0 (stable BPR gradients)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """r_hat(u, i) = p_u · q_i + b_i, elementwise over the batch."""
        return (self.user_emb(u) * self.item_emb(i)).sum(-1) + self.item_bias(i).squeeze(-1)

    def bpr_loss(self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                 reg: float = 0.0) -> torch.Tensor:
        """BPR loss for a batch of (user, positive, negative) triples."""
        pu = self.user_emb(u)
        qi, qj = self.item_emb(i), self.item_emb(j)
        bi = self.item_bias(i).squeeze(-1)
        bj = self.item_bias(j).squeeze(-1)
        # score difference: want r_hat(u,i) - r_hat(u,j) to be large & positive
        x_uij = ((pu * qi).sum(-1) + bi) - ((pu * qj).sum(-1) + bj)
        loss = -F.logsigmoid(x_uij).mean()
        if reg:
            loss = loss + reg * (
                pu.pow(2).sum(-1).mean()
                + qi.pow(2).sum(-1).mean()
                + qj.pow(2).sum(-1).mean()
                + bi.pow(2).mean() + bj.pow(2).mean()
            )
        return loss

    @torch.no_grad()
    def full_scores(self, u: torch.Tensor) -> torch.Tensor:
        """Score EVERY item for each user in `u` → [len(u), n_items]. (Eval/debug;
        production uses FAISS on the exported item vectors instead.)"""
        return self.user_emb(u) @ self.item_emb.weight.t() + self.item_bias.weight.squeeze(-1)

    @torch.no_grad()
    def item_vectors_for_faiss(self) -> np.ndarray:
        """Item vectors augmented with the bias as an extra dimension, so that a
        plain inner-product search reproduces p_u · q_i + b_i exactly.

        item_aug = [q_i , b_i]   and the user query is [p_u , 1]  →  dot = p_u·q_i + b_i
        """
        q = self.item_emb.weight.detach().cpu().numpy()
        b = self.item_bias.weight.detach().cpu().numpy()  # [n_items, 1]
        return np.hstack([q, b]).astype("float32")

    @torch.no_grad()
    def user_vectors_for_faiss(self, u: torch.Tensor) -> np.ndarray:
        """User queries augmented with a constant 1 to pick up the item bias."""
        p = self.user_emb(u).detach().cpu().numpy()
        ones = np.ones((p.shape[0], 1), dtype="float32")
        return np.hstack([p, ones]).astype("float32")


class PopularityNegativeSampler:
    """Draws negative items with probability ∝ (train popularity)^power.

    power = 1.0  → sample exactly proportional to popularity (course default).
    power = 0.75 → word2vec-style dampening (optional; softens the head).
    """

    def __init__(self, item_counts: np.ndarray, power: float = 1.0, seed: int = 0):
        probs = np.asarray(item_counts, dtype=np.float64) ** power
        self.probs = torch.tensor(probs / probs.sum(), dtype=torch.float)
        self.g = torch.Generator().manual_seed(seed)

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Return n negative item indices sampled by popularity (with replacement)."""
        idx = torch.multinomial(self.probs, n, replacement=True, generator=self.g)
        return idx.to(device) if device is not None else idx
