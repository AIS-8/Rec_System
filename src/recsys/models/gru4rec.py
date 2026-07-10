"""GRU4Rec — a sequential (session-based) recommender, implemented from scratch.

A GRU reads a user's time-ordered item sequence and predicts the *next* item. It is
trained with in-batch-negative softmax cross-entropy (the same objective family as the
two-tower), so its item embeddings live in a cosine space and can be indexed with FAISS
— letting it serve as an additional retriever alongside MF-BPR and the two-tower.

    seq = [i1, i2, ..., i_{t}]  ->  GRU  ->  h_t  ->  proj  ->  predict i_{t+1}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, emb: int = 64, hidden: int = 64,
                 max_len: int = 30, temperature: float = 0.1):
        super().__init__()
        self.n_items = n_items
        self.pad_id = n_items                    # the extra last row is the padding vector
        self.max_len = max_len
        self.temperature = temperature
        self.item_emb = nn.Embedding(n_items + 1, emb, padding_idx=n_items)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, emb)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()

    def seq_repr(self, seqs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """seqs [B, L] right-padded with pad_id, lengths [B] -> L2-normalized [B, emb].

        We pack the padded batch so the GRU only reads the real items and returns the
        hidden state right after each sequence's *last* item.
        """
        e = self.item_emb(seqs)                                        # [B, L, emb]
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                                        # h [1, B, hidden]
        return F.normalize(self.proj(h[-1]), dim=-1)                   # [B, emb]

    def item_vectors(self) -> torch.Tensor:
        """L2-normalized item embedding matrix [n_items, emb] (for FAISS / scoring)."""
        return F.normalize(self.item_emb.weight[:self.n_items], dim=-1)

    def loss(self, seqs: torch.Tensor, lengths: torch.Tensor,
             targets: torch.Tensor) -> torch.Tensor:
        """In-batch-negative softmax cross-entropy: the sequence must score its true
        next item above every *other* sequence's next item in the batch."""
        v = self.seq_repr(seqs, lengths)                              # [B, emb]
        t = F.normalize(self.item_emb(targets), dim=-1)               # [B, emb]
        logits = (v @ t.t()) / self.temperature                       # [B, B]
        labels = torch.arange(len(seqs), device=seqs.device)
        return F.cross_entropy(logits, labels)
