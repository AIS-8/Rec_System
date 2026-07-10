"""
FAISS helpers for approximate nearest-neighbour retrieval over item embeddings.

At serving time we can't score every one of the ~20k items for every request, so
we index the item vectors once and retrieve the top-K by inner product in
~O(log n). We use INNER PRODUCT because our score is a dot product (p_u · q_i,
with the item bias folded in as an extra dimension — see MFBPR.item_vectors_for_faiss).

  * Flat  — exact inner-product search. Best for small/medium catalogs (ours).
  * IVF   — inverted-file index; approximate but faster on large catalogs.
"""

from __future__ import annotations

import faiss
import numpy as np


def build_flat_ip(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index (IndexFlatIP)."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_ivf_ip(vectors: np.ndarray, nlist: int = 100, nprobe: int = 10) -> faiss.Index:
    """Approximate inverted-file index over inner product.

    nlist  = number of Voronoi cells to partition the space into.
    nprobe = how many cells to actually scan at query time (recall/speed knob).
    """
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int):
    """Return (scores, item_ids), each shape [len(queries), k]."""
    queries = np.ascontiguousarray(queries, dtype="float32")
    scores, ids = index.search(queries, k)
    return scores, ids
