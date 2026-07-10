"""
Evaluation metrics for top-K recommendation.

This is the shared "ruler" every model in the project is measured with — baselines,
MF-BPR, two-tower, and the LightGBM ranker all call `evaluate_recommendations`.
Keeping it in one place guarantees every model is scored identically.

Formulas follow Session 1 (Foundations), slides 32-34:
  * Recall@K    = |recommended_K ∩ relevant| / |relevant|
                  "of the items the user actually liked, what fraction did we find?"
  * Precision@K = |recommended_K ∩ relevant| / K
                  "of the K items we showed, what fraction were correct?"
  * NDCG@K      = DCG@K / IDCG@K,  DCG@K = Σ_{i=1..K} rel_i / log2(i + 1)
                  rank-aware: a hit at position 1 is worth more than at position 20.
  * Coverage    = |distinct items ever recommended| / |catalog|
                  a model that only shows blockbusters has high accuracy, low coverage.

Course guidance (slide 36): retrieval optimizes Recall (find all relevant),
ranking optimizes NDCG (order them well).
"""

from __future__ import annotations

import numpy as np


def _dcg(relevance: np.ndarray) -> float:
    """Discounted Cumulative Gain of a ranked 0/1 relevance vector.

    Position i (1-indexed in the slides) has discount log2(i + 1); here the array
    is 0-indexed so element j uses log2(j + 2) = log2((j+1) + 1) — identical.
    """
    return float(np.sum(relevance / np.log2(np.arange(2, relevance.size + 2))))


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    n_items: int,
    ks: tuple[int, ...] = (10, 20, 50),
    coverage_k: int = 20,
) -> dict[str, float]:
    """Average top-K metrics over all evaluated users.

    Args:
        recommendations: user_idx -> ranked list of recommended item_idx (best first).
        ground_truth:    user_idx -> set of item_idx the user positively interacted
                         with in the test period (the "correct answers").
        n_items:         total catalog size (for coverage).
        ks:              the K values to report Recall/Precision/NDCG for.
        coverage_k:      list length at which catalog coverage is measured.

    Returns:
        {'Recall@20': ..., 'Precision@20': ..., 'NDCG@10': ..., 'Coverage@20': ...}
    """
    max_k = max(ks)
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    precisions: dict[int, list[float]] = {k: [] for k in ks}
    ndcgs: dict[int, list[float]] = {k: [] for k in ks}
    recommended_pool: set[int] = set()

    for user, relevant in ground_truth.items():
        if not relevant:
            continue
        ranked = recommendations.get(user, [])[:max_k]
        recommended_pool.update(ranked[:coverage_k])

        # 1 where a recommended item is a correct answer, else 0 — in rank order
        rel = np.fromiter((1.0 if it in relevant else 0.0 for it in ranked),
                          dtype=np.float64, count=len(ranked))

        for k in ks:
            topk = rel[:k]
            hits = topk.sum()
            recalls[k].append(hits / len(relevant))
            precisions[k].append(hits / k)
            ideal_hits = min(len(relevant), k)
            idcg = _dcg(np.ones(ideal_hits)) if ideal_hits else 0.0
            ndcgs[k].append(_dcg(topk) / idcg if idcg > 0 else 0.0)

    results: dict[str, float] = {}
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        results[f"Precision@{k}"] = float(np.mean(precisions[k])) if precisions[k] else 0.0
        results[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    results[f"Coverage@{coverage_k}"] = len(recommended_pool) / n_items
    return results
