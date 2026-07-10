"""
Non-personalized baselines (Session 1, slide 20: "Three Simple Baselines").

Every model we build later must beat these — "if your model doesn't beat
popularity, something is wrong" (slide 19).

  * Random     — sample random items. The absolute floor.
  * Popularity — recommend the most-interacted items in the TRAIN set. Same list
                 for everyone (no personalization), but a strong, cold-start-safe
                 bar that routinely beats complex models.
  * Recency    — recommend the most recently interacted items in the train set
                 (course's optional third baseline; good for trending/news).

All three exclude items a user has already seen in training, and return a ranked
list per user so the shared `evaluate_recommendations` can score them identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_seen(train: pd.DataFrame) -> dict[int, set[int]]:
    """user_idx -> set of item_idx the user interacted with in training.

    We never recommend an item the user has already seen (standard top-K protocol).
    """
    return {u: set(g) for u, g in train.groupby("user_idx")["item_idx"]}


def popularity_ranking(train: pd.DataFrame, positive_only: bool = True) -> np.ndarray:
    """Item indices sorted by training popularity (most interacted first).

    Popularity = number of (positive) interactions in the train set — the item's
    column count in the interaction matrix.
    """
    df = train[train["positive"]] if positive_only else train
    counts = df.groupby("item_idx").size().sort_values(ascending=False)
    return counts.index.to_numpy()


def recency_ranking(train: pd.DataFrame, positive_only: bool = True) -> np.ndarray:
    """Item indices sorted by most-recent training interaction first."""
    df = train[train["positive"]] if positive_only else train
    last_seen = df.groupby("item_idx")["timestamp"].max().sort_values(ascending=False)
    return last_seen.index.to_numpy()


def recommend_from_global_ranking(
    users: list[int],
    ranking: np.ndarray,
    seen: dict[int, set[int]],
    k: int,
) -> dict[int, list[int]]:
    """Give every user the same global `ranking`, minus items they've already seen.

    Used by both Popularity and Recency. Scans the ranking in order and keeps the
    first K items the user hasn't already interacted with.
    """
    recs: dict[int, list[int]] = {}
    ranking_list = ranking.tolist()
    for u in users:
        s = seen.get(u, set())
        out: list[int] = []
        for it in ranking_list:
            if it not in s:
                out.append(it)
                if len(out) == k:
                    break
        recs[u] = out
    return recs


def recommend_random(
    users: list[int],
    n_items: int,
    seen: dict[int, set[int]],
    k: int,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Recommend K random unseen items per user (reproducible via `seed`)."""
    rng = np.random.default_rng(seed)
    recs: dict[int, list[int]] = {}
    for u in users:
        s = seen.get(u, set())
        # sample a few extra so we still have K after dropping seen items
        cand = rng.choice(n_items, size=k + len(s), replace=False)
        recs[u] = [int(it) for it in cand if it not in s][:k]
    return recs
