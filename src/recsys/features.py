"""
Feature engineering for the LightGBM ranker.

The retriever returns ~100 plausible candidates per user using only a cosine score.
The ranker re-orders them using a rich flat feature vector per (user, item, score)
row. We engineer 13 features spanning four groups:

  user-level     u_activity, u_avg_rating, u_n_cats, u_days_since_last
  item-level     i_pop, i_avg_rating_train, i_meta_avg_rating, i_rating_number, i_price,
                 i_days_since_last
  cross          cat_affinity, cat_match         (does the user like this item's category?)
  retrieval/     retrieval_score                 (the two-tower cosine — a strong prior)

Trees can't discover user×item crosses on their own, so `cat_affinity`/`cat_match`
are pre-computed here. All aggregate tables are built from the TRAIN split only, so
the same function produces training features and (leak-free) serving features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MS_PER_DAY = 86_400_000

FEATURE_COLS = [
    # user
    "u_activity", "u_avg_rating", "u_n_cats", "u_days_since_last",
    # item
    "i_pop", "i_avg_rating_train", "i_meta_avg_rating", "i_rating_number",
    "i_price", "i_days_since_last",
    # cross
    "cat_affinity", "cat_match",
    # retrieval prior
    "retrieval_score",
]


def compute_user_stats(train: pd.DataFrame, cat_of_item: np.ndarray, split_ts: int) -> pd.DataFrame:
    """Per-user aggregates from the training split."""
    t = train.copy()
    t["cat"] = cat_of_item[t["item_idx"].to_numpy()]
    g = t.groupby("user_idx")
    s = pd.DataFrame({
        "u_activity": g.size(),
        "u_avg_rating": g["rating"].mean(),
        "u_n_cats": g["cat"].nunique(),
        "u_last_ts": g["timestamp"].max(),
    })
    s["u_days_since_last"] = (split_ts - s["u_last_ts"]) / MS_PER_DAY
    return s.drop(columns="u_last_ts")


def compute_item_stats(train: pd.DataFrame, items: pd.DataFrame,
                       n_items: int, split_ts: int) -> pd.DataFrame:
    """Per-item aggregates from train + static metadata, indexed 0..n_items-1."""
    posc = train[train["positive"]].groupby("item_idx").size()
    avg = train.groupby("item_idx")["rating"].mean()
    last = train.groupby("item_idx")["timestamp"].max()

    s = pd.DataFrame(index=pd.RangeIndex(n_items, name="item_idx"))
    s["i_pop"] = posc.reindex(s.index).fillna(0)
    s["i_avg_rating_train"] = avg.reindex(s.index).fillna(0)
    s["i_days_since_last"] = ((split_ts - last.reindex(s.index)) / MS_PER_DAY)
    s["i_days_since_last"] = s["i_days_since_last"].fillna(s["i_days_since_last"].max())

    meta = items.set_index("item_idx")
    s["i_meta_avg_rating"] = meta["average_rating"].reindex(s.index).fillna(0)
    s["i_rating_number"] = meta["rating_number"].reindex(s.index).fillna(0)
    price = meta["price"].reindex(s.index)
    s["i_price"] = price.fillna(price.median())
    return s


def compute_user_cat_counts(train: pd.DataFrame, cat_of_item: np.ndarray) -> pd.Series:
    """Series indexed by (user_idx, cat) → how many train interactions the user had
    in that category. Drives the category-affinity cross feature."""
    t = train[["user_idx", "item_idx"]].copy()
    t["cat"] = cat_of_item[t["item_idx"].to_numpy()]
    return t.groupby(["user_idx", "cat"]).size().rename("cat_affinity")


def build_features(cand: pd.DataFrame, user_stats: pd.DataFrame, item_stats: pd.DataFrame,
                   user_cat_counts: pd.Series, cat_of_item: np.ndarray) -> pd.DataFrame:
    """Assemble the flat feature matrix for candidate rows.

    `cand` must have columns [user_idx, item_idx, retrieval_score].
    Returns `cand` with the FEATURE_COLS added (same row order).
    """
    df = cand.copy()
    df["cat"] = cat_of_item[df["item_idx"].to_numpy()]
    df = df.merge(user_stats, on="user_idx", how="left")
    df = df.merge(item_stats, on="item_idx", how="left")
    df = df.merge(
        user_cat_counts.reset_index(), on=["user_idx", "cat"], how="left"
    )
    df["cat_affinity"] = df["cat_affinity"].fillna(0)
    df["cat_match"] = (df["cat_affinity"] > 0).astype(np.int8)
    for c in FEATURE_COLS:
        if c in df:
            df[c] = df[c].fillna(0)
    return df
