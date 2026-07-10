"""
Streaming subsampler + loaders for the Amazon Reviews 2023 (Electronics) dataset.

The raw files are far too large to load into RAM:
    Electronics.jsonl        ~22.6 GB   (reviews / interactions)
    meta_Electronics.jsonl    ~5.2 GB   (item metadata)

This module streams them line-by-line and produces small Parquet files that
Colab can load in seconds:

    data/interactions.parquet   user_idx, item_idx, rating, timestamp, positive, ...
    data/items.parquet          item metadata for surviving items only
    data/user_map.parquet       user_id  -> user_idx
    data/item_map.parquet       parent_asin -> item_idx

Design decisions (documented for the notebook / spec):
  * Subsample = the MOST RECENT 5,000,000 raw interactions (spec-mandated).
    Found via a two-pass streaming scan so we never sort 22 GB in RAM.
  * positive interaction = rating >= 4.0  (explicit-feedback convention).
  * 5-core filtering on the POSITIVE-interaction bipartite graph, applied
    iteratively, so every modelled user and item has >= 5 positives.
    Non-positive interactions of surviving users/items are kept for feature
    engineering (e.g. user/item average rating).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_INTERACTIONS = 5_000_000     # spec: subsample to most recent 5M
POSITIVE_THRESHOLD = 4.0         # rating >= this counts as a positive
K_CORE = 5                       # drop users/items with < K positives

# Fields we keep from each review line
REVIEW_FIELDS = ("user_id", "parent_asin", "rating", "timestamp", "verified_purchase")


# ---------------------------------------------------------------------------
# Pass 1: find the timestamp cutoff for the most-recent N interactions
# ---------------------------------------------------------------------------

def _find_time_cutoff(reviews_path: str, max_n: int, limit: int | None = None) -> tuple[int, int]:
    """Stream once, collect all timestamps, return (cutoff_ts, total_lines).

    Any interaction with timestamp >= cutoff_ts is in the most-recent `max_n`.
    (Ties on the boundary are all kept, so the survivor count is ~max_n.)
    """
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    total = 0
    with open(reviews_path, "rb") as f:
        for line in tqdm(f, desc="pass 1/2 (scan timestamps)", unit=" lines"):
            if not line.strip():
                continue
            ts = orjson.loads(line).get("timestamp")
            if ts is None:
                continue
            buf.append(ts)
            total += 1
            if len(buf) >= 2_000_000:
                chunks.append(np.asarray(buf, dtype=np.int64))
                buf.clear()
            if limit and total >= limit:
                break
    if buf:
        chunks.append(np.asarray(buf, dtype=np.int64))

    all_ts = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    if all_ts.size <= max_n:
        cutoff = int(all_ts.min()) if all_ts.size else 0
    else:
        # kth largest timestamp = the (size - max_n)th smallest
        cutoff = int(np.partition(all_ts, all_ts.size - max_n)[all_ts.size - max_n])
    return cutoff, total


# ---------------------------------------------------------------------------
# Pass 2: keep interactions with timestamp >= cutoff
# ---------------------------------------------------------------------------

def _collect_recent(reviews_path: str, cutoff_ts: int, limit: int | None = None) -> pd.DataFrame:
    users, items, ratings, tss, verified = [], [], [], [], []
    seen = 0
    with open(reviews_path, "rb") as f:
        for line in tqdm(f, desc="pass 2/2 (collect recent)", unit=" lines"):
            if not line.strip():
                continue
            seen += 1
            if limit and seen > limit:
                break
            r = orjson.loads(line)
            ts = r.get("timestamp")
            if ts is None or ts < cutoff_ts:
                continue
            uid = r.get("user_id")
            iid = r.get("parent_asin")
            rating = r.get("rating")
            if uid is None or iid is None or rating is None:
                continue
            users.append(uid)
            items.append(iid)
            ratings.append(float(rating))
            tss.append(int(ts))
            verified.append(bool(r.get("verified_purchase", False)))

    return pd.DataFrame(
        {
            "user_id": users,
            "parent_asin": items,
            "rating": np.asarray(ratings, dtype=np.float32),
            "timestamp": np.asarray(tss, dtype=np.int64),
            "verified_purchase": verified,
        }
    )


# ---------------------------------------------------------------------------
# Iterative k-core on the positive-interaction graph
# ---------------------------------------------------------------------------

def _iterative_kcore(df: pd.DataFrame, k: int) -> tuple[set, set]:
    """Return (valid_users, valid_items): every user/item has >= k positives,
    computed by iterating until the graph is stable."""
    pos = df[df["rating"] >= POSITIVE_THRESHOLD][["user_id", "parent_asin"]].drop_duplicates()
    while True:
        uc = pos["user_id"].value_counts()
        ic = pos["parent_asin"].value_counts()
        good_u = uc[uc >= k].index
        good_i = ic[ic >= k].index
        before = len(pos)
        pos = pos[pos["user_id"].isin(good_u) & pos["parent_asin"].isin(good_i)]
        if len(pos) == before:
            break
    return set(pos["user_id"].unique()), set(pos["parent_asin"].unique())


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def _first_image(images) -> str | None:
    if not images:
        return None
    for key in ("hi_res", "large", "thumb"):
        for img in images:
            if img.get(key):
                return img[key]
    return None


def _filter_metadata(meta_path: str, keep_items: set, item_map: dict) -> pd.DataFrame:
    rows = []
    with open(meta_path, "rb") as f:
        for line in tqdm(f, desc="filter metadata", unit=" lines"):
            if not line.strip():
                continue
            m = orjson.loads(line)
            pa_ = m.get("parent_asin")
            if pa_ not in keep_items:
                continue
            cats = m.get("categories") or []
            rows.append(
                {
                    "item_idx": item_map[pa_],
                    "parent_asin": pa_,
                    "title": (m.get("title") or "").strip(),
                    "image_url": _first_image(m.get("images")),
                    "main_category": m.get("main_category"),
                    "category": cats[-1] if cats else m.get("main_category"),
                    "categories": "|".join(cats) if cats else None,
                    "price": m.get("price"),
                    "average_rating": m.get("average_rating"),
                    "rating_number": m.get("rating_number"),
                    "store": m.get("store"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(
    reviews_path: str,
    meta_path: str,
    out_dir: str,
    max_n: int = MAX_INTERACTIONS,
    k: int = K_CORE,
    limit: int | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1] Scanning for most-recent {max_n:,} interactions ...")
    cutoff, total = _find_time_cutoff(reviews_path, max_n, limit=limit)
    print(f"    total interactions seen : {total:,}")
    print(f"    timestamp cutoff        : {cutoff}")

    print("[2] Collecting recent interactions ...")
    df = _collect_recent(reviews_path, cutoff, limit=limit)
    print(f"    collected               : {len(df):,}")

    print(f"[3] Iterative {k}-core on positive (rating>={POSITIVE_THRESHOLD}) graph ...")
    valid_u, valid_i = _iterative_kcore(df, k)
    df = df[df["user_id"].isin(valid_u) & df["parent_asin"].isin(valid_i)].copy()
    print(f"    users kept              : {len(valid_u):,}")
    print(f"    items kept              : {len(valid_i):,}")
    print(f"    interactions kept       : {len(df):,}")

    print("[4] Building integer id maps ...")
    users_sorted = sorted(valid_u)
    items_sorted = sorted(valid_i)
    user_map = {u: i for i, u in enumerate(users_sorted)}
    item_map = {it: i for i, it in enumerate(items_sorted)}
    df["user_idx"] = df["user_id"].map(user_map).astype(np.int32)
    df["item_idx"] = df["parent_asin"].map(item_map).astype(np.int32)
    df["positive"] = (df["rating"] >= POSITIVE_THRESHOLD).astype(bool)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("[5] Filtering item metadata ...")
    items_df = _filter_metadata(meta_path, valid_i, item_map)
    missing = len(valid_i) - len(items_df)
    print(f"    items with metadata     : {len(items_df):,}  (missing: {missing:,})")

    print("[6] Writing Parquet outputs ...")
    df[["user_idx", "item_idx", "user_id", "parent_asin", "rating",
        "timestamp", "verified_purchase", "positive"]].to_parquet(
        out / "interactions.parquet", index=False)
    items_df.to_parquet(out / "items.parquet", index=False)
    pd.DataFrame({"user_id": users_sorted, "user_idx": range(len(users_sorted))}).to_parquet(
        out / "user_map.parquet", index=False)
    pd.DataFrame({"parent_asin": items_sorted, "item_idx": range(len(items_sorted))}).to_parquet(
        out / "item_map.parquet", index=False)

    summary = {
        "total_seen": total,
        "n_interactions": len(df),
        "n_users": len(valid_u),
        "n_items": len(valid_i),
        "n_items_with_meta": len(items_df),
        "sparsity": 1 - len(df) / (len(valid_u) * len(valid_i)) if valid_u and valid_i else None,
        "date_min": pd.to_datetime(df["timestamp"].min(), unit="ms").isoformat(),
        "date_max": pd.to_datetime(df["timestamp"].max(), unit="ms").isoformat(),
        "positive_rate": float(df["positive"].mean()),
    }
    print("[7] Done. Summary:")
    for k_, v in summary.items():
        print(f"    {k_:20s}: {v}")
    return summary


def time_based_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MANDATORY time-based (global) split.

    A single wall-clock cutoff separates past from future -- NO random splitting
    (a random split scores 0 on the spec's evaluation criterion). The most recent
    `test_frac` of interactions (by timestamp) become the test period; the most
    recent `val_frac` of what remains becomes validation. This mirrors deployment:
    we train on the past and serve/evaluate on genuinely unseen future activity,
    and the test-period interactions double as each user's "recent history" in the
    webapp.

    Returns (train, val, test), each a copy sorted by time.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    ts = df[time_col].to_numpy()
    test_cut = np.quantile(ts, 1 - test_frac)
    trainval = df[df[time_col] < test_cut]
    test = df[df[time_col] >= test_cut]

    tv_ts = trainval[time_col].to_numpy()
    val_cut = np.quantile(tv_ts, 1 - val_frac)
    train = trainval[trainval[time_col] < val_cut].copy()
    val = trainval[trainval[time_col] >= val_cut].copy()
    return train, val, test.copy()


def _cli():
    p = argparse.ArgumentParser(description="Subsample Amazon Electronics reviews.")
    p.add_argument("--reviews", default="Electronics.jsonl/Electronics.jsonl")
    p.add_argument("--meta", default="meta_Electronics.jsonl")
    p.add_argument("--out", default="data")
    p.add_argument("--max-n", type=int, default=MAX_INTERACTIONS)
    p.add_argument("--k", type=int, default=K_CORE)
    p.add_argument("--limit", type=int, default=None,
                   help="Only read the first N lines (for quick validation).")
    a = p.parse_args()
    build_dataset(a.reviews, a.meta, a.out, max_n=a.max_n, k=a.k, limit=a.limit)


if __name__ == "__main__":
    _cli()
