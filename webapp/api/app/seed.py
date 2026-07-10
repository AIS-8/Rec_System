"""
Populate Postgres from the exported Parquet files (the "offline -> online" hand-off).

Idempotent: if the `items` table already has rows we assume the DB is seeded and
return immediately, so restarting the API doesn't reload everything.
"""

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config

ITEM_COLS = [
    "item_idx", "parent_asin", "title", "image_url",
    "main_category", "category", "price", "average_rating", "rating_number",
]

# schema (also created by db/init.sql; repeated here so seeding works even on a
# pre-existing volume without the init script)
_DDL = [
    """CREATE TABLE IF NOT EXISTS items (
        item_idx INTEGER PRIMARY KEY, parent_asin TEXT, title TEXT, image_url TEXT,
        main_category TEXT, category TEXT, price DOUBLE PRECISION,
        average_rating DOUBLE PRECISION, rating_number BIGINT)""",
    """CREATE TABLE IF NOT EXISTS interactions (
        id BIGSERIAL PRIMARY KEY, user_idx INTEGER NOT NULL, item_idx INTEGER NOT NULL,
        rating REAL, ts BIGINT, split TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_inter_user  ON interactions (user_idx)",
    "CREATE INDEX IF NOT EXISTS idx_inter_split ON interactions (split)",
    "CREATE INDEX IF NOT EXISTS idx_inter_item  ON interactions (item_idx)",
]


def _already_seeded(engine: Engine) -> bool:
    with engine.connect() as conn:
        exists = conn.execute(text(
            "SELECT 1 FROM information_schema.tables WHERE table_name='items'"
        )).first()
        if not exists:
            return False
        n = conn.execute(text("SELECT COUNT(*) FROM items")).scalar_one()
        return n > 0


def seed(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in _DDL:
            conn.execute(text(stmt))

    if _already_seeded(engine):
        print("[seed] items table already populated — skipping", flush=True)
        return

    data = config.DATA_DIR
    print("[seed] loading items ...", flush=True)
    items = pd.read_parquet(data / "items.parquet")
    items = items[[c for c in ITEM_COLS if c in items.columns]].copy()
    items.to_sql("items", engine, if_exists="append", index=False,
                 chunksize=5000, method="multi")

    print("[seed] loading interactions (train + test) ...", flush=True)
    frames = []
    for split in ("train", "test"):
        df = pd.read_parquet(data / "splits" / f"{split}.parquet")
        df = df[["user_idx", "item_idx", "rating", "timestamp"]].rename(
            columns={"timestamp": "ts"})
        df["split"] = split
        frames.append(df)
    inter = pd.concat(frames, ignore_index=True)
    inter.to_sql("interactions", engine, if_exists="append", index=False,
                 chunksize=10000, method="multi")

    print(f"[seed] done: {len(items):,} items, {len(inter):,} interactions", flush=True)
