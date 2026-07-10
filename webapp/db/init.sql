-- Schema for the recommendation webapp.
-- Runs once when the Postgres data volume is first created. The API's seeder
-- (webapp/api/app/seed.py) fills these tables from the exported Parquet files.

CREATE TABLE IF NOT EXISTS items (
    item_idx        INTEGER PRIMARY KEY,
    parent_asin     TEXT,
    title           TEXT,
    image_url       TEXT,
    main_category   TEXT,
    category        TEXT,
    price           DOUBLE PRECISION,
    average_rating  DOUBLE PRECISION,
    rating_number   BIGINT
);

-- One row per user-item interaction. `split` is 'train' (what the models learned
-- from) or 'test' (the recent activity used as live history in the webapp).
CREATE TABLE IF NOT EXISTS interactions (
    id        BIGSERIAL PRIMARY KEY,
    user_idx  INTEGER NOT NULL,
    item_idx  INTEGER NOT NULL,
    rating    REAL,
    ts        BIGINT,
    split     TEXT
);

CREATE INDEX IF NOT EXISTS idx_inter_user  ON interactions (user_idx);
CREATE INDEX IF NOT EXISTS idx_inter_split ON interactions (split);
CREATE INDEX IF NOT EXISTS idx_inter_item  ON interactions (item_idx);
