"""Runtime configuration, read from environment variables (set by docker-compose)."""

import os
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://recsys:recsys@localhost:5432/recsys"
)
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

# Two-tower architecture — must match how it was trained (notebook 04).
EMB_DIM = 64
HIDDEN = (128,)
OUT_DIM = 64
TEMPERATURE = 0.1

# Serving knobs
N_CANDIDATES = 100     # top-K retrieved per user before ranking
POSITIVE_THRESHOLD = 4.0

# Personalized search: final rank blends text relevance with the personalized
# LightGBM score. Higher = sharper on the query wording; lower = more personalized.
SEARCH_TEXT_WEIGHT = 0.6
