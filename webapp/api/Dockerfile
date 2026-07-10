# FastAPI backend. Build context is the REPO ROOT (see docker-compose.yml) so we
# can copy src/recsys, the trained artifacts, and the small Parquet data files.
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime needed by LightGBM and faiss-cpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (small) + the rest of the deps
COPY webapp/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# shared model code (imported by the engine)
COPY src/recsys /app/recsys

# trained artifacts (two-tower, LightGBM, item embeddings, categories)
COPY artifacts /app/artifacts

# small data files needed to seed Postgres + size the model
COPY data/items.parquet        /app/data/items.parquet
COPY data/user_map.parquet     /app/data/user_map.parquet
COPY data/item_map.parquet     /app/data/item_map.parquet
COPY data/splits/train.parquet /app/data/splits/train.parquet
COPY data/splits/test.parquet  /app/data/splits/test.parquet

# the API itself
COPY webapp/api/app /app/app

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
