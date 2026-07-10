"""
FastAPI backend for the recommendation webapp.

Startup: wait for Postgres -> seed it (idempotent) -> load models/FAISS/ranker.
Every response carries a `latency_ms`; /metrics reports the rolling average and p95
(used for the "average response time" figure in the README).

Interactive docs: http://localhost:8000/docs
"""

import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from .db import get_engine, wait_for_db
from .engine import Engine
from .seed import seed

STATE: dict = {}
LATENCIES: deque = deque(maxlen=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sql = get_engine()
    wait_for_db(sql)
    seed(sql)
    engine = Engine(sql)
    engine.load()
    STATE["engine"] = engine
    print("[api] ready", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="RecSys API — Amazon Electronics",
              description="Two-tower retrieval + LightGBM ranking, served in real time.",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _engine() -> Engine:
    eng = STATE.get("engine")
    if eng is None:
        raise HTTPException(503, "engine still loading")
    return eng


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000
    LATENCIES.append(ms)
    return result, round(ms, 2)


@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("engine") else "loading"}


@app.get("/metrics", response_model=schemas.Metrics)
def metrics():
    lat = list(LATENCIES)
    if not lat:
        return schemas.Metrics(requests=0, avg_latency_ms=0.0, p95_latency_ms=0.0)
    return schemas.Metrics(requests=len(lat),
                           avg_latency_ms=round(float(np.mean(lat)), 2),
                           p95_latency_ms=round(float(np.percentile(lat, 95)), 2))


@app.get("/users", response_model=list[schemas.DemoUser])
def users(limit: int = Query(60, ge=1, le=500)):
    return _engine().list_demo_users(limit)


@app.get("/items/{item_idx}", response_model=schemas.ItemCard)
def item(item_idx: int):
    d = _engine().get_item(item_idx)
    if d is None:
        raise HTTPException(404, "item not found")
    return d


@app.get("/popular", response_model=schemas.RecResponse)
def popular(k: int = Query(10, ge=1, le=50), temperature: float = 0.5):
    items, ms = _timed(lambda: _engine().popular(k, temperature))
    return schemas.RecResponse(section="Popular right now (not logged in)",
                               model="Popularity", items=items, latency_ms=ms)


@app.get("/recommend/{user_idx}", response_model=schemas.RecResponse)
def recommend(user_idx: int, k: int = Query(10, ge=1, le=50), temperature: float = 0.4):
    items, ms = _timed(lambda: _engine().recommend_for_user(user_idx, k, temperature))
    model = items[0]["model"] if items else "Two-tower + LightGBM"
    return schemas.RecResponse(section="Recommended for you", model=model,
                               items=items, latency_ms=ms)


@app.get("/search", response_model=schemas.RecResponse)
def search(q: str = Query(..., min_length=1),
           user_idx: int | None = Query(None),
           k: int = Query(10, ge=1, le=50)):
    out, ms = _timed(lambda: _engine().search(q, user_idx, k))
    model = "Personalized search" if out["personalized"] else "Search (TF-IDF)"
    return schemas.RecResponse(section=f"Results for “{q}”", model=model,
                               items=out["items"], latency_ms=ms)


@app.get("/similar/{item_idx}", response_model=schemas.RecResponse)
def similar(item_idx: int, k: int = Query(10, ge=1, le=50)):
    items, ms = _timed(lambda: _engine().similar_items(item_idx, k))
    return schemas.RecResponse(section="Similar items", model="Cosine similarity",
                               items=items, latency_ms=ms)


@app.get("/because-you-liked/{user_idx}", response_model=schemas.BecauseResponse)
def because_you_liked(user_idx: int, k: int = Query(10, ge=1, le=50)):
    out, ms = _timed(lambda: _engine().because_you_liked(user_idx, k))
    if out is None:
        raise HTTPException(404, "no recent history for this user")
    return schemas.BecauseResponse(
        section=f"Because you liked “{(out['seed']['title'] or '')[:60]}”",
        model="Cosine similarity", seed=out["seed"],
        items=out["items"], latency_ms=ms)
