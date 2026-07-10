"""Pydantic response models — these drive the Swagger docs at /docs."""

from pydantic import BaseModel


class ItemCard(BaseModel):
    item_idx: int
    title: str | None = None
    image_url: str | None = None
    category: str | None = None
    price: float | None = None
    average_rating: float | None = None
    model: str | None = None


class RecResponse(BaseModel):
    section: str
    model: str
    items: list[ItemCard]
    latency_ms: float


class BecauseResponse(BaseModel):
    section: str
    model: str
    seed: ItemCard
    items: list[ItemCard]
    latency_ms: float


class DemoUser(BaseModel):
    user_idx: int
    n_history: int
    n_recent: int


class Metrics(BaseModel):
    requests: int
    avg_latency_ms: float
    p95_latency_ms: float
