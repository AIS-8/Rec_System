"""Database engine + a small readiness wait so the API survives Postgres starting up."""

import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config


def get_engine() -> Engine:
    return create_engine(config.DATABASE_URL, pool_pre_ping=True, future=True)


def wait_for_db(engine: Engine, retries: int = 30, delay: float = 2.0) -> None:
    """Block until Postgres accepts a connection (compose healthcheck usually
    handles this, but we retry anyway for robustness)."""
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"[db] connected (attempt {attempt})", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[db] not ready ({attempt}/{retries}): {exc}", flush=True)
            time.sleep(delay)
    raise RuntimeError("database never became available")
