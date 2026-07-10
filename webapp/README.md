# ElectroRec — real-time recommendation webapp

A three-service app that serves the trained Amazon-Electronics recommender in
context, exactly like a real product: different pages use different models.

```
┌──────────────┐      ┌───────────────────────────┐      ┌──────────────────┐
│  Streamlit   │  →   │  FastAPI                   │  →   │  PostgreSQL      │
│  storefront  │ HTTP │  two-tower + LightGBM +    │ SQL  │  items +         │
│  :8501       │      │  FAISS, loaded at startup  │      │  interactions    │
└──────────────┘      │  :8000  (/docs = Swagger)  │      │  :5432           │
                      └───────────────────────────┘      └──────────────────┘
```

## Run it

From the **repository root** (one level up from this folder):

```bash
docker compose up --build
```

Then open:

- **Store:** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs

First start takes a few minutes (installs PyTorch/FAISS/LightGBM and seeds the
database). On startup the API waits for Postgres, seeds it from the exported
Parquet files if empty, then loads the models — watch the `api` logs for
`[api] ready`.

Stop with `Ctrl-C`; `docker compose down -v` also removes the database volume.

## The sections (each powered by a different model)

| Where | Model | Endpoint |
|---|---|---|
| Homepage — **guest** | Popularity | `GET /popular` |
| Homepage — **logged in** | Two-tower → LightGBM | `GET /recommend/{user_idx}` |
| **"Because you liked X"** row | Cosine on learned embeddings | `GET /because-you-liked/{user_idx}` |
| Item page — **"Similar items"** | Cosine on learned embeddings | `GET /similar/{item_idx}` |
| **Search bar** (top of page) | TF-IDF retrieval → LightGBM re-rank | `GET /search?q=&user_idx=` |

"Sign in" via the main-page dropdown (a list of demo users that have history);
choose *Guest* to see the logged-out experience. Every card shows **which model
produced it**, and recommendations **resample on each reload** (temperature
sampling over the top candidates).

### Personalized search

The **search bar** at the top runs a two-step flow: item **titles** are indexed
with **TF-IDF**, so a text query retrieves the relevant candidates; then, for a
signed-in user, the final rank **blends the TF-IDF text-relevance score with a
LightGBM re-rank** on the user's features (the two-tower preference score +
category affinity + popularity). The text weight (`SEARCH_TEXT_WEIGHT`, default
0.6) keeps strong title matches near the top while personalization reorders ties.
The result: the *same query returns a different order per user* — personalized
search. A guest gets pure text relevance; sign in and search the same term to see
it re-ranked for that user. The TF-IDF index is built at API startup from the item
titles (no extra artifact needed).

## Offline → online architecture

- **Offline (notebooks):** train the models, export `two_tower.pt`, item
  embeddings (`two_tower_item_emb.npy`), `item_categories.npy`, and the LightGBM
  ranker (`lgbm_ranker.txt`) into `../artifacts/`. Postgres is seeded with item
  metadata and interaction history.
- **Online (per request):** the API runs the user tower forward pass → FAISS
  top-100 retrieval → build 13 ranking features → LightGBM re-rank → temperature
  sample → return top-K with metadata joined from Postgres.
- The time-based split is respected: models were trained on the **train** split;
  the **test**-period interactions are stored as each user's *recent history* in
  the app (activity the model wasn't trained on), used for the "because you
  liked" row and to exclude already-seen items.

**Note on the user embedding:** our two-tower user tower is ID-based (it takes no
request-time features), so its forward pass is a deterministic lookup+MLP. Item
embeddings are pre-computed offline (as in production); the user embedding is
computed live at request time in the API.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | readiness probe |
| GET | `/metrics` | rolling average + p95 response time |
| GET | `/users?limit=` | demo users for the login dropdown |
| GET | `/items/{item_idx}` | item metadata |
| GET | `/popular?k=&temperature=` | popularity recommendations |
| GET | `/recommend/{user_idx}?k=&temperature=` | personalized (retrieve → rank) |
| GET | `/search?q=&user_idx=&k=` | personalized search (TF-IDF → LightGBM); `user_idx` optional |
| GET | `/similar/{item_idx}?k=` | item-to-item cosine neighbours |
| GET | `/because-you-liked/{user_idx}?k=` | neighbours of an item from recent history |

## Response time

The API records every recommendation call's latency; `GET /metrics` returns the
rolling **average** and **p95** (also shown in the store's sidebar).

Measured on CPU (Docker, `docker compose up`):

| Endpoint | Model(s) run | Typical latency |
|---|---|---|
| `/similar`, `/because-you-liked` | FAISS cosine only | ~2–3 ms |
| `/popular` | in-memory popularity | ~11 ms |
| `/recommend` | **full pipeline**: user tower → FAISS top-100 → 13 features → LightGBM | **~36 ms** (warm mean; min ~32 ms) |

The full retrieve→rank path stays well under the ~100 ms interactive budget; FAISS
retrieval over ~20k items is sub-millisecond, and the LightGBM re-rank dominates.
