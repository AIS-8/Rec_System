# ElectroRec: A Real-Time Recommendation System

ElectroRec is a recommender built on the Amazon Reviews 2023 Electronics dataset. It goes all the way from raw review data to a running online store, where different parts of the page are powered by different models, just like a real e-commerce site. Under the hood it combines a retrieval stage that finds candidate products and a ranking stage that orders them, and it serves everything live through an API.



## What it does

The store presents six different recommendation models, and every product card is tagged with the model that produced it so you can always tell what is running.

- **Popular right now** shows trending products to logged out visitors.
- **Recommended for you** runs the full pipeline for a signed in user. The two-tower model retrieves candidates, FAISS finds the closest ones, and LightGBM re-ranks them.
- **Because you liked** shows products close to something the user liked, using cosine similarity on the learned item embeddings.
- **Similar items** does the same on a product page.
- **Up next for you** uses a sequential model (GRU4Rec) that reads the user's history in time order and predicts what comes next.
- **Search** matches a text query against product titles and then re-orders the results for the signed in user.

The feed also responds to what you do during a session. When you search for something, like a product, or add it to the cart, the recommendations shift toward that interest in real time.

## Results

Every model was evaluated on a held-out, time-based test period, where we train on the past and test on the future. Retrieval returns the top 100 unseen candidates per user.

| Method | Recall@20 | Recall@50 | NDCG@10 | Coverage@20 |
|---|---|---|---|---|
| Random | 0.0012 | 0.0020 | 0.0004 | 0.9987 |
| Popularity | 0.0115 | 0.0250 | 0.0034 | 0.0012 |
| Recency | 0.0052 | 0.0138 | 0.0016 | 0.0011 |
| MF-BPR | 0.0207 | 0.0256 | 0.0129 | 0.6847 |
| Two-tower | 0.0283 | 0.0369 | 0.0136 | 0.9910 |
| Two-tower + LightGBM | 0.0243 | 0.0396 | 0.0128 | 0.9501 |
| GRU4Rec | 0.0312 | 0.0393 | 0.0177 | 0.7876 |

GRU4Rec gives the best accuracy because it uses the order of a user's history. The two-tower gives the widest catalog coverage. The full write up, including the ablation study, the cold start analysis, feature importance, and the latency breakdown, is in [ANALYSIS.md](ANALYSIS.md).

## Average response time

The API measures the latency of every call. A personalized recommendation serves in about 36 ms on average once the service is warm. The lighter rows (similar items, because you liked, up next) return in a few milliseconds because they skip the ranking stage. You can watch the live average in the app sidebar or read it from `GET /metrics`.

## Architecture

The project has an offline part and an online part.

Offline, in the notebooks:

- Clean and subsample the data, then build the time-based split.
- Train the models and export the item embeddings, the FAISS index, and the LightGBM model.

Online, in the running app, three services work together:

- **PostgreSQL** stores the item metadata and the user interaction history.
- **FastAPI** loads the models once at startup and serves recommendations over a REST API, with interactive documentation at `/docs`.
- **Streamlit** is the storefront that calls the API and shows the results.

They all start together with Docker Compose. This mirrors a real deployment, where the database, the model serving, and the interface are separate services.

## How to run

### The web app

You need Docker installed. From the project root, run:

```bash
docker compose up --build
```

Then open:

- Store: http://localhost:8501
- API docs: http://localhost:8000/docs

The first start takes a few minutes because it installs the dependencies and seeds the database. Watch the `api` logs and wait for the `ready` message.

### The notebooks

The notebooks run top to bottom, either on Google Colab or in a local Python environment. They read the processed data from `data/` and write the model files to `artifacts/`. Run them in order, from `01` to `07`.

To rebuild the processed data from the raw dataset, run:

```bash
python src/recsys/data.py --out data
```

The raw dataset, the processed parquet files, and the trained models are large, so they stay out of the repository and are shared separately through Google Drive. Place the `data/` and `artifacts/` folders in the project root before starting the app.

## Project structure

```
notebooks/          one notebook per stage (01 to 07), each narrated end to end
src/recsys/         model code and utilities (data, metrics, baselines, models, features, faiss)
webapp/
  api/              FastAPI backend and the recommendation engine
  frontend/         Streamlit storefront
  db/               PostgreSQL setup
ANALYSIS.md         results, tables, and discussion
docker-compose.yml  runs Postgres, the API, and the frontend together
```

## Who did what

| Member | Work |
|---|---|
| Aishwarya | Data preparation and exploratory analysis, LightGBM ranking model, personalized search, GRU4Rec sequential retriever, analysis and documentation |
| Taha | Baseline models, MF-BPR retriever, evaluation metrics |
| Omar | Two-tower retriever |
| Aymane | Web application: FastAPI backend, Streamlit frontend, PostgreSQL, Docker setup |

## Extra features

On top of the core pipeline, the project adds a sequential model (GRU4Rec) that reads history in order, a personalized search built on TF-IDF and LightGBM, a study of uniform versus popularity negative sampling, a fairness look across user activity levels, and live session based personalization inside the store.
