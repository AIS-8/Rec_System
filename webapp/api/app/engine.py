"""
The recommendation engine — loaded once at API startup.

Offline artifacts (produced by the notebooks) + live Postgres data are combined
into an in-memory engine that serves four recommendation flows:

  * popular()            non-personalized, for logged-out visitors
  * recommend_for_user() two-tower retrieval -> LightGBM ranking (logged-in home)
  * similar_items()      cosine nearest neighbours on learned item embeddings
  * because_you_liked()  similar items to one item from the user's recent history

The retrieve->rank pipeline mirrors production: user tower forward pass -> FAISS
top-100 -> LightGBM re-rank -> temperature-sampled top-K (so results vary on reload).
"""

from __future__ import annotations

import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text

from recsys.faiss_index import build_flat_ip, search as faiss_search
from recsys.features import (FEATURE_COLS, build_features, compute_item_stats,
                             compute_user_cat_counts, compute_user_stats)
from recsys.models.two_tower import TwoTower
from recsys.models.gru4rec import GRU4Rec

from . import config


def _softmax_sample(item_ids: np.ndarray, scores: np.ndarray, k: int,
                    temperature: float, rng: random.Random) -> list[int]:
    """Sample k items without replacement, weighted by softmax(scores / T).

    Temperature > 0 injects controlled randomness so the same request returns a
    slightly different (still relevant) list on reload. T -> 0 would be greedy.
    """
    k = min(k, len(item_ids))
    if k == 0:
        return []
    if temperature <= 0:
        return item_ids[np.argsort(-scores)[:k]].tolist()
    s = scores.astype(np.float64)
    s = (s - s.max()) / max(temperature, 1e-6)
    p = np.exp(s)
    p /= p.sum()
    seed = rng.randint(0, 2**31 - 1)
    chosen = np.random.default_rng(seed).choice(len(item_ids), size=k, replace=False, p=p)
    return item_ids[chosen].tolist()


class Engine:
    def __init__(self, sql_engine):
        self.sql = sql_engine
        self.rng = random.Random()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        art, data = config.ARTIFACTS_DIR, config.DATA_DIR
        print("[engine] loading artifacts ...", flush=True)

        self.cat_of_item = np.load(art / "item_categories.npy")
        self.n_items = int(len(self.cat_of_item))
        self.n_cats = int(self.cat_of_item.max()) + 1
        self.n_users = int(pd.read_parquet(data / "user_map.parquet").shape[0])

        # item embeddings (L2-normalized) -> FAISS inner-product = cosine
        self.item_emb = np.load(art / "two_tower_item_emb.npy").astype("float32")
        self.item_index = build_flat_ip(self.item_emb)

        # two-tower model — used for the live user-tower forward pass
        self.model = TwoTower(self.n_users, self.n_items, self.n_cats,
                              emb=config.EMB_DIM, hidden=config.HIDDEN,
                              out_dim=config.OUT_DIM, temperature=config.TEMPERATURE)
        self.model.load_state_dict(torch.load(art / "two_tower.pt", map_location="cpu"))
        self.model.set_item_categories(self.cat_of_item)
        self.model.eval()

        # LightGBM ranker
        self.ranker = lgb.Booster(model_file=str(art / "lgbm_ranker.txt"))

        # GRU4Rec sequential retriever (bonus) — optional; skip gracefully if absent
        self.gru = None
        try:
            self.gru = GRU4Rec(self.n_items, emb=config.EMB_DIM, hidden=config.GRU_HIDDEN,
                               max_len=config.GRU_MAX_LEN, temperature=config.TEMPERATURE)
            self.gru.load_state_dict(torch.load(art / "gru4rec.pt", map_location="cpu"))
            self.gru.eval()
            self.gru_item_emb = np.load(art / "gru4rec_item_emb.npy").astype("float32")
            self.gru_index = build_flat_ip(self.gru_item_emb)
            print("[engine] GRU4Rec sequential model loaded", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] GRU4Rec not available ({exc}) — 'Up next' disabled", flush=True)

        print("[engine] loading data from Postgres ...", flush=True)
        self.items_df = pd.read_sql("SELECT * FROM items", self.sql).set_index("item_idx")
        inter = pd.read_sql(
            "SELECT user_idx, item_idx, rating, ts, split FROM interactions", self.sql)

        train = inter[inter.split == "train"].rename(columns={"ts": "timestamp"}).copy()
        train["positive"] = train["rating"] >= config.POSITIVE_THRESHOLD
        test = inter[inter.split == "test"].copy()
        split_ts = int(train["timestamp"].max())

        print("[engine] computing ranking feature tables ...", flush=True)
        items_for_stats = self.items_df.reset_index()[
            ["item_idx", "average_rating", "rating_number", "price"]]
        self.user_stats = compute_user_stats(train, self.cat_of_item, split_ts)
        self.item_stats = compute_item_stats(train, items_for_stats, self.n_items, split_ts)
        self.user_cat = compute_user_cat_counts(train, self.cat_of_item)

        # per-user caches
        self.seen = {int(u): set(map(int, g)) for u, g in train.groupby("user_idx")["item_idx"]}
        recent = (test.sort_values("ts", ascending=False)
                      .groupby("user_idx")["item_idx"].apply(lambda s: list(map(int, s))))
        self.recent = {int(u): v for u, v in recent.items()}

        # time-ordered training sequence per user (input to GRU4Rec's "Up next")
        tr_sorted = train.sort_values("timestamp")
        self.user_seq = {int(u): list(map(int, g))
                         for u, g in tr_sorted.groupby("user_idx")["item_idx"]}

        pop = train[train.positive]["item_idx"].value_counts()
        self.pop_items = pop.index.to_numpy()
        self.pop_scores = pop.to_numpy().astype(np.float64)

        # TF-IDF over item titles, for personalized text search.
        # Row i == item_idx i (reindexed to match self.item_emb), so a text query
        # retrieves candidates that the LightGBM ranker can then personalize.
        print("[engine] building TF-IDF title index ...", flush=True)
        titles = (self.items_df["title"].reindex(range(self.n_items))
                  .fillna("").astype(str).tolist())
        self.tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                         max_features=50000)
        self.tfidf_mat = self.tfidf_vec.fit_transform(titles)

        print(f"[engine] ready: {self.n_items:,} items, {self.n_users:,} users", flush=True)

    # -------------------------------------------------------------- helpers
    def _decorate(self, item_idxs: list[int], model_label: str) -> list[dict]:
        out = []
        for it in item_idxs:
            if it not in self.items_df.index:
                continue
            row = self.items_df.loc[it]
            out.append({
                "item_idx": int(it),
                "title": _clean(row.get("title")),
                "image_url": _clean(row.get("image_url")),
                "category": _clean(row.get("category")),
                "price": _num(row.get("price")),
                "average_rating": _num(row.get("average_rating")),
                "model": model_label,
            })
        return out

    def get_item(self, item_idx: int) -> dict | None:
        d = self._decorate([item_idx], "item")
        return d[0] if d else None

    def list_demo_users(self, limit: int = 60) -> list[dict]:
        """Users that have both training history and recent (test) activity — good
        candidates for the login dropdown and the 'because you liked' row."""
        users = []
        for u, rec in self.recent.items():
            if u in self.seen and len(rec) >= 1:
                users.append((u, len(self.seen.get(u, ())), len(rec)))
        users.sort(key=lambda t: t[2], reverse=True)
        return [{"user_idx": u, "n_history": nh, "n_recent": nr}
                for u, nh, nr in users[:limit]]

    # ------------------------------------------------------------ flows
    def popular(self, k: int = 10, temperature: float = 0.5) -> list[dict]:
        pool = min(len(self.pop_items), max(k * 4, 50))
        chosen = _softmax_sample(self.pop_items[:pool], self.pop_scores[:pool],
                                 k, temperature, self.rng)
        return self._decorate(chosen, "Popular right now")

    def similar_items(self, item_idx: int, k: int = 10) -> list[dict]:
        """Item-to-item similarity: cosine nearest neighbours on the learned
        two-tower item embeddings (FAISS inner product over L2-normalized vectors)."""
        if item_idx < 0 or item_idx >= self.n_items:
            return []
        vec = self.item_emb[item_idx: item_idx + 1]
        _, ids = faiss_search(self.item_index, vec, k + 1)
        neighbours = [int(i) for i in ids[0] if int(i) != item_idx][:k]
        return self._decorate(neighbours, "Cosine similarity")

    def recommend_for_user(self, user_idx: int, k: int = 10, temperature: float = 0.4,
                           recent: list[int] | None = None) -> list[dict]:
        # cold / unknown user -> popularity fallback
        if user_idx not in self.seen and user_idx not in self.recent:
            return self.popular(k, temperature)

        valid_recent = [int(r) for r in (recent or []) if 0 <= int(r) < self.n_items]
        exclude = set(self.seen.get(user_idx, set())) | set(valid_recent)

        # 1) base personalized feed: user tower -> FAISS -> LightGBM -> temperature sample
        with torch.no_grad():
            uvec = self.model.user_vectors(torch.tensor([user_idx], dtype=torch.long))
        scores, ids = faiss_search(self.item_index, uvec, config.N_CANDIDATES + len(exclude))
        cand = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) not in exclude]
        cand = cand[: config.N_CANDIDATES]
        if not cand:
            return self.popular(k, temperature)
        df = pd.DataFrame(cand, columns=["item_idx", "retrieval_score"])
        df["user_idx"] = user_idx
        feat = build_features(df, self.user_stats, self.item_stats,
                              self.user_cat, self.cat_of_item)
        feat["lgbm"] = self.ranker.predict(feat[FEATURE_COLS])
        top = feat.sort_values("lgbm", ascending=False).head(max(k * 3, 30))
        base = _softmax_sample(top["item_idx"].to_numpy(), top["lgbm"].to_numpy(),
                               k, temperature, self.rng)

        if not valid_recent:
            return self._decorate(base, "Two-tower + LightGBM")

        # 2) session slots: items content-similar (TF-IDF titles) to recent activity —
        # reliable and category-true (a viewed backpack surfaces other backpacks).
        m = min(k, max(1, round(k * config.SESSION_FRACTION)))
        sess = self._content_similar(valid_recent, exclude | set(base), m)
        feed = (sess + [b for b in base if b not in sess])[:k]
        out = self._decorate(feed, "Two-tower + LightGBM")
        sset = set(sess)
        for d in out:
            if d["item_idx"] in sset:
                d["model"] = "From your recent activity"
        return out

    def next_for_user(self, user_idx: int, k: int = 10,
                      recent: list[int] | None = None) -> list[dict]:
        """GRU4Rec 'Up next' — feed the user's time-ordered sequence (history + any live
        in-session activity) through the GRU and retrieve the predicted next items."""
        if self.gru is None:
            return []
        seq = list(self.user_seq.get(user_idx, []))
        live = [int(r) for r in (recent or []) if 0 <= int(r) < self.n_items]
        seq = (seq + live)[-config.GRU_MAX_LEN:]
        if not seq:
            return []
        inp = torch.tensor([seq], dtype=torch.long)
        lens = torch.tensor([len(seq)], dtype=torch.long)
        with torch.no_grad():
            rep = self.gru.seq_repr(inp, lens).cpu().numpy().astype("float32")
        exclude = set(self.seen.get(user_idx, set())) | set(live)
        _, ids = faiss_search(self.gru_index, rep, k + len(exclude))
        chosen = [int(i) for i in ids[0] if int(i) not in exclude][:k]
        return self._decorate(chosen, "GRU4Rec (sequential)")

    def _content_similar(self, recent: list[int], exclude: set, m: int) -> list[int]:
        """Items whose titles are most similar (TF-IDF cosine) to the recent activity.

        Recent items are **recency-weighted** (newest = highest) so the user's latest
        search/like dominates rather than being diluted by older activity.
        """
        w = np.arange(1, len(recent) + 1, dtype=float)               # newest (last) heaviest
        w /= w.sum()
        q = np.asarray(self.tfidf_mat[recent].multiply(w[:, None]).sum(axis=0))  # [1, vocab]
        sims = np.asarray(self.tfidf_mat @ q.T).ravel()              # [n_items]
        out = []
        for i in np.argsort(-sims):
            i = int(i)
            if sims[i] <= 0:
                break
            if i in exclude:
                continue
            out.append(i)
            if len(out) >= m:
                break
        return out

    def search(self, query: str, user_idx: int | None = None, k: int = 10) -> dict:
        """Personalized search: TF-IDF title retrieval -> LightGBM re-rank.

        Text handles relevance (candidate generation); for a signed-in user the
        LightGBM ranker reorders those candidates by predicted preference, so the
        same query returns a different order per user. Guests get pure text order.
        """
        q = (query or "").strip()
        if not q:
            return {"items": [], "personalized": False}
        qv = self.tfidf_vec.transform([q])
        scores = np.asarray((self.tfidf_mat @ qv.T).todense()).ravel()
        order = np.argsort(-scores)[: max(k * 10, 100)]
        cand = order[scores[order] > 0]                      # keep actual matches only
        if len(cand) == 0:
            return {"items": [], "personalized": False}

        personalized = user_idx is not None and (
            user_idx in self.seen or user_idx in self.recent)
        if not personalized:
            return {"items": self._decorate([int(i) for i in cand[:k]], "Search (TF-IDF)"),
                    "personalized": False}

        # personalized re-rank: retrieval_score = two-tower user<->item cosine
        uvec = self.model.user_vectors(torch.tensor([user_idx], dtype=torch.long))[0]
        rscore = self.item_emb[cand] @ uvec
        df = pd.DataFrame({"item_idx": cand.astype(int), "retrieval_score": rscore})
        df["user_idx"] = user_idx
        feat = build_features(df, self.user_stats, self.item_stats,
                              self.user_cat, self.cat_of_item)
        feat["lgbm"] = self.ranker.predict(feat[FEATURE_COLS])

        # final rank = blend of text relevance + personalized score (both min-max
        # normalized over the candidate set) so a strong title match isn't buried
        # by personalization, while the user's preference still reorders ties.
        feat["text"] = feat["item_idx"].map(
            dict(zip(cand.astype(int).tolist(), scores[cand]))).astype(float)
        beta = config.SEARCH_TEXT_WEIGHT
        feat["blend"] = beta * _minmax(feat["text"]) + (1 - beta) * _minmax(feat["lgbm"])
        top = feat.sort_values("blend", ascending=False).head(k)
        return {"items": self._decorate(top["item_idx"].astype(int).tolist(),
                                        "Personalized search"),
                "personalized": True}

    def because_you_liked(self, user_idx: int, k: int = 10,
                          recent: list[int] | None = None) -> dict | None:
        valid_recent = [int(r) for r in (recent or []) if 0 <= int(r) < self.n_items]
        if valid_recent:
            seed_item = valid_recent[-1]                 # the most recent in-session action
        else:
            rec = self.recent.get(user_idx) or list(self.seen.get(user_idx, []))
            if not rec:
                return None
            seed_item = self.rng.choice(rec)             # else vary the anchor on reload
        seed = self.get_item(seed_item)
        if seed is None:
            return None
        return {"seed": seed, "items": self.similar_items(seed_item, k)}


def _minmax(x) -> np.ndarray:
    """Scale a column to [0, 1]; all-equal columns map to 0 (no signal)."""
    a = np.asarray(x, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def _num(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
