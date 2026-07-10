# ANALYSIS — Amazon Electronics Real-Time Recommender

Results, tables, and discussion for the full **retrieval → ranking** pipeline and the served
web application. All numbers are produced by the notebooks (`01`–`07`). 

---

## 0 · Dataset, models & evaluation setup

**Dataset**

| | |
|---|---|
| Source | Amazon Reviews 2023 — Electronics |
| Raw size | ~44M reviews (22.6 GB) → subsampled to the **most-recent 5M**, then **5-core** filtered |
| Final dataset | **320,184 interactions · 25,546 users · 20,450 items** |
| Sparsity | 99.94% |
| Positive interaction | `rating ≥ 4` (92.2% of interactions) |
| Split | **Time-based** — train 231,332 / val 40,824 / test 48,028 |
| Evaluable test users | 6,962 (avg 2.32 positive test items each) |

A **time-based split** is used throughout — *train on the past, evaluate on the future* (a random
split is disallowed). Item metadata is joined on `parent_asin` (Amazon's variant-group key).

**Models built** (both core retrieval models hand-written from scratch in PyTorch)

| Model | Role | Notebook |
|---|---|---|
| Random · Popularity · Recency | Baselines | `02` |
| **MF-BPR** | Retrieval — matrix factorization, BPR pairwise loss | `03` |
| **Two-tower** | Retrieval — neural towers, in-batch negatives | `04` |
| **LightGBM (LambdaRank)** | Ranking — 13 engineered features | `05` |
| **GRU4Rec** | Retrieval — sequential (GRU over the ordered history) | `07` |
| TF-IDF + LightGBM | Personalized search | `06` |

**Metrics.** *Recall@20 / Recall@50* — did we retrieve the user's future items? · *NDCG@10* — are
they ranked near the top? · *Catalog Coverage@20* — what % of the 20,450 items ever get
recommended (a diversity measure). Absolute values are small **by construction**: ~20k items and
only ~2.3 relevant test items per user make top-K a hard target, which is exactly why beating the
popularity baseline is meaningful.

---

## 1 · Final comparison — all methods, all metrics  

Evaluated on the test period; each retriever returns its top-100 unseen candidates per user.

| Method | Recall@20 | Recall@50 | NDCG@10 | Coverage@20 |
|---|---|---|---|---|
| Random | 0.0012 | 0.0020 | 0.0004 | 0.9987 |
| Popularity | 0.0115 | 0.0250 | 0.0034 | 0.0012 |
| Recency | 0.0052 | 0.0138 | 0.0016 | 0.0011 |
| MF-BPR *(from scratch)* | 0.0207 | 0.0256 | 0.0129 | 0.6847 |
| Two-tower *(from scratch)* | 0.0283 | 0.0369 | 0.0136 | **0.9910** |
| Two-tower + LightGBM | 0.0243 | **0.0396** | 0.0128 | 0.9501 |
| **GRU4Rec *(from scratch, sequential)*** | **0.0312** | 0.0393 | **0.0177** | 0.7876 |

**Reading it**

- Every **learned retriever crushes the non-personalized baselines** — e.g. the two-tower more than
  doubles Popularity's Recall@20 and lifts NDCG@10 ~4×. Personalization is finding individual taste,
  not re-showing the head (Popularity covers just **0.1%** of the catalog).
- **GRU4Rec is the best retriever on accuracy** — Recall@20 **0.0312** and NDCG@10 **0.0177**, beating
  the two-tower by **~10%** and **~30%**. Reading interactions **in time order** lets it capture
  "what comes next" patterns (laptop → sleeve → charger) that the order-agnostic models miss.
- **The two-tower is the best on coverage** (99% vs GRU4Rec's 79%) — a clean **accuracy-vs-diversity
  trade-off**: GRU4Rec ranks the likely next item highest but concentrates on fewer products; the
  two-tower spreads across the long tail. A production stack would use GRU4Rec as the primary
  retriever with a two-tower/popularity blend for diversity.
- MF-BPR is solid but the weakest learned model (bare ID dot-product vs MLP towers / a GRU).

---

## 2 · Ablation — retrieval-only vs retrieval + ranking 

Same top-100 two-tower candidates per test user; only the **ordering** differs (cosine vs LightGBM
re-rank). This isolates the ranker's contribution.

| Stage | NDCG@10 | Recall@20 | Recall@50 | Coverage@20 |
|---|---|---|---|---|
| Two-tower (retrieval only) | **0.0136** | **0.0283** | 0.0369 | 0.9910 |
| Two-tower + LightGBM (ranking) | 0.0128 | 0.0243 | **0.0396** | 0.9501 |

**Finding — the ranker does not improve the top of the list here; it slightly widens mid-list recall.**
This is a genuine, defensible result, not a bug:

1. **The retriever is already strong** — its candidates arrive well-ordered, so re-shuffling the very
   top tends to hurt more than help.
2. **No impression features (offline setting)** — we lack "what was shown / clicked / skipped," which
   are normally a ranker's most predictive signals. The ranker therefore leans almost entirely on the
   retrieval score (see §4).
3. **Training-label circularity** — the ranker is trained on train-history positives that the
   two-tower was itself trained to rank high, so the retrieval score dominates and the extra features
   mostly reshuffle the *middle* of the list (lifting Recall@50 to 0.0396).

On graded, log-based data **with** impression features, a LambdaMART ranker typically adds clear
NDCG@10 gains. The honest result on this offline dataset: the ranker **matches the retriever at the
top and slightly improves mid-list recall.**

---

## 3 · Cold-start analysis — performance by user activity  

Test users bucketed by their **training-history size**; two-tower retrieval metrics per bucket.

| Train activity | Users | Recall@20 | NDCG@10 |
|---|---|---|---|
| 5–6 interactions | 1,038 | **0.0328** | **0.0159** |
| 7–10 | 1,149 | 0.0120 | 0.0046 |
| 11–20 | 1,219 | 0.0077 | 0.0030 |
| 21+ | 938 | 0.0057 | 0.0027 |

**Finding — measured accuracy *decreases* with activity**, the opposite of the naïve "cold users are
harder" intuition. Two reasons:

1. **Metric denominator effect (dominant).** Recall@K and NDCG@K normalize by the number of relevant
   test items. Low-activity users have **few** test positives, so covering a fraction of them in the
   top-K is *easier* (a user with one test item scores Recall = 1.0 if we find it). High-activity users
   have many test positives spread across diverse tastes → harder to cover in a short list.
2. **ID-based embeddings degrade gracefully.** Because the two-tower learns a per-user embedding, even
   a 5-interaction user gets a usable vector — so the true cold-start cliff (brand-new users with
   *zero* history) doesn't appear here; every evaluated user has ≥5 interactions by the 5-core filter.

**Interpretation for fairness.** The system does **not** under-serve low-activity users on this
metric; if anything the gap reflects that active users are a harder target, not a fairness failure. A
production system would still add an onboarding/popularity fallback for genuinely new (zero-history)
users. *(This doubles as the bonus "fairness across user groups by activity level".)*

---

## 4 · Feature importance — LightGBM ranker  

By total gain across the **13** engineered features:

| Feature | Gain share |
|---|---|
| `retrieval_score` (two-tower cosine) | **65.8%** |
| `cat_affinity` (user × item-category history) | **26.1%** |
| `i_pop` (item training popularity) | **7.2%** |
| remaining 10 features combined | < 2% |

**Finding.** The retrieval score alone carries **two-thirds** of the ranking signal — confirming the
retriever already encodes most of the preference. Category affinity and item popularity add the only
meaningful refinements; `price`, `cat_match`, and `u_avg_rating` are never split on. This
concentration explains §2: with the retrieval score dominating, the ranker mostly re-affirms the
retriever. The clear lever to make the ranker earn its keep is **impression/session features**, which
this offline dataset lacks.

---

## 5 · Latency breakdown — real-time serving  

Mean time per component for a single personalized `/recommend` request (CPU, against the loaded
artifacts):

| Component | Mean latency |
|---|---|
| DB fetch (user history — cached in memory at startup) | ~0 ms |
| User-tower forward pass | 0.19 ms |
| FAISS retrieval (top-100 over 20,450 items) | 0.21 ms |
| Feature build (13 features × 100 candidates) | **46.79 ms** |
| LightGBM re-rank | 3.80 ms |
| **Total (retrieve → rank)** | **~51 ms** |

*(End-to-end via the API's `/metrics`, warm, is ~36 ms. The item-similarity, "because you liked", and
GRU4Rec "up next" endpoints are ~2–5 ms since they skip the ranker.)*

**Finding.** The **models are not the bottleneck** — the user tower and FAISS are **sub-millisecond**
and the LightGBM re-rank is ~4 ms. The dominant cost is the **pandas feature-building step (~47 ms)**.
The whole pipeline still comfortably beats the ~100 ms interactive budget, but the obvious
optimization is to replace the per-request pandas joins with vectorized array lookups or a feature
store (see §6).

---

## 6 · Limitations & future work  

**Primary limitation — the ranker does not beat the retriever.** As shown in §2 and §4, the LightGBM
ranker adds no top-of-list gain because (a) the retriever is already strong and (b) we have **no
impression features** in this offline dataset, so the ranker leans almost entirely on the retrieval
score.

**How we'd address it with more time:** log **impressions and session behaviour** (which items were
shown, in what position, clicked vs skipped, dwell time) and train the ranker on **graded relevance**
(purchase > cart > click). These are consistently the most predictive ranking features in production
and would give the ranker independent signal to reorder the *top* of the list — the setting in which
LambdaMART reliably improves NDCG@10.

**Secondary limitations**

- **Feature build is the latency bottleneck** (§5) — move per-request pandas joins to precomputed
  array lookups / a Redis feature store.
- **"Similar items" can mix categories** — the item embeddings are learned from co-interaction only,
  so on sparse data (median 10 interactions/item) a long-tail item's behavioural neighbours can cross
  categories (e.g. a charger next to ink cartridges). Adding a content signal (a title/text encoder or
  a content blend) or a same-category re-rank would tighten them.
- **Small, sparse dataset** (5-core, 320k interactions) — more data would sharpen embeddings and
  likely lift all metrics.

---

## 7 · Bonus — GRU4Rec sequential retriever *(best accuracy)*

MF-BPR and the two-tower treat a user as an unordered *bag* of items. **GRU4Rec** (from scratch,
notebook `07`) instead reads the user's interactions **in time order** through a GRU and predicts the
**next** item, trained with the same in-batch-negative softmax as the two-tower.

| Retriever | Recall@20 | Recall@50 | NDCG@10 | Coverage@20 |
|---|---|---|---|---|
| Two-tower | 0.0283 | 0.0369 | 0.0136 | **0.9910** |
| **GRU4Rec** | **0.0312** | **0.0393** | **0.0177** | 0.7876 |

**Finding.** GRU4Rec is our **best retriever on every accuracy metric** (+10% Recall@20, +30%
NDCG@10) because **order carries signal** — Electronics purchases follow sequences (laptop → sleeve →
charger). It trades catalog coverage (0.79 vs 0.99): by chasing the most-likely next item it
concentrates on fewer, more predictable products. This is the precision-vs-diversity trade-off of §1.
GRU4Rec is **live in the web app** as the **"⏭️ Up next for you"** row.

---

## 8 · Bonus — negative sampling (uniform vs popularity)

Two otherwise-identical MF-BPR models; the **only** difference is the negative sampler (notebook `03`).

| Sampler | Recall@20 | Recall@50 | NDCG@10 | Coverage@20 |
|---|---|---|---|---|
| **Uniform** negatives | **0.0264** | 0.0347 | **0.0157** | 0.4135 |
| **Popularity** negatives | 0.0207 | 0.0256 | 0.0129 | **0.6847** |

**Finding — a clear accuracy ↔ coverage trade-off.** Uniform gives **~+22% accuracy**; popularity
gives **~65% more coverage**. Popularity sampling repeatedly uses popular items *as negatives*,
pushing them down the ranking — spreading recommendations across the long tail (coverage ↑) but
costing recall on a popularity-skewed test set (accuracy ↓). Neither strictly dominates; the choice
depends on whether the product prioritizes precision at the top or long-tail discovery.

---

## 9 · Bonus — personalized search (TF-IDF + LightGBM re-rank)

A text-query retrieval mode (notebook `06`, **live in the app's search bar**). Every item **title** is
indexed with **TF-IDF** (unigrams + bigrams, stop-words removed); a query retrieves the relevant
candidates, then the **LightGBM ranker re-orders them per user** (`retrieval_score` = two-tower cosine
for that user). The final rank blends text relevance with the personalized score
(`SEARCH_TEXT_WEIGHT = 0.6`).


---

## 10 · Bonus — session personalization & fairness

- **Session personalization (live in the app).** A user's recent in-session activity
  (search / like / cart / view) is fed back: ~40% of the "Recommended for you" feed is filled with
  items **content-similar** to that activity, and the GRU4Rec "Up next" row re-reads the updated
  sequence. Result: the feed **reacts in real time** (search a backpack → backpacks surface, tagged
  "From your recent activity"). It changes *serving behaviour*, not the offline metrics.
- **Fairness by activity level** — see §3: performance is reported across user-activity buckets, and
  low-activity users are **not** under-served (the score gap is a metric-denominator effect).

---

## Summary

- **Retrieval works and personalization is real.** Three from-scratch retrievers (MF-BPR, two-tower,
  GRU4Rec) all beat popularity by a wide margin. **Sequential GRU4Rec is best on accuracy**
  (Recall@20 0.0312, NDCG@10 0.0177) because order carries signal; the **two-tower is best on
  coverage** (99%) — a clean accuracy-vs-diversity trade-off.
- **Ranking is honest.** On this offline dataset the LightGBM ranker matches the retriever rather than
  beating it — traced to the missing impression features and the retriever's dominance (65.8% of
  feature gain).
- **Serving is fast.** ~36–51 ms end-to-end, bottlenecked by feature construction, not the models.
- **Bonuses (4):** sequential model (GRU4Rec), negative-sampling trade-off, personalized search,
  session personalization — plus the fairness-by-activity analysis.
