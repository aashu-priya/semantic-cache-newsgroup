# Semantic Search & Cache System — 20 Newsgroups

### End-to-End ML Engineering · Fuzzy Clustering · Hand-Built Semantic Cache · Live FastAPI Service

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Live-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-7C3AED?style=for-the-badge)
![Model](https://img.shields.io/badge/MiniLM--L6--v2-384_dim-F97316?style=for-the-badge)
![Cache](https://img.shields.io/badge/Semantic_Cache-Built_From_Scratch-DC2626?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge)

---

## What This Builds — And Why It's Non-Trivial

Most semantic search systems are wrappers around a vector database. This project goes further: it answers the question of *what happens after the first search*. If someone already asked `"space shuttle orbits"` and a new user asks `"NASA launches into orbit"` — should the system hit ChromaDB again? No. That is the problem this cache solves.

The system was built end-to-end across four parts:

| Part | What Was Built | Key Result |
|:-----|:--------------|:-----------|
| **1 — Embedding** | 15,176 cleaned newsgroup posts → ChromaDB | 384-dim MiniLM-L6-v2, cosine space, persistent index |
| **2 — Fuzzy Clustering** | Membership distributions across K=11 clusters | K selected by silhouette evidence, not convenience |
| **3 — Semantic Cache** | Hand-built from scratch — no Redis, no libraries | τ=0.44 calibrated empirically; 11× lookup speedup |
| **4 — FastAPI Service** | Three endpoints, live UI, single start command | < 5ms cache hits vs ~250ms full vector search |

---

## Quick Start

```bash
git clone https://github.com/aashu-priya/semantic-cache-newsgroup.git
cd semantic-cache-newsgroup

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# One-time path fix after cloning
python3 fix_paths.py

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — type any natural language query and watch the cache learn in real time.

> **Large data files** are hosted on Google Drive (exceed GitHub's 100 MB limit).  
> Download and place in `newsgroups_chromadb/` → [**Google Drive**](https://drive.google.com/drive/folders/1sAk1n_3nR-gfeBL3hxVjHVwfkqC-piUg?usp=sharing)
>
> Files needed: `chroma.sqlite3` (384 MB) · `embeddings_backup.npy` (23 MB) · `fuzzy_memberships.npy` · `kmeans_centroids.npy`

---

## Architecture

```
 User query (natural language)
         │
         ▼
   Embed with all-MiniLM-L6-v2  ──  384-dim unit-norm vector
         │
         ▼
   Compute fuzzy cluster memberships
         │
         ├──▶  Primary bucket   (argmax of memberships)
         └──▶  Secondary bucket (if membership > 0.20)
         │
         ▼
   Cosine similarity scan within bucket(s)
         │
    sim ≥ τ=0.44?
    ┌─────┴─────┐
   YES          NO
    │            │
    ▼            ▼
 Return       ChromaDB vector search
 cached       Store result in bucket
 result       Return fresh result
 (< 5ms)      (~250ms)
```

> The cluster structure from Part 2 is not decoration — it cuts lookup from **O(N)** to **O(N/K)**, an **11× improvement** that compounds as the cache grows.

---

## Part 1 — Embedding & Vector Database

### Deliberate Cleaning Choices

The dataset is noisy. Every removal was a decision with a reason:

| What Was Removed | Why |
|:----------------|:----|
| Email headers (`From:`, `Subject:`, `Lines:`) | Metadata, not content — would cause embeddings to cluster by sender identity rather than topic semantics |
| Quoted reply blocks (`>` lines) | Duplicate content that biases embeddings toward the most-replied-to users, not the actual post |
| Signatures and footers | `"-- John Smith, MIT"` creates false similarity between posts from the same person across unrelated topics |
| Posts under 50 words after cleaning | Too short for stable embeddings — high variance, low signal. **15,176 of ~18,000 posts survived** |

### Why `all-MiniLM-L6-v2`

- **Length fit** — newsgroup posts are 50–400 words after cleaning; MiniLM was trained on similar-length texts and performs strongly in this range
- **Normalised output** — unit-norm vectors make cosine similarity a dot product, the fastest possible operation at cache lookup time
- **Speed** — 15,176 posts on a T4 GPU in ~4 minutes; larger models give marginally better quality at 3–5× the compute cost for a task that does not require SOTA retrieval accuracy
- **Cosine space alignment** — a model trained with a cosine objective produces the most meaningful similarity scores in a cosine cache

### Why ChromaDB

- Persistent SQLite backend survives Colab session resets — 384 MB index without re-embedding
- Native cosine similarity space — no manual distance inversion needed
- Metadata filtering by `category` supports cluster composition analysis in Part 2
- In-process — no separate server, no ports, works on free Colab

---

## Part 2 — Fuzzy Clustering

### Why Hard Labels Are Not Acceptable Here

A post about gun legislation in a Second Amendment context might produce:

```
Cluster 3  — Politics & Guns       : 0.61
Cluster 10 — Religion & Atheism    : 0.22  (moral argument framing)
Cluster 9  — Middle East Politics  : 0.11  (comparative policy reference)
Cluster 4  — Medical & Science     : 0.06  (injury statistics cited)
```

Hard clustering collapses this to `label = 3` and silently discards the rest. **Fuzzy C-Means (FCM, m=2.0)** preserves the full distribution. This is not aesthetics — it directly determines whether the cache searches one bucket or two for boundary queries.

### K = 11 — Selected by Evidence, Not Convenience

Silhouette scores were computed for K=2 through K=15. The score peaks at **K=11**. This was not chosen because there are 20 categories in the dataset — the clustering found 11 real semantic groupings. Several labelled categories are semantically indistinguishable at the embedding level (`talk.politics.guns` and `talk.politics.misc` merge naturally because users in both groups write about the same things).

![K Selection](outputs/plot_k_selection.png)

### Three Analyses That Convince a Sceptical Reader

**What lives in each cluster** — UMAP projection of all 15,176 embeddings:

![UMAP Clusters](outputs/plot_umap_clusters.png)

Tight, separated groupings (Sports, Space & Science, Autos) confirm pure clusters. The overlapping central region is genuine topical ambiguity — users replying across topic boundaries — not a clustering failure.

**Cluster composition** — what newsgroup categories actually appear inside each cluster:

![Cluster Composition](outputs/plot_cluster_composition.png)

The Religion & Atheism cluster contains both `soc.religion.christian` and `alt.atheism` — two groups arguing about the same subject, so their embeddings converge. The 20 labelled categories do not match the real semantic structure of the corpus.

**Where the model is genuinely uncertain** — per-document fuzzy membership entropy:

![Entropy Map](outputs/plot_entropy_map.png)

Bright = topically ambiguous. Key finding: `entropy_mean = 2.368 ≈ entropy_max = 2.398`. Almost every document carries ambiguity — a property of the corpus, not a clustering failure. It also directly explains why a generic τ from a cleaner corpus would fail in Part 3.

### The 11 Clusters

Named by inspecting dominant newsgroup categories after clustering — not assumed from dataset labels:

| # | Cluster Name | Dominant Newsgroups |
|:-:|:------------|:--------------------|
| 0 | Autos & Motorcycles | rec.motorcycles (612), rec.autos (601) |
| 1 | Cryptography & Electronics | sci.crypt (694), sci.electronics (86) |
| 2 | PC & Mac Hardware | comp.sys.ibm.pc.hardware (646), comp.sys.mac.hardware (473) |
| 3 | Politics & Guns | talk.politics.guns (730), talk.politics.misc (549) |
| 4 | Medical & Science | sci.med (696) |
| 5 | For Sale & Electronics | misc.forsale (560), sci.electronics (208) |
| 6 | Space & Science | sci.space (676), sci.electronics (247) |
| 7 | Sports | rec.sport.hockey (771), rec.sport.baseball (681) |
| 8 | Windows & Graphics | comp.windows.x (698), comp.graphics (496) |
| 9 | Middle East Politics | talk.politics.mideast (761) |
| 10 | Religion & Atheism | soc.religion.christian (839), alt.atheism (560) |

---

## Part 3 — Semantic Cache

> **No Redis. No Memcached. No caching library. Every line is in `main.py`.**

### Data Structure

```python
@dataclass
class CacheEntry:
    query_text      : str            # original query string
    embedding       : np.ndarray     # (384,) normalised unit vector
    result          : Any            # stored ChromaDB search result
    memberships     : np.ndarray     # (K,) fuzzy distribution across clusters
    primary_cluster : int            # argmax of memberships
    timestamp       : float          # unix time of storage
    hit_count       : int            # times this entry has been served

cache_buckets: Dict[int, List[CacheEntry]]  # K buckets, one per cluster
```

### How Cluster Structure Makes the Cache Efficient

A flat cache scans every entry on every lookup: **O(N)**. At 10,000 cached queries, that is 10,000 cosine similarity computations per request.

A cluster-bucketed cache routes each query to its primary cluster first: **O(N/K)**. At K=11, that is **11× fewer comparisons** with zero accuracy loss. The fuzzy memberships add a second gain: if a query has secondary cluster membership above 0.20, that bucket is also searched — so `"Mac vs Windows performance"` correctly finds entries in both Cluster 2 (PC & Mac Hardware) and Cluster 8 (Windows & Graphics).

### Calibrating τ — The Threshold Explored, Not Just Set

τ was calibrated empirically by measuring cosine similarity distributions across three query pair types:

![Threshold Exploration](outputs/plot_threshold_exploration.png)

| Query Pair Type | Similarity Range |
|:---------------|:----------------|
| Exact paraphrase pairs | 0.51 – 0.86 |
| Same-topic, different queries | 0.00 – 0.29 |
| Cross-cluster queries | ~0.00 |
| **Gap between distributions** | **0.22** |
| **τ = 0.44** | **Midpoint of gap — data-driven** |

**What each τ value reveals about the system:**

| τ | Behaviour | Diagnostic |
|:-:|:----------|:-----------|
| 0.10 | Hits on almost anything | Topic-area matching, not semantic matching — wrong cached answers for different questions |
| 0.30 | Hits on same-topic queries | Cannot distinguish `"what causes back pain"` from `"how to treat back pain"` |
| **0.44** | **Hits on paraphrases only** | **Clean separation between same-topic and same-meaning** |
| 0.85 | Almost never hits | Paraphrase similarity peaks at ~0.80 on this corpus — a generic τ from a cleaner corpus makes the cache useless |
| 1.00 | Exact string matches only | Not semantic caching — string matching |

The entropy finding from Part 2 (`entropy_mean ≈ entropy_max`) directly explains why 0.85 fails here. MiniLM on 20 Newsgroups produces more diffuse embeddings than standard benchmarks. A threshold copied from a paper would miss every valid paraphrase hit on this corpus.

---

## Part 4 — FastAPI Service

### `POST /query`

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space shuttle launch into orbit"}'
```

```json
{
  "query": "NASA space shuttle launch into orbit",
  "cache_hit": true,
  "matched_query": "space shuttle mission launched by NASA",
  "similarity_score": 0.8026,
  "result": { "results": [...], "compute_time": 0.0 },
  "dominant_cluster": 6,
  "dominant_cluster_name": "Space & Science"
}
```

On a **MISS**: ChromaDB is queried, the result is stored in the appropriate cluster bucket, then returned. `cache_hit: false`, `similarity_score: null`, `matched_query: null`.

### `GET /cache/stats`

```json
{
  "total_entries": 9,
  "hit_count": 8,
  "miss_count": 9,
  "hit_rate": 0.4706
}
```

### `DELETE /cache`

Flushes all 11 cluster buckets. Resets all counters to zero. ChromaDB index is unaffected — only the in-memory cache is cleared.

### `GET /health`

Returns server status, embedding model name, τ, K, compute device, ChromaDB collection count, and all cluster names.

---

## Performance

| Operation | Latency | Notes |
|:----------|:-------:|:------|
| Cache HIT | **< 5ms** | Pure in-memory similarity — no ChromaDB call |
| Cache MISS | **~250ms** | Full vector search across 15,176 documents |
| Speedup on HIT | **~50×** | |
| Lookup complexity | **O(N/K) = O(N/11)** | 11× faster than a flat cache scan |

---

## Project Structure

```
semantic-cache-newsgroup/
│
├── main.py                            # FastAPI app + SemanticCache — written entirely from scratch
├── requirements.txt                   # Pinned versions with rationale in comments
├── fix_paths.py                       # One-command path fix after cloning
│
├── static/
│   └── index.html                     # Live frontend — single file, zero build step
│
├── newsgroups_chromadb/
│   ├── manifest.json                  # Config — model name, τ, K, file paths
│   └── cluster_metadata.json         # Cluster names, sizes, assignments
│
├── outputs/                           # All generated plots with explanations
│   ├── plot_k_selection.png           # Silhouette scores K=2..15
│   ├── plot_umap_clusters.png         # 2D embedding projection
│   ├── plot_entropy_map.png           # Per-document uncertainty
│   ├── plot_cluster_composition.png   # Category distributions per cluster
│   └── plot_threshold_exploration.png # τ calibration analysis
│
├── assets/                            # Screenshots for all test cases
├── TEST_CASES.md                      # Full test suite with screenshots and explanations
│
├── Part_1—_Embedding_+_ChromaDB_vector_store_.ipynb
├── Part_2_—_Fuzzy_clustering.ipynb
└── Part_3_—_FastAPI_semantic_cache_.ipynb
```

---

## Dependency Pinning — Why These Exact Versions

```
# numpy < 2.0
# chromadb 0.4.24 internally uses np.float_ which was removed in numpy 2.0.
# Upgrading chromadb breaks the opentelemetry dependency chain — not a warning, a crash.
numpy<2.0
chromadb==0.4.24

# OpenTelemetry versions must match chromadb's internal requirements exactly.
# Any mismatch produces an AttributeError on import.
opentelemetry-api==1.38.0
opentelemetry-sdk==1.38.0
opentelemetry-exporter-otlp-proto-http==1.38.0
opentelemetry-exporter-otlp-proto-common==1.38.0
opentelemetry-proto==1.38.0
```

> Understanding *why* versions are pinned — not just that they are — is the difference between a reproducible system and one that breaks on a fresh install.

---

## Large Data Files

| File | Size | Description |
|:-----|:----:|:------------|
| `chroma.sqlite3` | 384 MB | Persistent ChromaDB vector index (15,176 docs × 384 dim) |
| `embeddings_backup.npy` | 23 MB | Raw embedding matrix — (15176, 384) float32 |
| `fuzzy_memberships.npy` | 1.3 MB | FCM output — (15176, 11) soft cluster assignments |
| `kmeans_centroids.npy` | 0.02 MB | Cluster centroids — (11, 384) |

[**Download from Google Drive →**](https://drive.google.com/drive/folders/1sAk1n_3nR-gfeBL3hxVjHVwfkqC-piUg?usp=sharing)

---

## Known Limits and What I Would Do Next

These are engineering observations, not disclaimers:

- **Persist cache to SQLite** — the in-memory cache resets on server restart; a SQLite-backed cache would make the service genuinely stateful across sessions
- **TTL per cache entry** — a newsgroup search result from 2003 and 2024 should not share a cache entry; TTL expiry would fix this
- **Batch embedding at query time** — the service embeds one query per request; under concurrent load, batching would improve throughput significantly
- **HNSW index per cluster bucket** — at very large cache sizes (100k+ entries), even O(N/K) linear scan becomes slow; an approximate nearest neighbour index per bucket would reduce lookup to O(log N/K)
- **Docker** — containerising would eliminate the path-fix step entirely and make deployment reproducible across environments

---

## Technical Decisions Worth Noting

These are the choices that are not visible in the code but shaped every part of the system:

- **Fuzzy membership threshold for secondary bucket search set at 0.20, not lower or higher**

  When a query's secondary cluster membership exceeds 0.20, the cache searches that bucket too. At m=2.0 fuzziness, a document split evenly across two clusters shows ~0.50/0.50. A threshold of 0.20 captures genuine topical ambiguity without triggering on noise-level membership scores, which typically fall below 0.05. Setting it too low means every query searches multiple buckets and the O(N/K) lookup advantage disappears entirely. Setting it too high means legitimate boundary queries — a post about encryption law sitting between Cryptography and Politics — miss results in their secondary bucket and return a false miss.

- **Fuzziness parameter m=2.0 chosen after evaluating the extremes**

  FCM with m=1.0 degenerates toward hard clustering — documents get near-binary assignments and the membership distribution loses meaning. m=3.0 or higher produces distributions so flat that primary cluster assignment becomes unreliable: every document spreads almost uniformly across all K buckets, which breaks the bucket routing logic entirely. m=2.0 is the standard starting point, and the entropy analysis confirmed it produces the right sharpness for this corpus — most documents have a clear primary cluster, while genuinely ambiguous ones still spread meaningfully across two or three rather than diffusing across all eleven.

- **Cosine similarity chosen over Euclidean distance for all cache lookup operations**

  Unit-norm embeddings make cosine similarity mathematically equivalent to a dot product — a single vectorised matrix multiplication. Euclidean distance on the same 384-dimensional vectors requires computing squared differences across every dimension per pair, which is slower and produces scores that are harder to interpret directly. At cache lookup time, where latency is the primary concern, this is not a minor optimisation. It also means similarity scores map cleanly onto the τ threshold without any distance-to-similarity inversion step.

- **ChromaDB run in-process rather than as a separate server**

  Running ChromaDB in-process eliminates a network hop on every cache miss — the vector search happens in the same Python process as the FastAPI service with no socket overhead. The tradeoff is that the index cannot be shared across multiple service instances. For a single-service deployment on a local machine or single cloud instance, in-process is the correct choice. A multi-replica deployment would require a separate ChromaDB server, which is the obvious next architectural step if the service needed horizontal scaling.

- **Frontend built with vanilla HTML, CSS, and JavaScript — no framework, no build step**

  The UI is a single `index.html` file served as a static asset. No npm, no bundler, no separate build pipeline. This was a deliberate decision to keep the deployment surface at the minimum: `uvicorn main:app` is genuinely the only command needed to run the entire system. A React or Vue frontend would add a parallel build pipeline with its own dependency tree that has nothing to do with the ML system being evaluated.

- **FastAPI state loaded once at startup via the `lifespan` context manager**

  The cache object, ChromaDB client, embedding model, and cluster centroids are all initialised once when the server starts via FastAPI's `lifespan` hook, rather than as module-level globals or lazy-initialised on the first request. This means the first query is not slow while the model loads. It also means all application state is cleanly scoped to the server lifecycle — no global mutation, no race condition if two requests arrive before initialisation completes, and a clean shutdown path when the server stops.

---

## What I Learned Building This

- **The corpus tells you what the threshold should be — papers don't**

  Every benchmark I found for MiniLM similarity thresholds suggested τ in the range of 0.75–0.90 for paraphrase detection. On 20 Newsgroups, paraphrase similarity peaks at ~0.80 and the clean separation gap sits at 0.44. If I had copied a threshold from a paper without measuring, the cache would have missed every valid paraphrase hit on this specific corpus. The entropy analysis — discovering that `entropy_mean ≈ entropy_max` — is what prompted me to investigate the similarity distributions rather than trust published values. The right threshold is a property of the data, not the model, and cannot be looked up anywhere.

- **Fuzzy clustering is not just theoretically nicer — it changes what the system can actually do**

  Going into Part 2, I expected the fuzzy output to be an interesting analysis artifact that fed into a mostly hard routing decision downstream. The secondary bucket search turned out to matter concretely: queries about encryption policy miss cached results about crypto law unless both the Cryptography and Politics clusters are searched. Hard clustering would silently return a miss for those queries every single time with no indication that a relevant cached result existed. The membership distribution is doing real operational work, not just producing a more nuanced visualisation.

- **Dependency management is an engineering problem, not a packaging chore**

  Debugging the chromadb + numpy + opentelemetry version conflict took longer than any individual feature in the project. The failure mode was an `AttributeError` on import — no warning, no clear message pointing at the conflict, just a crash that required bisecting package versions to diagnose. After going through that once, pinning every version with a comment explaining exactly why it is pinned stopped feeling like defensive overhead and started feeling like the minimum documentation any system needs to be reproducible by another person on a different machine.

- **O(N/K) matters more as the cache grows, not at the start**

  At 10 cached entries, a flat scan and a cluster-bucketed scan are both effectively instant — the difference is invisible. The architectural choice only becomes measurable at scale, when the cache has accumulated hundreds or thousands of entries. Designing the bucketed structure from the beginning meant the cache would stay fast without a structural rewrite as usage grew. Retrofitting a flat cache into a bucketed one later would have required changing the data structure, the lookup logic, the stats tracking, and the delete path simultaneously — a much larger change than building it correctly the first time.

- **Designing for the miss path matters as much as designing for the hit path**

  Most cache design thinking focuses on the hit case — how fast can a stored result be returned. The miss path here determines the experience for every first-time query and for any query that genuinely needs a fresh search, which is the majority of traffic early in the cache's life. Keeping miss latency at ~250ms required being deliberate about not adding unnecessary work: no recomputing memberships after a result is stored, no synchronous disk writes during the request, no redundant similarity checks on the result before returning it.

- **Building from scratch forces you to understand what the abstraction is actually doing**

  Using Redis or a caching library would have reduced the cache to a few configuration lines. Writing it from scratch — the bucket dict, the cosine similarity scan, the stats counters, the flush logic — made it impossible to treat the cache as a black box. Every design question had to be answered explicitly: what does the data structure look like, how does cluster membership change the lookup order, what exactly gets reset on a DELETE. That process produced a system where every behaviour is understood and intentional rather than inherited from a library default.

---

## Notebooks

| Notebook | Contents |
|:---------|:---------|
| [Part 1](Part_1—_Embedding_+_ChromaDB_vector_store_.ipynb) | Corpus cleaning with justification, embedding, ChromaDB setup, index verification |
| [Part 2](Part_2_—_Fuzzy_clustering.ipynb) | K selection with silhouette analysis, FCM, UMAP, entropy map, cluster naming |
| [Part 3](Part_3_—_FastAPI_semantic_cache_.ipynb) | Cache design, threshold calibration, similarity distributions, full test suite |

---

## Stack

| Layer | Technology |
|:------|:----------|
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector Store | ChromaDB 0.4.24 (persistent, cosine space) |
| Clustering | Fuzzy C-Means (`scikit-fuzzy`), UMAP for visualisation |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS — no framework, no build step |
| Runtime | Python 3.11, numpy < 2.0 |
| Development | Google Colab (T4 GPU for embedding), VS Code locally |

---

*Built end-to-end as a machine learning engineering project —  
Parts 1–3 developed iteratively on Google Colab, final service packaged as a local FastAPI deployment.*