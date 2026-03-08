# Semantic Cache — 20 Newsgroups

> A production-grade semantic search and caching system built from scratch — embedding 15,176 documents, clustering them into semantic groups, and serving results through a FastAPI backend with a live frontend UI.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What This Project Does

Most search systems treat every query as independent — even if someone already asked the same question in different words. This project solves that with a **cluster-aware semantic cache**: instead of re-running expensive vector search for every query, it recognises paraphrases and returns stored results instantly.

The system was built end-to-end across 4 parts:

| Part | What was built | Key outcome |
|------|---------------|-------------|
| **Part 1** | Embedded 15,176 newsgroup posts into a ChromaDB vector store | 384-dim MiniLM-L6-v2 embeddings, cosine space |
| **Part 2** | Fuzzy clustered the embedding space into K=11 semantic groups | Silhouette-selected K, fuzziness m=2.0 |
| **Part 3** | Built a similarity threshold calibration system | Data-driven τ=0.44, 22-point clean gap between hit/miss |
| **Part 4** | FastAPI service + live frontend UI | Sub-50ms cache hits, real-time stats dashboard |

---

## Live Demo

Start the service locally in one command:

```bash
git clone https://github.com/aashu-priya/semantic-cache-newsgroup.git
cd semantic-cache-newsgroups
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — type any natural language query and see results instantly.

> **Note:** Download the large data files (ChromaDB index + embeddings) from [Google Drive →](https://drive.google.com/drive/folders/1sAk1n_3nR-gfeBL3hxVjHVwfkqC-piUg?usp=sharing) and place them in `newsgroups_chromadb/`

---

## Architecture

```
User query (natural language)
        │
        ▼
  Embed with MiniLM-L6-v2
        │
        ▼
  Compute fuzzy cluster memberships
        │
        ▼
  Look up in cluster bucket  ──── HIT (sim ≥ τ=0.44) ──▶  Return cached result instantly
        │                                                    (< 5ms)
      MISS
        │
        ▼
  ChromaDB vector search (cosine)
        │
        ▼
  Store result in cache bucket
        │
        ▼
  Return fresh result
```

### Why cluster-bucketed cache?

A flat cache scans all N entries on every lookup — O(N). By routing queries into K=11 semantic buckets first, lookup becomes O(N/K) — an **11× speedup** at no accuracy cost. The fuzzy membership also searches a secondary bucket when a query sits near a cluster boundary, so edge cases are handled correctly.

---

## Technical Decisions Worth Noting

### Threshold calibration (τ=0.44)
Rather than picking τ by intuition, the threshold was calibrated empirically on this corpus:

```
Paraphrase pairs      →  similarity: 0.51 – 0.86  (floor = 0.51)
Same-topic diff query →  similarity: 0.00 – 0.29  (ceiling = 0.29)
Gap                   →  0.22  (clean separation)
τ = 0.44              →  midpoint of gap, data-driven
```

This matters because MiniLM on 20 Newsgroups produces more diffuse embeddings than typical benchmarks (entropy_mean = 2.368 ≈ entropy_max = 2.398). A generic τ=0.85 would miss nearly every valid hit.

### Fuzzy vs hard clustering
Hard clustering assigns each document to exactly one cluster. Fuzzy clustering (FCM, m=2.0) gives each document a membership probability across all K clusters. This means a query about "Mac vs Windows" correctly searches both the Mac Hardware and Windows & Software buckets — a hard clustering would miss one entirely.

### Why ChromaDB 0.4.24 + numpy < 2.0
ChromaDB's telemetry stack has a strict dependency on OpenTelemetry versions that conflict with numpy 2.x. All package versions are pinned to reproduce the exact Colab environment locally without breakage.

---

## Project Structure

```
semantic-cache-newsgroups/
│
├── main.py                        # FastAPI service — all endpoints
├── requirements.txt               # Pinned dependencies
│
├── static/
│   └── index.html                 # Frontend UI (single file, no build step)
│
├── newsgroups_chromadb/           # Data directory (large files on Drive)
│   ├── manifest.json              # Config — paths, model name, τ, K
│   ├── cluster_metadata.json      # Cluster names and assignments
│   └── ... (large files on Drive)
│
├── outputs/                       # All generated plots with explanations
│   ├── README.md
│   ├── plot_k_selection.png
│   ├── plot_umap_clusters.png
│   ├── plot_entropy_map.png
│   ├── plot_cluster_composition.png
│   └── plot_threshold_exploration.png
│
├── Part_1—_Embedding_+_ChromaDB_vector_store_.ipynb
├── Part_2_—_Fuzzy_clustering.ipynb
└── Part_3_—_FastAPI_semantic_cache_.ipynb
```

---

## API Reference

### `POST /query`
Accepts a natural language query, returns semantic search results from cache or ChromaDB.

```json
// Request
{ "query": "NASA space shuttle launch into orbit" }

// Response
{
  "query": "NASA space shuttle launch into orbit",
  "cache_hit": true,
  "matched_query": "space shuttle mission launched by NASA",
  "similarity_score": 0.8026,
  "dominant_cluster": 0,
  "dominant_cluster_name": "Space & Astronomy",
  "result": {
    "results": [...],
    "compute_time": 0.0
  }
}
```

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
Flushes all cached entries and resets stats.

### `GET /health`
Returns server status, model info, τ, K, and collection size.

---

## The 11 Semantic Clusters

| # | Cluster Name | Dominant Newsgroups |
|---|-------------|---------------------|
| 0 | Space & Astronomy | sci.space |
| 1 | PC Hardware & Tech | comp.sys.ibm.pc.hardware |
| 2 | Politics & Guns | talk.politics.guns |
| 3 | Religion & Ethics | talk.religion.misc, soc.religion.christian |
| 4 | Sports | rec.sport.hockey, rec.sport.baseball |
| 5 | Cryptography & Privacy | sci.crypt, talk.politics.misc |
| 6 | Windows & Software | comp.os.ms-windows.misc |
| 7 | Medical & Science | sci.med |
| 8 | Autos & Motorcycles | rec.autos, rec.motorcycles |
| 9 | Mac Hardware | comp.sys.mac.hardware |
| 10 | Middle East Politics | talk.politics.mideast |

---

## Visual Results

All plots are in the [`outputs/`](./outputs/) folder with full explanations.

| Plot | Description |
|------|-------------|
| [K Selection](./outputs/plot_k_selection.png) | Silhouette score vs K — shows why K=11 was chosen |
| [UMAP Clusters](./outputs/plot_umap_clusters.png) | 2D projection of 15,176 embeddings by cluster |
| [Entropy Map](./outputs/plot_entropy_map.png) | Per-document topical ambiguity across the corpus |
| [Cluster Composition](./outputs/plot_cluster_composition.png) | Category distribution within each cluster |
| [Threshold Exploration](./outputs/plot_threshold_exploration.png) | Similarity distributions used to calibrate τ |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Documents indexed | 15,176 |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Clusters (K) | 11 |
| Fuzziness (m) | 2.0 |
| Similarity threshold (τ) | 0.44 |
| Cache lookup complexity | O(N/K) — 11× faster than flat scan |
| Paraphrase similarity range | 0.51 – 0.86 |
| Same-topic miss ceiling | 0.29 |
| Separation gap | 0.22 |
| Vector index size | 384 MB |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector store | ChromaDB 0.4.24 (persistent, cosine space) |
| Clustering | Fuzzy C-Means (scikit-fuzzy), UMAP for visualisation |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Runtime | Python 3.11, numpy < 2.0 |
| Development | Google Colab (T4 GPU for embedding), VS Code locally |

---

## What I Learned

Building this end-to-end forced decisions that papers skip over — how to pick τ when your corpus is more diffuse than benchmarks, why fuzzy membership beats hard assignment for boundary queries, and how pinned dependency versions are the difference between a reproducible system and one that breaks on reinstall.

The most interesting finding was that MiniLM on 20 Newsgroups sits at entropy_mean ≈ entropy_max — meaning almost every document is semantically ambiguous across clusters. This makes the fuzzy approach not just theoretically nicer but practically necessary.

---

## Setup from Scratch

```bash
# 1. Clone
git clone https://github.com/aashu-priya/semantic-cache-newsgroup.git
cd semantic-cache-newsgroups

# 2. Virtual environment (Python 3.11 required)
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download large data files from Google Drive
# Place in newsgroups_chromadb/:
#   chroma.sqlite3         (384 MB)
#   embeddings_backup.npy  (23 MB)
#   fuzzy_memberships.npy
#   kmeans_centroids.npy

# 5. Fix local paths in manifest
python3 fix_paths.py   # or run the path-fix snippet from the docs

# 6. Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 7. Open browser
open http://localhost:8000
```

---

*Built as part of a machine learning internship project — Parts 1–3 developed iteratively on Google Colab with final deployment packaged as a local FastAPI service.*
