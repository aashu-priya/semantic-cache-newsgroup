# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   PART 4 — FASTAPI SEMANTIC CACHE SERVICE                               ║
# ║                                                                          ║
# ║   Start with:                                                            ║
# ║     uvicorn main:app --host 0.0.0.0 --port 8000 --reload               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import json
import time
import logging
import numpy as np
import chromadb
import torch

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = os.environ.get(
    "PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "newsgroups_chromadb")
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHE DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CacheEntry:
    query_text      : str
    embedding       : np.ndarray
    result          : Any
    memberships     : np.ndarray
    primary_cluster : int
    timestamp       : float = field(default_factory=time.time)
    hit_count       : int   = 0


class SemanticCache:
    def __init__(self, tau, K, centroids, cluster_names,
                 fuzziness=2.0, secondary_threshold=0.2):
        self.tau                 = tau
        self.K                   = K
        self.centroids           = centroids
        self.cluster_names       = cluster_names
        self.fuzziness           = fuzziness
        self.secondary_threshold = secondary_threshold
        self.buckets             = {i: [] for i in range(K)}
        self.hit_count           = 0
        self.miss_count          = 0

    def _get_memberships(self, embedding):
        dist        = cdist([embedding], self.centroids, metric='cosine')[0]
        sim         = np.clip(1.0 - dist + 1.0, 1e-10, None)
        sim_powered = sim ** self.fuzziness
        return sim_powered / sim_powered.sum()

    def _get_search_buckets(self, memberships):
        sorted_clusters   = np.argsort(memberships)[::-1]
        primary           = int(sorted_clusters[0])
        buckets_to_search = [primary]
        secondary         = int(sorted_clusters[1])
        if memberships[secondary] >= self.secondary_threshold:
            buckets_to_search.append(secondary)
        return buckets_to_search

    def lookup(self, embedding):
        memberships = self._get_memberships(embedding)
        buckets     = self._get_search_buckets(memberships)
        best_sim    = -1.0
        best_entry  = None
        for bucket_id in buckets:
            for entry in self.buckets[bucket_id]:
                sim = 1.0 - cosine_distance(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim   = sim
                    best_entry = entry
        if best_sim >= self.tau:
            self.hit_count += 1
            best_entry.hit_count += 1
            return True, best_entry, float(best_sim), buckets
        else:
            self.miss_count += 1
            return False, None, float(best_sim), buckets

    def store(self, query_text, embedding, result):
        memberships     = self._get_memberships(embedding)
        primary_cluster = int(np.argmax(memberships))
        entry = CacheEntry(
            query_text=query_text, embedding=embedding, result=result,
            memberships=memberships, primary_cluster=primary_cluster,
        )
        self.buckets[primary_cluster].append(entry)
        return entry

    def flush(self):
        self.buckets    = {i: [] for i in range(self.K)}
        self.hit_count  = 0
        self.miss_count = 0

    def get_all_entries(self):
        entries = []
        for cluster_id, bucket in self.buckets.items():
            for entry in bucket:
                entries.append({
                    "query"           : entry.query_text,
                    "cluster_id"      : cluster_id,
                    "cluster_name"    : self.cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                    "hit_count"       : entry.hit_count,
                    "timestamp"       : entry.timestamp,
                })
        return entries

    @property
    def total_entries(self):
        return sum(len(b) for b in self.buckets.values())

    @property
    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APP STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppState:
    model      : SentenceTransformer
    collection : Any
    cache      : SemanticCache
    device     : str
    manifest   : dict

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("── Starting up ───────────────────────────────────────────")
    logger.info(f"PERSIST_DIR : {PERSIST_DIR}")

    required = ["manifest.json", "chroma.sqlite3",
                "embeddings_backup.npy", "fuzzy_memberships.npy",
                "kmeans_centroids.npy"]
    for fname in required:
        fpath = os.path.join(PERSIST_DIR, fname)
        if not os.path.exists(fpath):
            raise RuntimeError(f"Required file missing: {fpath}")

    with open(os.path.join(PERSIST_DIR, "manifest.json")) as f:
        state.manifest = json.load(f)

    if "part2" not in state.manifest:
        raise RuntimeError("manifest.json has no 'part2' section.")

    client = chromadb.PersistentClient(
        path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    state.collection = client.get_collection(state.manifest["collection_name"])
    logger.info(f"ChromaDB    : {state.collection.count():,} docs")

    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    state.model  = SentenceTransformer(
        state.manifest["embedding_model"], device=state.device)
    logger.info(f"Model       : {state.manifest['embedding_model']} device={state.device}")

    p2            = state.manifest["part2"]
    centroids     = np.load(p2["kmeans_centroids"])
    cluster_names = {int(k): v for k, v in p2["cluster_names"].items()}
    K             = p2["K"]
    fuzziness     = p2["fuzziness_m"]
    tau           = state.manifest.get("part3", {}).get("default_tau", 0.44)

    state.cache = SemanticCache(
        tau=tau, K=K, centroids=centroids,
        cluster_names=cluster_names, fuzziness=fuzziness,
    )

    logger.info(f"Cache       : K={K}  τ={tau}  fuzziness={fuzziness}")
    logger.info(f"Clusters    : {list(cluster_names.values())}")
    logger.info("── Ready ─────────────────────────────────────────────────")
    yield
    logger.info("── Shutting down ─────────────────────────────────────────")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Semantic Cache Service",
    description="Cluster-aware semantic cache over 20 Newsgroups.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QueryRequest(BaseModel):
    query : str

class QueryResponse(BaseModel):
    query            : str
    cache_hit        : bool
    matched_query    : Optional[str]
    similarity_score : Optional[float]
    result           : Any
    dominant_cluster : int
    dominant_cluster_name : str


class StatsResponse(BaseModel):
    total_entries : int
    hit_count     : int
    miss_count    : int
    hit_rate      : float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_semantic_search(query: str, embedding: np.ndarray, n: int = 6) -> dict:
    t0  = time.time()
    res = state.collection.query(
        query_embeddings=embedding.tolist(),
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    results = [
        {
            "category"   : meta["category"],
            "similarity" : round(1.0 - dist, 4),
            "snippet"    : doc[:300],
        }
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]
    return {
        "query"        : query,
        "results"      : results,
        "compute_time" : round(time.time() - t0, 4),
        "n_results"    : len(results),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
async def serve_frontend():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Semantic Cache API", "docs": "/docs"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    embedding = state.model.encode(
        [req.query], normalize_embeddings=True,
        device=state.device, show_progress_bar=False,
    )[0]

    memberships           = state.cache._get_memberships(embedding)
    dominant_cluster      = int(np.argmax(memberships))
    dominant_cluster_name = state.cache.cluster_names.get(
        dominant_cluster, f"Cluster {dominant_cluster}")

    hit, entry, best_sim, _ = state.cache.lookup(embedding)

    if hit:
        logger.info(f"HIT  sim={best_sim:.4f} q='{req.query[:60]}'")
        return QueryResponse(
            query=req.query, cache_hit=True,
            matched_query=entry.query_text,
            similarity_score=round(best_sim, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
            dominant_cluster_name=dominant_cluster_name,
        )
    else:
        result = run_semantic_search(req.query, embedding)
        state.cache.store(query_text=req.query, embedding=embedding, result=result)
        logger.info(f"MISS compute={result['compute_time']}s q='{req.query[:60]}'")
        return QueryResponse(
            query=req.query, cache_hit=False,
            matched_query=None,
            similarity_score=round(best_sim, 4) if best_sim >= 0 else None,
            result=result,
            dominant_cluster=dominant_cluster,
            dominant_cluster_name=dominant_cluster_name,
        )


@app.get("/cache/stats", response_model=StatsResponse)
async def cache_stats():
    return StatsResponse(
        total_entries=state.cache.total_entries,
        hit_count=state.cache.hit_count,
        miss_count=state.cache.miss_count,
        hit_rate=round(state.cache.hit_rate, 4),
    )


@app.get("/cache/entries")
async def cache_entries():
    return JSONResponse(content={
        "total_entries" : state.cache.total_entries,
        "tau"           : state.cache.tau,
        "entries"       : state.cache.get_all_entries(),
    })


@app.delete("/cache")
async def cache_flush():
    state.cache.flush()
    logger.info("Cache flushed")
    return JSONResponse(content={
        "status"  : "flushed",
        "message" : "Cache cleared and stats reset",
        "tau"     : state.cache.tau,
        "K"       : state.cache.K,
    })


@app.get("/health")
async def health():
    return {
        "status"        : "ok",
        "cache_entries" : state.cache.total_entries,
        "hit_rate"      : round(state.cache.hit_rate, 4),
        "tau"           : state.cache.tau,
        "K"             : state.cache.K,
        "device"        : state.device,
        "collection"    : state.collection.count(),
        "cluster_names" : state.cache.cluster_names,
    }