# Outputs — Visual Analysis

All plots generated during Parts 1–3 of the Semantic Cache pipeline.

---

## plot_k_selection.png
**What it shows:** Silhouette score vs number of clusters K (tested K=2 to 15).  
**How to read it:** The peak silhouette score indicates the optimal K. Our system selected K=11 as the best cluster count for the 20 Newsgroups embedding space.

---

## plot_umap_clusters.png
**What it shows:** 2D UMAP projection of all 15,176 document embeddings, coloured by cluster assignment.  
**How to read it:** Each colour is one of the 11 fuzzy clusters. Tight groupings indicate semantically coherent topics. Overlapping regions indicate documents that belong to multiple clusters (high fuzzy membership entropy).

---

## plot_entropy_map.png
**What it shows:** Per-document fuzzy membership entropy mapped onto the UMAP projection.  
**How to read it:** High entropy (bright) = document is spread across many clusters and is topically ambiguous. Low entropy (dark) = document sits firmly in one cluster. Mean entropy was 2.368 ≈ max 2.398, meaning embeddings are very diffuse in this corpus.

---

## plot_cluster_composition.png
**What it shows:** Stacked bar chart showing the distribution of original 20 Newsgroups categories within each of the 11 clusters.  
**How to read it:** A cluster dominated by one category (e.g. sci.space) is semantically pure. Mixed clusters indicate topical overlap between newsgroup categories.

---

## plot_threshold_exploration.png
**What it shows:** Similarity score distributions for three query pair types — paraphrases, same-topic different queries, and cross-cluster queries — used to calibrate the cache similarity threshold τ.  
**How to read it:** The gap between same-topic ceiling (0.29) and paraphrase floor (0.51) is 0.22. τ=0.44 sits at the midpoint of this gap, giving clean separation between hits and misses.

---

## Cluster Names (K=11)

| Index | Name |
|-------|------|
| 0 | Space & Astronomy |
| 1 | PC Hardware & Tech |
| 2 | Politics & Guns |
| 3 | Religion & Ethics |
| 4 | Sports |
| 5 | Cryptography & Privacy |
| 6 | Windows & Software |
| 7 | Medical & Science |
| 8 | Autos & Motorcycles |
| 9 | Mac Hardware |
| 10 | Middle East Politics |
