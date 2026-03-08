# Test Cases — Semantic Cache

Each group has two queries. Type the **MISS query first**, then the **HIT query**.  
A green **Cache Hit** badge means the cache recognised the paraphrase. A red **Cache Miss** means it computed fresh results.

> **Reset between full runs:** Click the **Flush Cache** button on the UI before starting again.

---

## Group 1 — Space & Astronomy

| | Query |
|---|---|
| 1️⃣ MISS | `NASA space shuttle launch into orbit` |
| 2️⃣ HIT  | `space shuttle mission launched by NASA` |

Expected similarity score: ~0.80

---

## Group 2 — Windows & Software

| | Query |
|---|---|
| 1️⃣ MISS | `Windows 95 driver crash blue screen of death` |
| 2️⃣ HIT  | `BSOD driver error on Windows system crash` |

---

## Group 3 — Cryptography & Privacy

| | Query |
|---|---|
| 1️⃣ MISS | `public key encryption and RSA algorithm` |
| 2️⃣ HIT  | `RSA cryptography asymmetric public key system` |

---

## Group 4 — Politics & Guns

| | Query |
|---|---|
| 1️⃣ MISS | `gun control laws and second amendment rights` |
| 2️⃣ HIT  | `second amendment firearm ownership restrictions` |

---

## Group 5 — Sports

| | Query |
|---|---|
| 1️⃣ MISS | `NHL ice hockey playoff game overtime goal` |
| 2️⃣ HIT  | `hockey playoffs sudden death overtime scoring` |

---

## Group 6 — Mac Hardware

| | Query |
|---|---|
| 1️⃣ MISS | `Apple Macintosh RAM upgrade memory expansion` |
| 2️⃣ HIT  | `Mac memory upgrade adding more RAM` |

---

## Group 7 — Religion & Ethics

| | Query |
|---|---|
| 1️⃣ MISS | `existence of God and religious faith arguments` |
| 2️⃣ HIT  | `theological debate on divine existence and belief` |

---

## Group 8 — Medical & Science (Exact Repeat)

| | Query |
|---|---|
| 1️⃣ MISS | `medical treatment for lower back pain relief` |
| 2️⃣ HIT  | `medical treatment for lower back pain relief` |

Expected similarity score: **1.0000** (exact match)

---

## Group 9 — Cross-Cluster (Both must be MISS)

| | Query |
|---|---|
| 1️⃣ MISS | `telescope observing distant galaxies and nebulae` |
| 2️⃣ MISS | `car engine oil change maintenance schedule` |

These are from different clusters — the second query must **not** hit the first.

---

## Expected Stats After All 9 Groups

| Metric | Expected |
|--------|----------|
| `total_entries` | 9 |
| `hit_count` | 8 |
| `miss_count` | 9 |
| `hit_rate` | ~0.47 |

Check via: `GET /cache/stats` or the stats panel on the UI.
