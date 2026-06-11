# Data Directory

> **Raw data is NOT committed to Git** (files are large). Download manually via the instructions below.

## Directory Structure

```
data/
├── raw/
│   └── ctu13/
│       └── scenario10.binetflow   # CTU-13 Scenario 10 — primary training set
├── interim/                        # Partially processed files
└── processed/                      # Final .parquet files for XGBoost training
    ├── ctu13_scenario10_flows.parquet
    └── dataset_stats.json
```

---

## Dataset: CTU-13 Botnet Dataset

### Official Description

- **Homepage:** https://www.stratosphereips.org/datasets-ctu13
- **Authoritative download:** https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/
- **Citation:** Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014).
  *An empirical comparison of botnet detection methods.*
  Computers & Security, 45, 100–123.
  https://doi.org/10.1016/j.cose.2014.05.011

### Key Properties

| Property | Value |
|---|---|
| Institution | Czech Technical University (CTU), Prague |
| Capture tool | Argus (bidirectional NetFlow) |
| Format | `.binetflow` — bidirectional NetFlow/Argus CSV |
| Scenarios | 13 botnet scenarios (different malware families and C2 protocols) |
| Year captured | 2011 |
| License | Public research dataset, free for academic use |

### What "binetflow" means

CTU-13 uses **bidirectional Argus flows** (`.binetflow`), not raw PCAP.
Each row = one bidirectional flow with: timestamp, duration, protocol, src/dst IP,
src/dst port, flags, bytes, packets, label.
This is NOT the same as standard NetFlow v5/v9 — it is Argus-generated and captures
both directions in one record (unlike unidirectional NetFlow).

### Appropriateness for this project

CTU-13 is well-suited for **proof-of-concept C2 graph detection** because:
- It includes real botnet traffic (not synthetic).
- It has structured periodic C2 beaconing (IRC, HTTP).
- Flow-level labels are available per-IP per-flow.

**Known limitations:**
- Dataset was captured in 2011. Modern C2 uses HTTPS/TLS, DGA, CDN fronting — not covered here.
- Botnet families are IRC/HTTP based (Neris, Rbot, Murlo). Fileless/encrypted C2 would not be detected.
- Single-scenario training does not guarantee cross-scenario generalization.
- XGBoost achieves very high F1 partly because Neris uses IRC (port 6667), a discriminative port feature.

---

## Scenarios Used

| Scenario | Botnet Family | C2 Protocol | Flows | Bot IPs | Role |
|---|---|---|---|---|---|
| **10** | Murlo | IRC (port 6667) | 5,178,417 | 1 | **Primary training + test** |
| 8 | Rbot | IRC | 1,227,654 | 10 | Cross-scenario eval (future) |

### Scenario 10 Statistics (verified)

| Split | Flows | Botnet flows | Botnet% |
|---|---|---|---|
| Full | 5,178,417 | 322,158 | 6.22% |
| Train (70%) | ~3,624,892 | ~225,511 | ~6.22% |
| Val (15%) | ~776,763 | ~48,324 | ~6.22% |
| Test (15%) | ~776,762 | ~48,323 | ~6.22% |

*Split is temporal (no shuffle) to prevent data leakage.*

---

## Label Schema

```
binetflow Label field → Binary class
─────────────────────────────────────────────────────
"Botnet"           → 1  (C2 traffic, malicious)
"Background"       → 0  (benign background traffic)
"LEGITIMATE"       → 0  (benign legitimate traffic)
"Normal"           → 0  (benign normal traffic)
```

**Node labeling:** A graph node (IP address) is labeled `botnet=1` if
**any adjacent edge** in the current time window is a botnet flow.
This may over-label hub nodes (DNS servers, gateways) that happen to
communicate with a botnet IP — acknowledged as a known limitation.

---

## Download Instructions

```bash
# Create data directory
mkdir -p data/raw/ctu13

# Scenario 10 — Murlo IRC botnet (primary)
# Source: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42-1/
wget -O data/raw/ctu13/scenario10.binetflow \
  "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42-1/capture20110818.binetflow"
```

After downloading, run preprocessing:
```bash
python scripts/02_preprocess.py
```

---

## Class Imbalance (Scenario 10)

| Class | Count | Percentage |
|---|---|---|
| Background/Normal | 4,856,259 | 93.78% |
| **Botnet** | **322,158** | **6.22%** |
| Imbalance ratio | ~15:1 | — |

**Graph-level imbalance is more extreme:** After building graph snapshots with `window_size=60s`,
botnet *nodes* in each snapshot are ~0.5–2% of total nodes due to aggregation effects.

Imbalance handling in this project:
1. `scale_pos_weight` in XGBoost
2. `max_class_weight=50` cap in GraphSAGE/GATv2 (prevents precision collapse)
3. `filter_empty_snapshots=True` (removes all-normal training snapshots)
4. Val-based threshold tuning (finds optimal decision boundary post-training)
