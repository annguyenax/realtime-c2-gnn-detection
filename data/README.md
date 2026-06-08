# Data Directory

> **Raw data is NOT committed to Git.** Tracked by DVC. Run `dvc pull` after setup.

## Directory Structure

```
data/
├── raw/
│   └── ctu13/          # Original CTU-13 .binetflow files (DVC tracked)
├── interim/            # Partially processed files
└── processed/          # Final .parquet files ready for training
    ├── ctu13_scenario10_flows.parquet
    ├── ctu13_scenario08_flows.parquet
    └── ctu13_combined_flows.parquet
```

## Dataset: CTU-13 Botnet Dataset

| Property | Value |
|----------|-------|
| Source | Czech Technical University (CTU), Prague |
| URL | https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/ |
| Format | NetFlow v5 (`.binetflow`) |
| Scenarios | 13 botnet scenarios (different malware families) |
| Recommended | Scenarios 1, 8, 9, 10, 11 for training |
| License | Public research dataset |

### Botnet Families in CTU-13

| Scenario | Botnet | Flows | Bot IPs |
|----------|--------|-------|---------|
| 1 | Neris (IRC C2) | 2,824,636 | 1 |
| 8 | Rbot (IRC C2) | 1,227,654 | 10 |
| 9 | Neris (HTTP C2) | 1,085,742 | 1 |
| 10 | Murlo (IRC C2) | 1,309,791 | 1 |
| 11 | Rbot (IRC C2) | 2,053,045 | 10 |

## Download Instructions

```bash
# Create data directories
mkdir -p data/raw/ctu13

# Scenario 10 — Murlo botnet (primary training set)
wget -P data/raw/ctu13/ \
  "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42/capture20110818.pcap.netflow.labeled"

# Scenario 8 — Rbot (generalization test)
wget -P data/raw/ctu13/ \
  "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-44/capture20110818-2.pcap.netflow.labeled"

# Scenario 1 — Neris original
wget -P data/raw/ctu13/ \
  "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42/capture20110818.binetflow"
```

## Label Schema

```
Flow label → Binary class
──────────────────────────────────────────────────────────────
"Botnet"                   → 1  (malicious)
"Background"               → 0  (benign)
"LEGITIMATE"               → 0  (benign)
"Normal"                   → 0  (benign)
```

## Preprocessing

Run after downloading:
```bash
make preprocess
# or directly:
python -m c2gnn.data.preprocess --input data/raw/ctu13 --output data/processed --scenario 10
```

Output: `data/processed/ctu13_scenario10_flows.parquet` (~200MB)

## Class Imbalance

Typical split in Scenario 10:
- Benign flows: ~97.5%
- Botnet flows: ~2.5%

The pipeline handles imbalance via:
1. `scale_pos_weight` in XGBoost
2. Weighted sampling in GraphSAGE NeighborLoader
3. Focal loss option in GATv2 trainer
