"""
Replicate training evaluation on parquet to understand model's actual behavior.
Build snapshots from parquet the same way as training did from binetflow.
"""
import sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.data.flow_builder import FlowRecord
from c2gnn.models.graphsage import GraphSAGEC2Detector

WINDOW = 300.0
TTL = 600.0
SNAPSHOT_INTERVAL = 150.0  # = window_size / 2

print('Loading parquet...')
df = pl.read_parquet('data/processed/scenario10_test.parquet').sort('timestamp')
n_total = len(df)
t0 = df['timestamp'][0]
t_end = df['timestamp'][-1]
print(f'  {n_total:,} flows  span={t_end-t0:.1f}s')

model = GraphSAGEC2Detector(hidden_channels=128)
ckpt = torch.load('models/artifacts/graphsage_best.pt', map_location='cpu')
state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(state)
model.eval()
print('  Model loaded')

print(f'\nBuilding snapshots (window={WINDOW}s, ttl={TTL}s, interval={SNAPSHOT_INTERVAL}s)...')

graph = SlidingWindowGraph(window_size=WINDOW, edge_ttl=TTL)
snapshots = []
last_snap_ts = None
t_start = time.time()

for i, row in enumerate(df.iter_rows(named=True)):
    flow = FlowRecord(
        timestamp=float(row['timestamp']),
        src_ip=str(row['src_ip']), dst_ip=str(row['dst_ip']),
        src_port=int(row.get('src_port', 0)), dst_port=int(row.get('dst_port', 0)),
        protocol=str(row.get('protocol', 'OTHER')),
        duration=float(row.get('duration', 0)),
        total_fwd_packets=int(row.get('total_fwd_packets', 0)),
        total_bwd_packets=int(row.get('total_bwd_packets', 0)),
        total_bytes=int(row.get('total_bytes', 0)),
        packet_rate=float(row.get('packet_rate', 0)),
        byte_rate=float(row.get('byte_rate', 0)),
        flow_iat_mean=float(row.get('flow_iat_mean', 0)),
        flow_iat_std=float(row.get('flow_iat_std', 0)),
        label=str(row.get('label', 'background')),
    )
    graph.update(flow)

    if last_snap_ts is None:
        last_snap_ts = flow.timestamp

    if flow.timestamp - last_snap_ts >= SNAPSHOT_INTERVAL:
        data = graph.to_pyg_data(include_ground_truth=True)
        if data is not None and data.x.shape[0] >= 3:
            snapshots.append(data)
        last_snap_ts = flow.timestamp

    if (i+1) % 200000 == 0:
        elapsed = time.time() - t_start
        print(f'  [{i+1:,}/{n_total:,}]  snaps={len(snapshots)}  t={elapsed:.1f}s')

elapsed = time.time() - t_start
print(f'\nBuilt {len(snapshots)} snapshots in {elapsed:.1f}s')

# 70/15/15 temporal split
n = len(snapshots)
n_train = int(n * 0.70)
n_val = int(n * 0.85)
train_snaps = snapshots[:n_train]
val_snaps = snapshots[n_train:n_val]
test_snaps = snapshots[n_val:]
print(f'Split: train={len(train_snaps)}  val={len(val_snaps)}  test={len(test_snaps)}')

# Evaluate on test snapshots
THRESHOLD = 0.911838
all_preds, all_labels, all_scores = [], [], []
tp_ips, fp_ips = set(), set()

with torch.no_grad():
    for snap in test_snaps:
        if not hasattr(snap, 'y') or snap.y is None:
            continue
        logits = model(snap.x, snap.edge_index, None)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs > THRESHOLD).long()
        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(snap.y.numpy().tolist())
        all_scores.extend(probs.numpy().tolist())

        # Track which IPs are TP/FP
        for i, ip in enumerate(snap.node_ips):
            if preds[i].item() == 1:
                if snap.y[i].item() == 1:
                    tp_ips.add(ip)
                else:
                    fp_ips.add(ip)

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_score = np.array(all_scores)

f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
tn, fp_n, fn, tp_n = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
fpr = fp_n / max(tn + fp_n, 1)
precision = tp_n / max(tp_n + fp_n, 1)
recall = tp_n / max(tp_n + fn, 1)
roc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0

print(f'\n=== Evaluation at threshold={THRESHOLD} ===')
print(f'  Total nodes evaluated: {len(y_true)}')
print(f'  Botnet nodes (gt=1):   {y_true.sum()}')
print(f'  F1:        {f1:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall:    {recall:.4f}')
print(f'  FPR:       {fpr*100:.4f}%')
print(f'  ROC-AUC:   {roc:.4f}')
print(f'  TP={tp_n}  FP={fp_n}  FN={fn}  TN={tn}')
print(f'\nTP IPs detected: {sorted(tp_ips)}')
print(f'FP IPs detected: {sorted(fp_ips)}')

# Distribution of scores for botnet vs background
bot_scores = y_score[y_true == 1]
bg_scores = y_score[y_true == 0]
print(f'\nBotnet score stats:     min={bot_scores.min():.4f}  max={bot_scores.max():.4f}  mean={bot_scores.mean():.4f}')
print(f'Background score stats: min={bg_scores.min():.4f}  max={bg_scores.max():.4f}  mean={bg_scores.mean():.4f}')
