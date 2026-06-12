"""
Find optimal threshold for parquet data and verify it produces actual TP detections.
Uses the same evaluation setup as training but on parquet.
"""
import sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.data.flow_builder import FlowRecord
from c2gnn.models.graphsage import GraphSAGEC2Detector

WINDOW, TTL, SNAPSHOT_INTERVAL = 300.0, 600.0, 150.0

print('Loading parquet + model...')
df = pl.read_parquet('data/processed/scenario10_test.parquet').sort('timestamp')
model = GraphSAGEC2Detector(hidden_channels=128)
ckpt = torch.load('models/artifacts/graphsage_best.pt', map_location='cpu')
model.load_state_dict(ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt)
model.eval()
print(f'  {len(df):,} flows loaded')

graph = SlidingWindowGraph(window_size=WINDOW, edge_ttl=TTL)
snapshots = []
last_snap_ts = None
t_start = time.time()

for i, row in enumerate(df.iter_rows(named=True)):
    flow = FlowRecord(
        timestamp=float(row['timestamp']), src_ip=str(row['src_ip']), dst_ip=str(row['dst_ip']),
        src_port=int(row.get('src_port',0)), dst_port=int(row.get('dst_port',0)),
        protocol=str(row.get('protocol','OTHER')), duration=float(row.get('duration',0)),
        total_fwd_packets=int(row.get('total_fwd_packets',0)), total_bwd_packets=int(row.get('total_bwd_packets',0)),
        total_bytes=int(row.get('total_bytes',0)), packet_rate=float(row.get('packet_rate',0)),
        byte_rate=float(row.get('byte_rate',0)), flow_iat_mean=float(row.get('flow_iat_mean',0)),
        flow_iat_std=float(row.get('flow_iat_std',0)), label=str(row.get('label','background')),
    )
    graph.update(flow)
    if last_snap_ts is None:
        last_snap_ts = flow.timestamp
    if flow.timestamp - last_snap_ts >= SNAPSHOT_INTERVAL:
        data = graph.to_pyg_data(include_ground_truth=True)
        if data is not None and data.x.shape[0] >= 3:
            snapshots.append(data)
        last_snap_ts = flow.timestamp
    if (i+1) % 300000 == 0:
        print(f'  [{i+1:,}] snaps={len(snapshots)} t={time.time()-t_start:.0f}s')

print(f'\nBuilt {len(snapshots)} snapshots in {time.time()-t_start:.0f}s')

# 70/15/15 split
n = len(snapshots)
n_train = int(n * 0.70); n_val = int(n * 0.85)
val_snaps = snapshots[n_train:n_val]
test_snaps = snapshots[n_val:]
print(f'Val: {len(val_snaps)}  Test: {len(test_snaps)}')

# Collect all probs on val+test for threshold tuning
def collect_probs(snaps):
    probs, labels = [], []
    with torch.no_grad():
        for snap in snaps:
            if not hasattr(snap, 'y') or snap.y is None:
                continue
            logits = model(snap.x, snap.edge_index, None)
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(p.numpy())
            labels.append(snap.y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

print('\nCollecting predictions...')
val_probs, val_labels = collect_probs(val_snaps)
test_probs, test_labels = collect_probs(test_snaps)

print(f'\nVal set: {len(val_labels)} nodes, {val_labels.sum()} botnet')
print(f'Test set: {len(test_labels)} nodes, {test_labels.sum()} botnet')

roc = roc_auc_score(val_labels, val_probs) if val_labels.sum() > 0 and val_labels.sum() < len(val_labels) else 0
print(f'Val ROC-AUC: {roc:.4f}')

print('\nBotnet score distribution (val):')
bot = val_probs[val_labels == 1]
bg = val_probs[val_labels == 0]
print(f'  Botnet: min={bot.min():.4f}  p25={np.percentile(bot,25):.4f}  p50={np.percentile(bot,50):.4f}  p75={np.percentile(bot,75):.4f}  max={bot.max():.4f}')
print(f'  Background: min={bg.min():.4f}  p75={np.percentile(bg,75):.4f}  p90={np.percentile(bg,90):.4f}  p95={np.percentile(bg,95):.4f}  max={bg.max():.4f}')

# Find optimal threshold on VAL (not test - no leakage)
if val_labels.sum() > 0 and val_labels.sum() < len(val_labels):
    prec_arr, rec_arr, thresholds = precision_recall_curve(val_labels, val_probs)
    best = {'threshold': 0.5, 'f1': 0, 'precision': 0, 'recall': 0}
    for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresholds):
        if r < 0.3: continue
        f1 = 2*p*r/max(p+r, 1e-12)
        if f1 > best['f1']:
            best = {'threshold': float(t), 'f1': float(f1), 'precision': float(p), 'recall': float(r)}
    print(f'\nOptimal threshold (val, min_recall=0.3): {best["threshold"]:.4f}')
    print(f'  Val F1={best["f1"]:.4f}  Prec={best["precision"]:.4f}  Rec={best["recall"]:.4f}')
    OPT_THRESH = best['threshold']
else:
    OPT_THRESH = 0.5
    print('Not enough class diversity in val for threshold tuning')

# Evaluate on test with found threshold
print(f'\n=== Test evaluation at threshold={OPT_THRESH:.4f} ===')
y_pred = (test_probs > OPT_THRESH).astype(int)
tp = ((y_pred==1)&(test_labels==1)).sum()
fp = ((y_pred==1)&(test_labels==0)).sum()
fn = ((y_pred==0)&(test_labels==1)).sum()
tn = ((y_pred==0)&(test_labels==0)).sum()
precision = tp/max(tp+fp,1)
recall = tp/max(tp+fn,1)
f1 = 2*precision*recall/max(precision+recall,1e-12)
fpr = fp/max(fp+tn,1)
print(f'  TP={tp} FP={fp} FN={fn} TN={tn}')
print(f'  F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}  FPR={fpr*100:.4f}%')

# Show what IPs are detected as TP
tp_ips, fp_ips = set(), set()
for snap in test_snaps:
    if not hasattr(snap, 'y'): continue
    with torch.no_grad():
        logits = model(snap.x, snap.edge_index, None)
        probs = torch.softmax(logits, dim=-1)[:, 1]
    for i, ip in enumerate(snap.node_ips):
        if probs[i].item() > OPT_THRESH:
            if snap.y[i].item() == 1:
                tp_ips.add(ip)
            else:
                fp_ips.add(ip)
print(f'\nTP IPs: {sorted(tp_ips)}')
print(f'FP IPs (sample): {sorted(list(fp_ips))[:10]}')

# Also check known training threshold 0.9118
print(f'\n=== Test at TRAINING threshold=0.9118 ===')
y_pred2 = (test_probs > 0.9118).astype(int)
tp2 = ((y_pred2==1)&(test_labels==1)).sum()
fp2 = ((y_pred2==1)&(test_labels==0)).sum()
fn2 = ((y_pred2==0)&(test_labels==1)).sum()
prec2 = tp2/max(tp2+fp2,1)
rec2 = tp2/max(tp2+fn2,1)
f12 = 2*prec2*rec2/max(prec2+rec2,1e-12)
print(f'  TP={tp2} FP={fp2} FN={fn2}')
print(f'  F1={f12:.4f}  Prec={prec2:.4f}  Rec={rec2:.4f}')
print(f'\nConclusion: use threshold={OPT_THRESH:.4f} for parquet-based realtime demo')
