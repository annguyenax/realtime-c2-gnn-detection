"""
Simulate at flow 900k (in test set range, t=2759s) with window_size=300 to check if
model correctly detects botnet nodes as it would during training evaluation.
"""
import sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import polars as pl
from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.data.flow_builder import FlowRecord
from c2gnn.models.graphsage import GraphSAGEC2Detector

BOTNET_IPS = {
    '147.32.84.165','147.32.84.191','147.32.84.192','147.32.84.193',
    '147.32.84.204','147.32.84.205','147.32.84.206','147.32.84.207',
    '147.32.84.208','147.32.84.209','147.32.96.69',
}

print('Loading parquet...')
df = pl.read_parquet('data/processed/scenario10_test.parquet').sort('timestamp')
n_total = len(df)
print(f'  {n_total:,} flows loaded')

model = GraphSAGEC2Detector(hidden_channels=128)
ckpt = torch.load('models/artifacts/graphsage_best.pt', map_location='cpu')
state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(state)
model.eval()
print('  Model loaded')

WINDOW, TTL = 300.0, 600.0
graph = SlidingWindowGraph(window_size=WINDOW, edge_ttl=TTL)
N = 900000
t_start = time.time()

print(f'\nBuilding graph with {N:,} flows (window={WINDOW}s, ttl={TTL}s)...')
for i, row in enumerate(df.head(N).iter_rows(named=True)):
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
    if (i+1) % 100000 == 0:
        elapsed = time.time() - t_start
        print(f'  [{i+1:,}/{N:,}]  nodes={graph.num_nodes}  t={elapsed:.1f}s')

elapsed = time.time() - t_start
print(f'\nGraph built in {elapsed:.1f}s')
print(f'  Nodes: {graph.num_nodes}  Edges: {graph.num_edges}')
stats = graph.stats
print(f'  Botnet nodes in window: {stats["botnet_nodes_in_window"]}')
print(f'  Current time: {stats["current_time"]:.1f}s (real ts)')

data = graph.to_pyg_data(include_ground_truth=True)
print(f'  Botnet labels in PyG: {int(data.y.sum())}')

with torch.no_grad():
    logits = model(data.x, data.edge_index, None)
    probs = torch.softmax(logits, dim=-1)

print(f'\nScore dist: min={probs[:,1].min():.4f}  max={probs[:,1].max():.4f}  mean={probs[:,1].mean():.4f}')
for thresh in [0.9118, 0.7, 0.5]:
    above = (probs[:,1] > thresh).sum().item()
    above_gt = sum(1 for i in range(len(data.node_ips)) if probs[i,1].item() > thresh and data.y[i].item() == 1)
    print(f'  Above {thresh}: {above} total  ({above_gt} TP, {above-above_gt} FP)')

print('\n=== All botnet ground-truth nodes (y=1) ===')
for i, ip in enumerate(data.node_ips):
    if data.y[i].item() == 1:
        score = probs[i,1].item()
        nd = graph._node_data.get(ip)
        out_fl = nd.out_flows if nd else 0
        flag = ' <<< TP' if score > 0.9118 else ''
        known = '[known]' if ip in BOTNET_IPS else '[new]'
        print(f'  {ip:22s}  prob={score:.4f}  out_flows={out_fl:,}  {known}{flag}')

print('\n=== Top 10 highest scores ===')
scores = [(data.node_ips[i], probs[i,1].item(), int(data.y[i].item())) for i in range(len(data.node_ips))]
scores.sort(key=lambda x: -x[1])
for ip, score, label in scores[:10]:
    tag = '[BOT]' if label == 1 else ''
    print(f'  {ip:22s}  prob={score:.4f}  gt={label}  {tag}')
