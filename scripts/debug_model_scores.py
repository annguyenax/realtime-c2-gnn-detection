"""
Debug: tại sao model không phát hiện botnet nodes trong demo parquet?
"""
import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import polars as pl
from pathlib import Path
from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.data.flow_builder import FlowRecord
from c2gnn.models.graphsage import GraphSAGEC2Detector

BOTNET_IPS = {
    '147.32.84.165','147.32.84.191','147.32.84.192','147.32.84.193',
    '147.32.84.204','147.32.84.205','147.32.84.206','147.32.84.207',
    '147.32.84.208','147.32.84.209','147.32.96.69',
}

df = pl.read_parquet('data/processed/scenario10_demo.parquet').sort('timestamp')
print(f'Demo parquet: {len(df)} flows')
print(f'Botnet flows: {(df["label"] == "botnet").sum()}  ({(df["label"] == "botnet").mean()*100:.1f}%)')
print(f'Time span: {df["timestamp"].max() - df["timestamp"].min():.1f}s')
print()

# Build graph from ALL 20000 flows
graph = SlidingWindowGraph(window_size=60.0, edge_ttl=120.0)
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
    if (i+1) % 5000 == 0:
        print(f'  [{i+1}/{len(df)}] nodes={graph.num_nodes}  edges={graph.num_edges}')

print()
print(f'Graph: {graph.num_nodes} nodes, {graph.num_edges} edges')
stats = graph.stats
print(f'Botnet nodes in window: {stats.get("botnet_nodes_in_window", "N/A")}')

# Get PyG data
data = graph.to_pyg_data(include_ground_truth=True)
print(f'\nPyG: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges')
print(f'Feature dim: {data.x.shape[1]}')
print(f'Botnet labels in PyG: {int(data.y.sum())} nodes')

# Show feature values for botnet nodes
print('\n=== Features of known botnet nodes ===')
feature_names = [
    'out_degree','in_degree','out_bytes','in_bytes','mean_duration',
    'top_proto_frac','top_dst_port_frac','mean_pkt_rate','mean_byte_rate',
    'flow_iat_mean','flow_iat_std','iat_cv','beacon_score',
    'out_unique_dst','unique_ports','bytes_ratio',
    'temporal_iat_cv','temporal_pkt_rate'
]

for i, ip in enumerate(data.node_ips):
    if ip in BOTNET_IPS:
        feat = data.x[i]
        label = int(data.y[i].item())
        print(f'\n  {ip}  (label={label})')
        for j, (name, val) in enumerate(zip(feature_names, feat)):
            print(f'    [{j:2d}] {name:25s} = {val:.4f}')

# Load model and score
ckpt_path = Path('models/artifacts/graphsage_best.pt')
if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    # Check first layer shape
    first_key = list(state.keys())[0]
    print(f'\nModel first layer key: {first_key}  shape: {state[first_key].shape}')

    model = GraphSAGEC2Detector(hidden_channels=128)
    try:
        model.load_state_dict(state)
        model.eval()
        print('Model loaded OK')
        with torch.no_grad():
            logits = model(data.x, data.edge_index, None)
            probs = torch.softmax(logits, dim=-1)

        pmin = probs[:,1].min().item()
        pmax = probs[:,1].max().item()
        pmean = probs[:,1].mean().item()
        print(f'\nScore distribution: min={pmin:.4f}  max={pmax:.4f}  mean={pmean:.4f}')

        above_03 = (probs[:,1] > 0.3).sum().item()
        above_05 = (probs[:,1] > 0.5).sum().item()
        above_09 = (probs[:,1] > 0.9118).sum().item()
        print(f'Nodes above 0.3: {above_03}  |  above 0.5: {above_05}  |  above 0.9118: {above_09}')

        print('\n=== Botnet node scores ===')
        for i, ip in enumerate(data.node_ips):
            if ip in BOTNET_IPS:
                score = probs[i,1].item()
                label = int(data.y[i].item())
                print(f'  {ip:22s}  prob={score:.4f}  label={label}')

        print('\n=== Top 15 highest scores ===')
        scores = [(data.node_ips[i], probs[i,1].item(), int(data.y[i].item())) for i in range(len(data.node_ips))]
        scores.sort(key=lambda x: -x[1])
        for ip, score, label in scores[:15]:
            tag = '[BOT]' if ip in BOTNET_IPS else ''
            print(f'  {ip:22s}  prob={score:.4f}  label={label}  {tag}')

    except Exception as e:
        print(f'Model load ERROR: {e}')
else:
    print('Model checkpoint not found at models/artifacts/graphsage_best.pt')
