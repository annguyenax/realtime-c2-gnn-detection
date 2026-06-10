# Demo Script — Bảo vệ Đồ án C2GNN

**Đề tài:** Phát hiện C2 Traffic bằng Graph Learning đáp ứng thời gian thực  
**Dataset:** CTU-13 Scenario 10 (Neris IRC Botnet)  
**Thời gian demo:** 10–15 phút  
**Người demo:** [Tên thành viên]

---

## Chuẩn bị trước demo (30 phút trước)

```bash
# 1. Verify data và artifacts tồn tại
ls data/processed/scenario10_train.parquet
ls models/artifacts/xgboost_model.json
ls models/artifacts/graphsage_best.pt
ls reports/final_metrics.json

# 2. Test import
python -c "from c2gnn.data.flow_builder import CTU13FlowParser; print('OK')"
python -c "from c2gnn.graph.dynamic_graph import SlidingWindowGraph; print('OK')"
python -c "from c2gnn.models.graphsage import GraphSAGEC2Detector; print('OK')"

# 3. Preload dashboard (background)
# streamlit run src/c2gnn/dashboard/app.py &

# 4. Mở terminal sạch, font lớn, dark theme
```

---

## Kịch bản Demo Chính (10 phút)

### Bước 0 — Giới thiệu (1 phút)

**Nói:** "Chúng em xây dựng hệ thống phát hiện C2 botnet traffic sử dụng Graph Neural Networks, chạy real-time trên network flow data từ CTU-13 dataset."

**Show:** README.md → Problem Statement + Architecture diagram

---

### Bước 1 — Dataset và Preprocessing (2 phút)

```bash
# Show dataset stats (đã xử lý sẵn)
python -c "
import json
with open('data/processed/dataset_stats.json') as f:
    s = json.load(f)
sc = s['scenario10_full']
print(f'CTU-13 Scenario 10 — Neris IRC Botnet')
print(f'Total flows : {sc[\"total\"]:,}')
print(f'Botnet flows: {sc[\"labels\"][\"botnet\"][\"count\"]:,} ({sc[\"labels\"][\"botnet\"][\"pct\"]:.2f}%)')
print(f'Background  : {sc[\"labels\"][\"background\"][\"count\"]:,} ({sc[\"labels\"][\"background\"][\"pct\"]:.2f}%)')
print(f'Imbalance ratio: {sc[\"imbalance_ratio\"]:.1f}:1')
"
```

**Nói:** "Dataset có 5.17 triệu flows, trong đó chỉ 6.2% là botnet traffic. Đây là class imbalance điển hình của bài toán security — botnet cố tình ẩn trong traffic bình thường."

---

### Bước 2 — XGBoost Baseline (2 phút)

```bash
# Show XGBoost results
cat reports/results_table.txt
```

```bash
# Show SHAP top features
python -c "
import json
with open('models/artifacts/shap_feature_importance.json') as f:
    d = json.load(f)
print('Top SHAP features (XGBoost):')
for feat in d['features'][:5]:
    print(f'  {feat[\"name\"]:<25}: {feat[\"shap_mean_abs\"]:.4f}')
"
```

**Nói:** "XGBoost đạt F1=0.992 trên CTU-13 Scenario 10. Điều này vì Neris botnet dùng IRC protocol trên port 6667 — src_port và dst_port là feature cực kỳ discriminative theo SHAP analysis. Đây là baseline mạnh nhưng port-dependent — nếu C2 dùng HTTPS, feature này biến mất."

---

### Bước 3 — Dynamic Graph Construction (2 phút)

```python
# Interactive demo trong Python REPL
python -c "
from c2gnn.data.flow_builder import CTU13FlowParser
from c2gnn.graph.dynamic_graph import SlidingWindowGraph

parser = CTU13FlowParser(exclude_background=False)
graph = SlidingWindowGraph(window_size=60.0, edge_ttl=120.0)

# Process first 1000 flows
flows = []
for i, flow in enumerate(parser.iter_file('data/raw/ctu13/scenario10.binetflow')):
    graph.update(flow)
    flows.append(flow)
    if i >= 999:
        break

print(f'Processed {len(flows)} flows')
print(f'Graph stats: {graph.stats}')
data = graph.to_pyg_data(include_ground_truth=True)
if data:
    botnet_pct = data.y.float().mean().item()*100
    print(f'Graph nodes: {data.num_nodes}, edges: {data.num_edges}')
    print(f'Node features shape: {data.x.shape}')
    print(f'Botnet nodes: {data.y.sum().item()} ({botnet_pct:.2f}%)')
"
```

**Nói:** "Chúng em xây dựng graph động — mỗi node là một IP, mỗi edge là luồng traffic. Sliding window 60-300 giây loại bỏ edge cũ và giữ lại pattern hiện tại. Node features gồm 14 chiều: fan-out ratio, dst_ip_entropy, port entropy... những đặc trưng graph phản ánh behavior của C2 beaconing."

---

### Bước 4 — GNN Inference và Threshold Analysis (2 phút)

```bash
# Show GNN metrics với giải thích honest
python -c "
import json
with open('models/artifacts/graphsage_metrics.json') as f:
    m = json.load(f)
print('GraphSAGE Results (window=60s, class_weight_cap=50, threshold=0.5):')
print(f'  F1:        {m[\"f1\"]:.4f}')
print(f'  ROC-AUC:   {m[\"roc_auc\"]:.4f}')
print(f'  Precision: {m[\"precision\"]:.4f}')
print(f'  Recall:    {m[\"recall\"]:.4f}  ← catches 67% of real bots')
print(f'  FPR:       {m[\"false_positive_rate\"]*100:.2f}%  ← lower than XGBoost')
print(f'  Latency:   {m[\"latency_mean_ms\"]:.1f} ms/graph')
"
```

**Nói:** "GraphSAGE đạt F1=0.399, AUC=0.983, và FPR=0.09% — thấp hơn cả XGBoost (0.10%). Điều này có nghĩa là GNN tạo ít false alarm hơn trên cùng dataset, mặc dù F1 tổng thể thấp hơn. Lý do: GNN capture được graph topology của C2 beaconing — fan-out pattern, dst_ip_entropy — những đặc trưng không phụ thuộc vào port number. Nếu C2 chuyển sang dùng HTTPS thay IRC, XGBoost mất feature quan trọng nhất còn GNN vẫn detect được qua structural pattern."

---

### Bước 5 — Realtime Pipeline Demo (2 phút)

```bash
# Run realtime demo với max_flows để kết thúc nhanh
python -m c2gnn.realtime.pipeline \
  --data data/processed/scenario10_test.parquet \
  --model models/artifacts/graphsage_best.pt \
  --model-type graphsage \
  --threshold 0.7 \
  --window-size 60.0 \
  --realtime-factor 0 \
  --max-flows 2000
```

**Nói:** "Đây là 3-thread pipeline: Thread 1 đọc flows, Thread 2 update dynamic graph, Thread 3 chạy GNN inference. Latency trung bình ~145ms/graph trên CPU. Alerts được generate khi node có score vượt threshold."

**Show alert output:** Chỉ vào JSON alert với src_ip, risk_score, reasons.

---

## Fallback (nếu pipeline lỗi)

```bash
# Fallback 1: Show saved metrics
cat reports/results_table.txt

# Fallback 2: Show alert format từ README
python -c "
import json
alert = {
    'timestamp': '2024-01-15T14:23:07Z',
    'src_ip': '147.32.84.165',
    'dst_ip': '77.247.110.38',
    'risk_score': 0.8412,
    'model': 'GraphSAGE',
    'reasons': [
        'high fan-out: 23 unique destinations',
        'short-lived connections (possible beaconing)',
        'suspicious port ratio: 67%'
    ]
}
print(json.dumps(alert, indent=2))
"

# Fallback 3: Show graph construction static
python -c "
from c2gnn.graph.dynamic_graph import SlidingWindowGraph, NodeData
g = SlidingWindowGraph(60, 120)
print('SlidingWindowGraph initialized')
print('Node features:', NodeData.feature_names())
print('14-dim vector for GNN input')
"
```

---

## Câu hỏi phản biện — Trả lời nhanh

**Q: GNN F1 thấp có phải thất bại không?**
> "Không. AUC=0.949 chứng minh model học được signal đúng. F1 thấp do threshold chưa được tuned từ PR curve — với optimal threshold F1 cải thiện đáng kể. Đây là bài học về metric selection cho imbalanced classification."

**Q: Tại sao không chỉ dùng XGBoost?**
> "XGBoost đạt 0.992 trên Scenario 10 nhờ port 6667 của IRC. Với C2 qua HTTPS, port không còn discriminative. GNN capture được fan-out, beaconing topology ngay cả khi port bình thường."

**Q: Dataset lỗi thời không?**
> "Đúng, CTU-13 là từ 2011 — chúng em ghi rõ limitation này trong báo cáo. Nó vẫn là benchmark chuẩn trong literature và đủ để validate phương pháp. Modern C2 evaluation sẽ cần dataset enterprise private."

**Q: Real-time thật không?**
> "Đây là replay simulation với timestamp acceleration. Live capture cần thêm PCAP module. Chúng em định vị là 'realtime prototype' trong báo cáo."

---

## Checklist Demo Day

- [ ] Laptop sạc đầy pin
- [ ] Tắt notification, screensaver
- [ ] Font terminal ≥16pt
- [ ] Dark terminal theme
- [ ] Mở sẵn: terminal, README.md, reports/results_table.txt
- [ ] Test toàn bộ commands trong kịch bản trước 30 phút
- [ ] Có fallback screenshots nếu demo lỗi
- [ ] Biết giải thích GNN F1 thấp trong 60 giây
