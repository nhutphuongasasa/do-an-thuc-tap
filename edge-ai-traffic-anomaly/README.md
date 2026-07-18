# ET-SSL Edge AI Traffic Anomaly Detection

Triển khai pipeline phát hiện bất thường lưu lượng mã hóa theo thiết kế trong [`../readme.md`](../readme.md).

## Flow vận hành (readme §0)

```
Traffic → Capture → Flow Aggregator → Feature Extractor → Inference → Alert → Incremental μ_norm → Dashboard
```

## Cấu trúc thư mục

```
edge-ai-traffic-anomaly/
├── configs/config.yaml       # δ, κ, α, đường dẫn model
├── data/                     # preprocess, feature schema
├── model/                    # encoder, inference, weights/
├── pipeline/
│   ├── capture.py            # Scapy live/pcap
│   ├── flow_aggregator.py    # 5-tuple + time-window
│   ├── feature_extractor.py  # PL, IPI, FD, PC, PM → 20 features
│   ├── inference.py          # re-export ETSSLInference
│   ├── inference_runner.py   # inference + alert + incremental
│   ├── alert_manager.py      # logs/alerts.jsonl
│   └── incremental_learner.py
├── optimization/             # benchmark, compare FP32/INT8/ONNX
├── evaluation/               # metrics, zero-day, drift, robustness
├── dashboard/app.py          # Streamlit
└── run_all.py                # evaluation suite
```

## Chạy nhanh

```bash
# Từ thư mục gốc repo
./start.sh              # demo stream + dashboard (Live Feed)
./start.sh test         # evaluation quick test
./start.sh pipeline     # full evaluation
./start.sh capture --pcap file.pcap
```

## Pipeline từng module

```bash
cd edge-ai-traffic-anomaly

# Demo traffic qua full pipeline (ghi logs/pipeline_state.json)
python pipeline/demo_stream.py --flows 200

# PCAP replay
python pipeline/capture.py --pcap data/raw/sample.pcap

# Dashboard
streamlit run dashboard/app.py
# → chọn "Live Feed" nếu chạy demo/capture riêng
```

## Dataset thật

1. Tải CSV vào `data/raw/unsw-nb15/` hoặc `data/raw/cic-darknet2020/`
2. Preprocess: `python data/preprocess.py --dataset unsw_nb15 --data_path data/raw/unsw-nb15/`
3. Evaluate: `python run_all.py --dataset unsw_nb15`

## Ghi chú

- **Không giải mã payload** — chỉ metadata gói tin (timing, length, TCP flags)
- Model ET-SSL đã train sẵn trong `model/weights/` — không có code contrastive training
- Ngưỡng thực tế: `δ × κ` (cấu hình trong `configs/config.yaml`)
