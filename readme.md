# Thiết kế Pipeline Realtime — ET-SSL Inference trên Raspberry Pi

## 1. Mục tiêu thiết kế

Bài báo bạn tham chiếu (Sattar et al., *Anomaly detection in encrypted network traffic
using self-supervised learning*, Sci Rep 2025) đo được trên GPU RTX 3090: latency
15–25 ms/flow, throughput tới 10 Gbps, và **năng lượng 0.5 J/detection** — con số
năng lượng này là lý do các tác giả nói thẳng model "đủ nhẹ để chạy trên Raspberry Pi
hoặc NVIDIA Jetson Nano". Trên Pi, mình không có GPU nên **không thể đạt cùng
throughput**, nhưng kiến trúc 3 khối của họ (feature extraction → contrastive
embedding → khoảng cách tới centroid μ_norm) là thứ ánh xạ trực tiếp vào code bạn
đang có (`feature_extractor.py` → ONNX encoder → `ETSSLInference.predict_batch`).

Mục tiêu cụ thể cho bản thiết kế này:

1. Tối ưu cho CPU ARM yếu (Pi 4/5, không GPU) — không tối ưu cho 10 Gbps.
2. Giữ đúng ngữ nghĩa model: cùng `MODEL_FEATURES`, cùng công thức
   `is_anomaly = score > Δ` với `Δ = δ × κ`.
3. Batch hoá inference thay vì gọi ONNX từng flow một (chi phí cố định mỗi lần
   gọi runtime trên CPU yếu chiếm phần lớn latency ở batch=1).
4. Không chỉ in ra terminal — **ghi JSON có cấu trúc** cho từng quyết định + một
   file thống kê định kỳ (packets/sec trung bình, số flow/giây, phân phối anomaly).
5. Giữ được cơ chế incremental learning (`μ_norm` cập nhật dần) đúng như mô tả
   trong bài báo, nhưng cách ly để tránh poisoning (đã nói ở phần trước) bằng
   cách chỉ cập nhật từ flow có `score` thấp hơn hẳn ngưỡng, không phải mọi flow
   "not anomaly".

## 2. Kiến trúc tổng thể

```
 ┌─────────────┐   raw packets   ┌───────────────┐  packet events  ┌──────────────────┐
 │ Packet      │ ──────────────▶ │ Flow Tracker   │ ──────────────▶ │ Feature Extractor │
 │ Capture     │   (AF_PACKET /  │ (5-tuple table,│  flow closed /  │ (feature_schema + │
 │ (Scapy /    │    libpcap,     │  timeout 60s)  │  window expired │  feature_extractor│
 │  pypcap,    │    no payload   │                │                 │  .py — 20 feats)  │
 │  no decrypt)│    read)        │                │                 │                   │
 └─────────────┘                 └───────────────┘                 └─────────┬─────────┘
                                                                               │ feature_vector (20,)
                                                                               ▼
                                                                    ┌─────────────────────┐
                                                                    │ Micro-batch queue    │
                                                                    │ (max N=32 hoặc       │
                                                                    │  timeout 200ms)      │
                                                                    └──────────┬──────────┘
                                                                               │ X: (B,20) float32
                                                                               ▼
                                                                    ┌─────────────────────┐
                                                                    │ ONNX Runtime (CPU,   │
                                                                    │ INT8 quantized,      │
                                                                    │ intra_op_threads=4)  │
                                                                    │ engine.predict_batch │
                                                                    └──────────┬──────────┘
                                                                               │ scores (B,)
                                                                               ▼
                                                        ┌───────────────────────────────────────┐
                                                        │ Decision: is_anomaly = score > Δ       │
                                                        │ Δ = δ × κ  (effective_delta)            │
                                                        └───────────┬───────────┬────────────────┘
                                                                    │ normal    │ anomaly
                                                                    ▼           ▼
                                                    ┌───────────────────┐  ┌──────────────────┐
                                                    │ Incremental update │  │ _handle_anomaly() │
                                                    │ μ_norm (EMA, chỉ   │  │ → alert + log      │
                                                    │ áp dụng nếu score  │  └──────────────────┘
                                                    │ < 0.5·Δ, xem §5)   │
                                                    └───────────────────┘
                                                                    │
                                                                    ▼
                                                    ┌─────────────────────────────────────┐
                                                    │ Logging layer                        │
                                                    │  - terminal: 1 dòng/giây (throughput)│
                                                    │  - JSON Lines: 1 record/flow          │
                                                    │  - JSON summary: mỗi 60s              │
                                                    └─────────────────────────────────────┘
```

Điểm khác với thiết kế "gọi ONNX từng packet" mà nhiều demo hay làm: ở đây quyết
định luôn nằm ở cấp **flow** (giống đúng bài báo — feature là packet-length
distribution, inter-arrival time, flow duration, không phải per-packet), và
inference luôn được **gom batch** trước khi đưa vào ONNX Runtime, vì trên Pi chi
phí khởi tạo mỗi lần gọi run() chiếm tỷ trọng lớn so với chi phí tính toán thật khi
batch nhỏ.

## 3. Vì sao micro-batching là tối ưu hoá quan trọng nhất trên Pi

Trên GPU 10 Gbps, batch lớn tận dụng song song hoá phần cứng. Trên Pi (CPU
ARM Cortex-A76, không SIMD FP mạnh), vấn đề khác hẳn: **overhead cố định**
(memory alloc, session run(), Python↔C++ boundary) trên mỗi lần gọi ONNX chiếm
phần lớn nếu bạn gọi `predict_batch([x])` với B=1 hàng nghìn lần/giây.

Chiến lược: gom flow đã đóng (kết thúc timeout hoặc FIN/RST) vào một hàng đợi,
flush khi đạt N=32 flow **hoặc** đã quá 200ms kể từ record đầu tiên trong hàng
đợi (lấy giá trị nhỏ hơn để không vi phạm ràng buộc latency 15–25ms nêu trong
bài báo — thực tế trên Pi bạn nên chấp nhận ngưỡng nới ra ~150–300ms thay vì
15ms của GPU, và ghi rõ điều này trong README/kỳ vọng hiệu năng, đừng quảng cáo
số liệu GPU cho thiết bị CPU).

```python
class MicroBatcher:
    def __init__(self, max_batch=32, max_wait_ms=200):
        self.buf = []
        self.first_ts = None
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000

    def add(self, flow_id, feature_vec):
        if not self.buf:
            self.first_ts = time.monotonic()
        self.buf.append((flow_id, feature_vec))
        if len(self.buf) >= self.max_batch:
            return self._flush()
        return None

    def maybe_flush_on_timeout(self):
        if self.buf and (time.monotonic() - self.first_ts) >= self.max_wait:
            return self._flush()
        return None

    def _flush(self):
        batch = self.buf
        self.buf = []
        self.first_ts = None
        return batch
```

## 4. Tối ưu hoá riêng cho phần cứng hạn chế (Pi 4/5)

| Kỹ thuật | Lý do | Rủi ro cần cân nhắc |
|---|---|---|
| Quantize model ONNX sang INT8 (`onnxruntime.quantization`) | Giảm 2–4x thời gian inference trên CPU không có FP16/AVX | AUC có thể giảm nhẹ (~0.5–1 điểm %) — nên re-validate trên tập test trước khi triển khai |
| `sess_options.intra_op_num_threads = số lõi vật lý` (thường 4 trên Pi 4/5) | ONNX Runtime mặc định có thể spawn quá nhiều thread, gây context-switch overhead trên chip ít lõi | Đặt cả `inter_op_num_threads=1` vì đồ thị tính toán của bạn không phân nhánh song song |
| Bắt gói tin bằng AF_PACKET/libpcap ở chế độ **không copy payload**, chỉ đọc header + timestamp | Bạn chỉ cần 20 feature thống kê (timing/payload-size/protocol), không cần nội dung → giảm I/O và bộ nhớ đáng kể | Vẫn cần đọc đủ header để lấy độ dài gói, cờ TCP, cổng |
| Flow table dùng dict với TTL chủ động dọn (không phải quét toàn bộ mỗi lần) — dùng `heap` theo last_seen | Tránh CPU tăng tuyến tính theo số flow đang mở khi traffic lớn | Cần khóa (lock) nhẹ nếu capture thread và extractor thread khác nhau |
| Tách tiến trình: 1 process bắt gói (I/O-bound) + 1 process suy luận (CPU-bound) qua `multiprocessing.Queue` | Tránh GIL Python chặn vòng lặp bắt gói khi ONNX đang chạy | Thêm độ trễ serialize/deserialize giữa 2 tiến trình — chấp nhận được vì ta đã batch |
| Giới hạn cứng kích thước hàng đợi (drop-oldest khi quá tải) | Trên thiết bị yếu, traffic tăng đột biến (chính là lúc có DDoS) dễ làm nghẽn toàn hệ thống nếu không có giới hạn | Phải log số lượng flow bị drop — nếu không bạn "mù" đúng lúc cần phát hiện nhất |

## 5. Cơ chế incremental update an toàn hơn

Bài báo dùng EMA đơn giản: `μ_norm(t+1) = α·μ_norm(t) + (1-α)·mean(z_i normal)`.
Nhưng nếu áp dụng y nguyên "mọi flow không bị flag là anomaly", một cuộc tấn công
low-and-slow (đã nói ở câu trước) có thể từ từ kéo μ_norm về phía nó. Thiết kế ở
đây thêm một điều kiện: **chỉ những flow có `score < 0.5 × Δ` (rõ ràng bình
thường, không phải "cận biên") mới được dùng để cập nhật μ_norm**, và việc cập
nhật được thực hiện theo batch tích lũy (mỗi 60s) chứ không theo từng flow —
giúp bạn log lại toàn bộ batch cập nhật để audit sau này nếu nghi ngờ poisoning.

## 6. Logging — JSON thay vì chỉ print

### 6.1 File log theo từng flow (JSON Lines, append-only)

`logs/flow_decisions.jsonl` — mỗi dòng một object, dùng JSON Lines để có thể
tail -f / stream mà không cần parse cả file:

```json
{"ts": "2026-07-22T14:03:11.482Z", "flow_id": "192.168.1.5:51422-93.184.216.34:443-TCP", "score": 0.812, "delta_effective": 0.65, "is_anomaly": true, "top_features": {"iat_std": 0.91, "pkt_len_var": 0.77}, "packet_count": 143, "duration_s": 4.2}
```

`top_features` là 2–3 feature đóng góp lệch nhiều nhất so với μ_norm (tính bằng
`abs(z_i[j] - μ_norm[j])`), giúp bạn/người vận hành biết *vì sao* bị gắn cờ mà
không cần đọc payload.

### 6.2 File thống kê định kỳ (mỗi 60s, ghi đè hoặc rotate)

`logs/stats_summary.json`:

```json
{
  "window_start": "2026-07-22T14:03:00Z",
  "window_end": "2026-07-22T14:04:00Z",
  "packets_total": 18422,
  "packets_per_sec_avg": 307.03,
  "flows_closed": 96,
  "flows_per_sec_avg": 1.6,
  "anomaly_count": 3,
  "anomaly_rate": 0.03125,
  "batch_inference": {
    "batches_run": 5,
    "avg_batch_size": 19.2,
    "avg_latency_ms": 187.4,
    "p95_latency_ms": 241.0
  },
  "queue_drops": 0,
  "mu_norm_updates": 1
}
```

### 6.3 Terminal — chỉ dòng tổng hợp, không spam từng gói

```
[14:04:00] pkts/s=307.0 avg | flows/s=1.6 | batch_avg=19.2 | anomalies=3 (3.1%) | queue_drop=0
[14:04:00] ANOMALY flow=192.168.1.5:51422->93.184.216.34:443 score=0.812 (Δ=0.65)
```

Dòng thống kê in mỗi giây (rolling window packets/s), dòng ANOMALY in ngay khi
có flow bị flag — không in dòng "normal" cho từng flow ra terminal (quá nhiều),
nhưng **mọi flow, kể cả normal, đều được ghi vào JSON Lines** để không mất dữ
liệu phục vụ điều tra sau này.

```python
import json, time, threading

class Logger:
    def __init__(self, jsonl_path, summary_path):
        self.jsonl_path = jsonl_path
        self.summary_path = summary_path
        self._lock = threading.Lock()
        self._pkt_counter = 0
        self._flow_counter = 0
        self._anomaly_counter = 0
        self._window_start = time.time()

    def log_flow_decision(self, record: dict):
        with self._lock, open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._flow_counter += 1
        if record["is_anomaly"]:
            self._anomaly_counter += 1
            print(f"[{record['ts']}] ANOMALY flow={record['flow_id']} "
                  f"score={record['score']:.3f} (Δ={record['delta_effective']:.3f})")

    def on_packet(self):
        self._pkt_counter += 1

    def flush_summary(self, extra: dict):
        now = time.time()
        elapsed = now - self._window_start
        summary = {
            "window_start": self._window_start,
            "window_end": now,
            "packets_total": self._pkt_counter,
            "packets_per_sec_avg": round(self._pkt_counter / max(elapsed, 1e-6), 2),
            "flows_closed": self._flow_counter,
            "flows_per_sec_avg": round(self._flow_counter / max(elapsed, 1e-6), 2),
            "anomaly_count": self._anomaly_counter,
            "anomaly_rate": round(self._anomaly_counter / max(self._flow_counter, 1), 4),
            **extra,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] pkts/s={summary['packets_per_sec_avg']} avg | "
              f"flows/s={summary['flows_per_sec_avg']} | anomalies={self._anomaly_counter} "
              f"({summary['anomaly_rate']*100:.1f}%)")
        self._pkt_counter = 0
        self._flow_counter = 0
        self._anomaly_counter = 0
        self._window_start = now
```

## 7. Kỳ vọng hiệu năng thực tế trên Pi (đừng dùng số của bài báo)

Bài báo đo trên RTX 3090 (10 Gbps, 15–25ms). Trên Pi 4 (4×Cortex-A72, không
GPU), với model đã quantize INT8 và batch ~32:

- Inference/batch: ước lượng vài chục ms (phụ thuộc kích thước encoder thật —
  cần đo bằng `time.perf_counter()` quanh `predict_batch` sau khi bạn có model
  ONNX cụ thể; không đoán số cụ thể ở đây vì phụ thuộc kiến trúc encoder).
- Throughput hợp lý để nhắm tới: vài trăm flow/giây, không phải Gbps — phù hợp
  mạng gia đình/văn phòng nhỏ, không phù hợp làm IDS biên mạng lõi tốc độ cao.
- Bottleneck thực tế thường nằm ở **feature extraction + flow tracking**
  (thuần Python), không phải ONNX — nên cân nhắc viết `feature_extractor` bằng
  Cython/numpy vector hoá nếu đo thấy đây là điểm nghẽn, thay vì tối ưu sớm chỗ
  ONNX.

## 8. Việc cần làm tiếp theo trước khi code

1. Xác nhận `feature_schema.json` gốc (đã nói ở phần trước) khớp thứ tự với
   `MODEL_FEATURES` — nếu chưa xác nhận, mọi số liệu hiệu năng ở trên vô nghĩa
   vì model có thể đang predict sai âm thầm.
2. Đo thời gian `predict_batch` thật trên Pi với batch=1 vs batch=32 để chọn
   `max_batch`/`max_wait_ms` phù hợp thay vì dùng số gợi ý ở trên.
3. Quyết định chính sách khi hàng đợi đầy lúc bị DDoS thật (drop-oldest vs
   drop-newest vs tăng `max_batch` tạm thời) — đây là quyết định thiết kế có
   đánh đổi, không có câu trả lời đúng tuyệt đối.