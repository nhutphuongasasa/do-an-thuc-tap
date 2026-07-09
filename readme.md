# NIDS Realtime Pipeline

Hệ thống phát hiện tấn công mạng theo thời gian thực (Network Intrusion Detection System), sử dụng model Machine Learning (Random Forest) đã được train sẵn từ dataset CICIDS2017.

Hệ thống gồm **4 Flow (luồng xử lý) chạy song song bằng nhiều thread**, giao tiếp với nhau qua các hàng đợi (Queue), giống một dây chuyền sản xuất — mỗi trạm xử lý một việc rồi chuyền sang trạm kế tiếp.

## Sơ đồ tổng quát

```
Card mạng → FLOW1 → Queue1 → Aggregator → FlowBuffer (kho tạm)
                                              ↓
                              FLOW2 (3 thread: 1s/3s/5s) đọc từ FlowBuffer
                                              ↓
                                          Queue2
                                              ↓
                                          FLOW3 (Inference)
                                              ↓
                                       Alert Queue
                                              ↓
                                          FLOW4 (Alert)
                                              ↓
                                CSV log + Telegram + console
```

## Mô tả chi tiết từng thành phần

### FLOW 1 — Bắt gói tin (`capture.py`)

Giống một người đứng "chặn" ở cổng mạng, nhìn từng gói tin đi qua.

- Dùng thư viện `scapy` để "nghe" mọi gói tin IP đi qua card mạng.
- Với mỗi gói tin bắt được:
  - Xác định nó thuộc luồng kết nối nào (flow) — dựa vào IP nguồn/đích + port (giống như biết "ai đang nói chuyện với ai").
  - Xác định gói này là "bên hỏi" (fwd) hay "bên trả lời" (bwd).
  - Đóng gói thông tin (thời gian, độ dài, cờ TCP...) thành một "sự kiện" (`PacketEvent`).
  - Đẩy sự kiện đó vào **Queue1** — giống như đưa vào "băng chuyền" để trạm sau xử lý tiếp.
- Nếu băng chuyền (Queue1) đầy quá → gói tin bị rớt (drop), có đếm số lượng để theo dõi.

### Aggregator (nằm trong `main.py`)

Người lấy gói tin từ Queue1, xếp vào "kho tạm" theo từng luồng kết nối.

- Lấy từng `PacketEvent` ra khỏi Queue1.
- Đưa vào `FlowBuffer` — nơi lưu tạm các gói tin, có tự động xóa gói tin quá cũ để không bị tràn bộ nhớ.

### FLOW 2 — Tính đặc trưng (`feature_extractor.py`)

Giống hệt CICFlowMeter (công cụ đã tạo ra dataset CICIDS2017 lúc train model) — nhưng làm real-time.

- Có 3 "cửa sổ thời gian" chạy song song: 1 giây, 3 giây, 5 giây — mỗi cửa sổ là một thread riêng.
- Mỗi giây, với mỗi luồng kết nối đang "sống", nó lấy các gói tin trong X giây gần nhất, rồi tính ra các đặc trưng giống lúc train model (Flow Duration, Flow Bytes/s, SYN Flag Count, Active/Idle...).
- Kết quả đóng thành `FeatureVector`, đẩy vào **Queue2**.

> Lý do có 3 cửa sổ khác nhau: tấn công nhanh (VD SYN Flood) thì cửa sổ 1s bắt nhạy hơn; tấn công chậm, kéo dài (VD port scan chậm) thì cửa sổ 5s nhìn rõ pattern hơn.

### FLOW 3 — Dự đoán bằng Model (`inference_engine.py`)

Đây là nơi dùng lại model đã train (`rf_final_model.pkl`) để đoán.

- Load lại model, scaler, label encoder, danh sách feature đã lưu lúc train.
- Với mỗi `FeatureVector` lấy từ Queue2:
  - Sắp xếp đúng thứ tự feature như lúc train, xử lý inf/nan giống lúc train.
  - Đưa qua scaler (chuẩn hóa số liệu — giống bước RobustScaler lúc train).
  - Đưa vào model, ra được nhãn dự đoán (BENIGN/DDoS/PortScan...) và độ tin cậy (confidence).
- Vì có 3 cửa sổ (1s/3s/5s) dự đoán riêng cho cùng 1 luồng kết nối → cần "bỏ phiếu" (vote) giữa 3 kết quả để ra quyết định cuối cùng (ưu tiên cửa sổ dài hơn vì đáng tin hơn với tấn công kéo dài — `WINDOW_WEIGHTS`).
- Nếu là tấn công nhưng độ tin cậy quá thấp → bỏ qua, tránh báo động giả.
- Kết quả cuối đẩy vào **Alert Queue**.

### FLOW 4 — Cảnh báo (`alerting.py`)

Người "gác cổng cuối", nhận kết quả và hành động.

- Lấy từng `Prediction` từ Alert Queue.
- Ghi vào file CSV (log lại mọi kết quả, cả BENIGN và tấn công, để lưu vết).
- Nếu là tấn công thật:
  - Ghi log cảnh báo (warning).
  - Gửi tin nhắn Telegram báo cho người quản trị — nhưng có "cooldown" (giữ khoảng cách thời gian) để tránh spam nếu 1 luồng liên tục bị đoán là tấn công.

### `main.py` — nơi khởi động và nối tất cả lại

- Tạo 3 hàng đợi (Queue1, Queue2, Alert Queue) — giống 3 đoạn băng chuyền nối các trạm.
- Khởi động lần lượt: Aggregator → Flow2 (3 worker) → Flow3 → Flow4 → cuối cùng mới Flow1 (vì phải có "người tiêu thụ" sẵn sàng trước khi bắt đầu "sản xuất" gói tin, tránh mất dữ liệu).
- Có xử lý khi nhận Ctrl+C (SIGINT) → dừng an toàn từng flow theo thứ tự.
- Mỗi 10 giây in ra thống kê: số gói tin bắt được, số bị rớt, độ đầy các hàng đợi, số luồng đang hoạt động...