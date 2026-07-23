# Danh sách 20 đặc trưng — thứ tự này phải khớp chính xác với model đã train.
# KHÔNG đổi tên, KHÔNG đổi thứ tự, kể cả khi refactor.
MODEL_FEATURES = [
    "duration",
    "packet_count",
    "byte_count",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_byte_count",
    "bwd_byte_count",
    "avg_packet_size",
    "avg_fwd_packet_size",
    "avg_bwd_packet_size",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_header_len",
    "bwd_header_len",
    "syn_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "ack_flag_count",
]
