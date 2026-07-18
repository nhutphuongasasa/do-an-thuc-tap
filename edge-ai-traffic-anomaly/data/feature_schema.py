"""
feature_schema.py — Mapping dataset column names to model feature names.

Model V5 expects 20 features in the order defined in config.yaml.
Different datasets (UNSW-NB15, CIC-Darknet2020) use different column names
→ this file maps them to the standard names.

To add a new dataset: add an entry to DATASET_REGISTRY.
"""

from typing import Optional

# =====================================================================
# Standard feature order — must match config.yaml exactly
# =====================================================================
MODEL_FEATURES = [
    "median_packets_delta_time",       # Median time between packets
    "packets_IAT_mean",                # Mean Inter-Arrival Time for the flow
    "fwd_init_win_bytes",              # TCP init window size forward
    "fwd_packets_IAT_mean",            # Mean IAT forward
    "bwd_init_win_bytes",              # TCP init window size backward
    "rst_flag_percentage_in_total",    # % packets with RST flag
    "mode_packets_delta_time",         # Mode delta time
    "bwd_packets_IAT_mean",            # Mean IAT backward
    "rst_flag_counts",                 # Count of packets with RST flag
    "psh_flag_percentage_in_total",    # % packets with PSH flag
    "fwd_min_header_bytes",            # Min header size forward
    "fwd_segment_size_min",            # Min segment size forward
    "skewness_payload_bytes_delta_len",# Skewness of delta payload length
    "payload_bytes_median",            # Median payload bytes
    "median_bwd_packets_delta_len",    # Median delta length backward
    "fwd_payload_bytes_median",        # Median payload bytes forward
    "mean_packets_delta_time",         # Mean delta time
    "cov_bwd_payload_bytes_delta_len", # Coefficient of variation bwd payload
    "mean_packets_delta_len",          # Mean delta packet length
    "ack_flag_percentage_in_total",    # % packets with ACK flag
]

assert len(MODEL_FEATURES) == 20, "Must have exactly 20 features!"


# =====================================================================
# UNSW-NB15 column mapping
# =====================================================================
# UNSW-NB15 has 49 features. Mapping to standard model names.
# The closest equivalent columns are selected.
# Ref: https://research.unsw.edu.au/projects/unsw-nb15-dataset
UNSW_NB15_MAP = {
    # model_feature_name              → column_name_in_dataset
    "median_packets_delta_time":       "Sintpkt",    # Std dev inter-packet time src (used as proxy)
    "packets_IAT_mean":                "sintpkt",    # Mean inter-packet time
    "fwd_init_win_bytes":              "Sintpkt",    # Not directly available → proxy
    "fwd_packets_IAT_mean":            "sintpkt",
    "bwd_init_win_bytes":              "dintpkt",    # dst inter-packet time
    "rst_flag_percentage_in_total":    "srst",       # Source RST count (normalize later)
    "mode_packets_delta_time":         "Sintpkt",
    "bwd_packets_IAT_mean":            "dintpkt",
    "rst_flag_counts":                 "srst",
    "psh_flag_percentage_in_total":    "spkts",      # Proxy (no direct PSH)
    "fwd_min_header_bytes":            "Sload",      # Proxy
    "fwd_segment_size_min":            "Sload",
    "skewness_payload_bytes_delta_len":"Sjit",       # Source jitter as proxy for skewness
    "payload_bytes_median":            "Sbytes",     # Source bytes
    "median_bwd_packets_delta_len":    "Dbytes",     # Dst bytes
    "fwd_payload_bytes_median":        "Sbytes",
    "mean_packets_delta_time":         "sintpkt",
    "cov_bwd_payload_bytes_delta_len": "Djit",       # Dst jitter
    "mean_packets_delta_len":          "Sload",
    "ack_flag_percentage_in_total":    "ackdat",     # ACK data packets
}

UNSW_NB15_LABEL_COL = "label"          # 0=normal, 1=attack
UNSW_NB15_NORMAL_VAL = 0

# Alternative label column names (for different versions)
UNSW_NB15_LABEL_ALTS = ["label", "Label", "attack_cat", "class"]


# =====================================================================
# CIC-Darknet2020 column mapping
# =====================================================================
# CIC-Darknet2020 exported from CICFlowMeter — clearer IAT/flag features
# Ref: https://www.kaggle.com/datasets/dhoogla/cicdarknet2020
CIC_DARKNET2020_MAP = {
    "median_packets_delta_time":       "Flow IAT Mean",  # Closest proxy
    "packets_IAT_mean":                "Flow IAT Mean",
    "fwd_init_win_bytes":              "Init_Win_bytes_forward",
    "fwd_packets_IAT_mean":            "Fwd IAT Mean",
    "bwd_init_win_bytes":              "Init_Win_bytes_backward",
    "rst_flag_percentage_in_total":    "RST Flag Count",   # needs normalization
    "mode_packets_delta_time":         "Flow IAT Std",     # proxy
    "bwd_packets_IAT_mean":            "Bwd IAT Mean",
    "rst_flag_counts":                 "RST Flag Count",
    "psh_flag_percentage_in_total":    "PSH Flag Count",   # needs normalization
    "fwd_min_header_bytes":            "min_seg_size_forward",
    "fwd_segment_size_min":            "min_seg_size_forward",
    "skewness_payload_bytes_delta_len":"Packet Length Std",  # proxy
    "payload_bytes_median":            "Avg Fwd Segment Size",
    "median_bwd_packets_delta_len":    "Avg Bwd Segment Size",
    "fwd_payload_bytes_median":        "Avg Fwd Segment Size",
    "mean_packets_delta_time":         "Flow IAT Mean",
    "cov_bwd_payload_bytes_delta_len": "Bwd Packet Length Std",
    "mean_packets_delta_len":          "Avg Packet Size",
    "ack_flag_percentage_in_total":    "ACK Flag Count",    # needs normalization
}

CIC_DARKNET2020_LABEL_COL = "Label"
CIC_DARKNET2020_NORMAL_VAL = "BENIGN"

# Attack label values
CIC_DARKNET2020_ATTACK_VALS = [
    "Darknet", "VPN", "Tor", "Non-Tor", "Non-VPN"
]


# =====================================================================
# Registry
# =====================================================================
DATASET_REGISTRY = {
    "unsw_nb15": {
        "column_map": UNSW_NB15_MAP,
        "label_col": UNSW_NB15_LABEL_COL,
        "normal_val": UNSW_NB15_NORMAL_VAL,
        "label_alts": UNSW_NB15_LABEL_ALTS,
        "description": "UNSW-NB15: 49 features, binary label (0=normal, 1=attack)",
        "proxy_features": True,  # Some features are proxy mappings, not exact
    },
    "cic_darknet2020": {
        "column_map": CIC_DARKNET2020_MAP,
        "label_col": CIC_DARKNET2020_LABEL_COL,
        "normal_val": CIC_DARKNET2020_NORMAL_VAL,
        "description": "CIC-Darknet2020: CICFlowMeter export, BENIGN vs attack",
        "proxy_features": False,  # Direct IAT/flag features exist
    },
}


def get_schema(dataset_name: str) -> dict:
    """
    Get schema for a dataset.

    Args:
        dataset_name: "unsw_nb15" | "cic_darknet2020"

    Returns:
        dict containing column_map, label_col, normal_val
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


def map_columns(df, dataset_name: str, verbose: bool = True):
    """
    Map dataset columns to standard model feature names.

    Args:
        df: pandas DataFrame of original dataset
        dataset_name: dataset name
        verbose: print warnings if columns are not found

    Returns:
        DataFrame with 20 standard columns (in MODEL_FEATURES order)
    """
    import pandas as pd

    schema = get_schema(dataset_name)
    col_map = schema["column_map"]

    if schema.get("proxy_features") and verbose:
        print(f"⚠️  [{dataset_name}] Some features use proxy mapping — not 100% accurate")

    result = {}
    missing_cols = []

    for model_feat, dataset_col in col_map.items():
        if dataset_col in df.columns:
            result[model_feat] = df[dataset_col].values
        else:
            # Try case-insensitive matching
            lower_cols = {c.lower(): c for c in df.columns}
            if dataset_col.lower() in lower_cols:
                actual = lower_cols[dataset_col.lower()]
                result[model_feat] = df[actual].values
                if verbose:
                    print(f"  [{model_feat}] Found as '{actual}' (case-insensitive)")
            else:
                missing_cols.append((model_feat, dataset_col))
                result[model_feat] = 0.0  # Fill with 0 if not found

    if missing_cols and verbose:
        print(f"⚠️  Missing columns ({len(missing_cols)}):")
        for mf, dc in missing_cols:
            print(f"    {mf} ← '{dc}' not found in dataset")

    # Return DataFrame with exactly 20 columns in the correct order
    out_df = pd.DataFrame(result)[MODEL_FEATURES]
    return out_df


def detect_label_column(df, dataset_name: Optional[str] = None) -> Optional[str]:
    """
    Automatically detect the label column in the DataFrame.
    """
    candidates = []
    if dataset_name and dataset_name in DATASET_REGISTRY:
        schema = DATASET_REGISTRY[dataset_name]
        candidates = [schema["label_col"]] + schema.get("label_alts", [])

    # Add common label names
    candidates += ["label", "Label", "class", "Class", "attack_cat", "Category"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


if __name__ == "__main__":
    print("📋 Feature Schema Info")
    print(f"  Model expects {len(MODEL_FEATURES)} features:")
    for i, f in enumerate(MODEL_FEATURES):
        print(f"    [{i:02d}] {f}")

    print("\n📊 Available datasets:")
    for name, info in DATASET_REGISTRY.items():
        print(f"  {name}: {info['description']}")
