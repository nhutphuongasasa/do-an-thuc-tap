"""
features/feature_schema.py
=============================
Canonical schema for the FeatureVector passed to the ML Worker Pool.
Keeping this centralized ensures the feature extractor and the ML
model loader/inference code agree on field order and names.
"""

FEATURE_FIELDS = [
    "duration",
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "std_packet_size",
    "tcp_flags_diversity",
    "entropy",
    "avg_iat",
    "std_iat",
]


def as_vector(feature_dict: dict) -> list:
    """Return features as an ordered numeric list, matching FEATURE_FIELDS,
    for feeding directly into a scikit-learn model."""
    return [float(feature_dict.get(f, 0.0)) for f in FEATURE_FIELDS]