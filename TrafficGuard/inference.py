"""
ml/inference.py
==================
Wraps a loaded model with a stable predict() interface returning:

    {
        "attack_label": "PortScan",
        "confidence": 0.91
    }

If no model is loaded (missing file, corrupt artifact, etc.) this
degrades gracefully to a "Benign"/0.0 passthrough rather than
crashing the worker process.
"""

import logging

from features.feature_schema import as_vector

logger = logging.getLogger("nids.ml")


class MLInference:
    def __init__(self, model):
        self.model = model

    def predict(self, feature: dict) -> dict:
        if self.model is None:
            return {"attack_label": "Benign", "confidence": 0.0}

        vector = [as_vector(feature)]

        try:
            label = self.model.predict(vector)[0]
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(vector)[0]
                confidence = float(max(proba))
            return {"attack_label": str(label), "confidence": round(confidence, 4)}
        except Exception:
            logger.exception("ML inference failed, defaulting to Benign")
            return {"attack_label": "Benign", "confidence": 0.0}