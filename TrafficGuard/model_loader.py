"""
ml/model_loader.py
=====================
Loads a pre-trained scikit-learn-compatible model from disk via
joblib. Each ML worker process calls this once at startup and keeps
the model resident in its own process memory (no cross-process
sharing needed since inference is stateless per-request).
"""

import logging
import os

import joblib

logger = logging.getLogger("nids.ml")


class ModelLoader:
    @staticmethod
    def load(model_path: str):
        if not os.path.exists(model_path):
            logger.warning(
                "Model file not found at %s — worker will run in "
                "passthrough mode (no predictions).", model_path
            )
            return None
        try:
            model = joblib.load(model_path)
            logger.info("Loaded ML model from %s", model_path)
            return model
        except Exception:
            logger.exception("Failed to load ML model from %s", model_path)
            return None