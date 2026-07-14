"""
workers/ml_worker.py
=======================
Each MLWorker is a separate OS process (multiprocessing.Process), as
required by the spec: "Không dùng một model duy nhất / mỗi worker load
model riêng." Running as processes (not threads) sidesteps the GIL so
CPU-bound model inference actually scales across cores.

    Feature Queue
         |
    +----+----+----+
    |    |    |
   ML1  ML2  ML3      <- each loads its OWN model file (round-robin
                          assignment from ML_MODEL_PATHS)

Results are pushed onto the MLResultQueue for the Correlation Worker
running back in the main process.
"""

import logging
import multiprocessing
import time

logger = logging.getLogger("nids.worker.ml")


def _ml_worker_main(
    worker_id: int,
    model_path: str,
    feature_queue,
    ml_result_queue,
    stop_event,
    get_timeout: float,
    put_timeout: float,
    log_level: str,
):
    """Entry point run inside the child process. Must be a top-level
    function (not a bound method) so it's picklable for
    multiprocessing.Process(target=...)."""
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [ML-{worker_id}] %(levelname)s %(message)s",
    )
    log = logging.getLogger(f"nids.worker.ml.{worker_id}")

    # Imports done inside the child process to avoid pickling issues
    # with native/C-extension objects (e.g. loaded sklearn models)
    # across the process fork/spawn boundary.
    from ml.model_loader import ModelLoader
    from ml.inference import MLInference

    model = ModelLoader.load(model_path)
    inference = MLInference(model)

    log.info("ML worker %d ready (model=%s)", worker_id, model_path)

    while not stop_event.is_set():
        feature = feature_queue.get(timeout=get_timeout)
        if feature is None:
            continue
        try:
            result = inference.predict(feature)
            result["_meta"] = feature.get("_meta", {})
            result["worker_id"] = worker_id
            ml_result_queue.put(result, timeout=put_timeout)
        except Exception:
            log.exception("ML worker %d failed on a feature vector", worker_id)

    log.info("ML worker %d shutting down", worker_id)


class MLWorker:
    def __init__(
        self,
        worker_id: int,
        model_path: str,
        feature_queue,
        ml_result_queue,
        get_timeout: float = 0.5,
        put_timeout: float = 0.1,
        log_level: str = "INFO",
    ):
        self.worker_id = worker_id
        self.model_path = model_path
        self.feature_queue = feature_queue
        self.ml_result_queue = ml_result_queue
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout
        self.log_level = log_level

        self._stop_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_ml_worker_main,
            args=(
                worker_id,
                model_path,
                feature_queue,
                ml_result_queue,
                self._stop_event,
                get_timeout,
                put_timeout,
                log_level,
            ),
            name=f"MLWorker-{worker_id}",
            daemon=True,
        )

    def load_model(self):
        # Loading happens inside the child process (see _ml_worker_main)
        # so the model object never needs to cross a process boundary.
        pass

    def predict(self, feature: dict) -> dict:
        raise NotImplementedError(
            "MLWorker.predict() runs inside the child process loop; "
            "use ml.inference.MLInference directly for synchronous/test use."
        )

    def start(self):
        self._process.start()
        logger.info("MLWorker-%d process started (pid=%s)", self.worker_id, self._process.pid)

    def stop(self):
        self._stop_event.set()
        self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning("MLWorker-%d did not exit cleanly, terminating", self.worker_id)
            self._process.terminate()

    def is_alive(self) -> bool:
        return self._process.is_alive()