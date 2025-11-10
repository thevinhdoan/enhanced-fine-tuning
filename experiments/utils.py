import logging
from typing import Dict, List, Optional
from pathlib import Path

import mlflow


class MLflowTracker:
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)

    def log_params(self, params: Dict[str, object]) -> None:
        serialized = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = str(value)
        if serialized:
            mlflow.log_params(serialized)

    def log_dict(self, data: Dict, artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)


def flush_logging_handlers() -> None:
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "flush"):
            handler.flush()
