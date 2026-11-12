"""Training, evaluation and prediction of the phage-host classifiers."""

from __future__ import annotations

from SmartAMR.models.classifiers import (
    BINARY,
    MODEL_NAMES,
    MULTILABEL,
    TASKS,
    TrainedModel,
    build_classifier,
    evaluate,
    load_model,
    save_model,
    train,
)
from SmartAMR.models.predict import (
    PLACEHOLDER_WARNING,
    Prediction,
    evaluate_sequences,
    predict_pair,
    predictions_to_frame,
    screen_database,
)

__all__ = [
    "BINARY",
    "MODEL_NAMES",
    "MULTILABEL",
    "PLACEHOLDER_WARNING",
    "Prediction",
    "TASKS",
    "TrainedModel",
    "build_classifier",
    "evaluate",
    "evaluate_sequences",
    "load_model",
    "predict_pair",
    "predictions_to_frame",
    "save_model",
    "screen_database",
    "train",
]
