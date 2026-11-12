"""Get comparable scores out of estimators that expose different interfaces.

Three of the four estimators provide ``predict_proba``; the SVM deliberately
does not, because on this dataset its calibrated probabilities invert (see the
note in :mod:`SmartAMR.reproduce.train`). Every page that ranks something has
to cope with both, and doing it inline meant the SVM silently produced no
ranking at all. These two helpers are the single place that knows the rule.

A margin is squashed with a logistic so it can share an axis with a
probability. The squash is monotonic, so rankings are unaffected; the resulting
number is a score to compare *within* one estimator, not a calibrated
probability, and the pages label it "score" rather than "probability".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _logistic(values: np.ndarray) -> np.ndarray:
    """Squash a real-valued margin into (0, 1), overflow-safe."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def positive_scores(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    """Score for the positive class of a binary estimator, one per row."""
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features), dtype=float)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return _logistic(np.asarray(pipeline.decision_function(features), dtype=float))
    return np.asarray(pipeline.predict(features), dtype=float)


def class_scores(pipeline: Any, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """``(scores, classes)`` for a multiclass estimator.

    ``scores`` has one column per class, in the order of ``classes``. With
    ``predict_proba`` the columns are probabilities and sum to 1; from
    ``decision_function`` they are squashed one-vs-rest margins, which rank
    correctly but do not sum to 1.
    """

    classes = [str(label) for label in pipeline.classes_]

    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features), dtype=float), classes

    if hasattr(pipeline, "decision_function"):
        margins = np.asarray(pipeline.decision_function(features), dtype=float)
        if margins.ndim == 1:
            # A two-class decision_function is a single column: the margin is
            # towards classes[1], so mirror it for classes[0].
            margins = np.column_stack([-margins, margins])
        return _logistic(margins), classes

    # Last resort: a one-hot of the hard prediction, so a ranking still exists.
    predicted = pipeline.predict(features)
    index = {label: position for position, label in enumerate(classes)}
    scores = np.zeros((len(predicted), len(classes)), dtype=float)
    for row, label in enumerate(predicted):
        scores[row, index[str(label)]] = 1.0
    return scores, classes


def score_is_probability(pipeline: Any) -> bool:
    """True when this estimator's scores are calibrated probabilities."""
    return hasattr(pipeline, "predict_proba")
