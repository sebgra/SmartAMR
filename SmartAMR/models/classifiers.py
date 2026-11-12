"""Train and evaluate the phage-host classifiers.

Logic lifted from ``notebooks/binary_classification.ipynb`` (does this phage
infect this bacteria?) and ``notebooks/multilabel_classification.ipynb`` (which
bacteria does this phage infect?). The estimators, their hyper-parameters and
the train/test split are unchanged; they are now reachable from one function so
the CLI and the dashboard share them.

scikit-learn and xgboost are imported lazily, so importing this module stays
cheap and the CLI runs without them installed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from SmartAMR.config import PHAGE_EMBEDDING_SIZE
from SmartAMR.embedding.features import balance_features, embedding_columns

BINARY = "binary"
MULTILABEL = "multilabel"
TASKS = (BINARY, MULTILABEL)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


def _estimator_factories() -> Dict[str, Callable[[str], Any]]:
    """Estimator constructors, keyed by the name used on the command line.

    Each factory takes the task so the few hyper-parameters that differed
    between the two notebooks stay faithful to the original.
    """

    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.svm import SVC

    def random_forest(task: str) -> Any:
        return RandomForestClassifier(max_depth=5, random_state=0)

    def svm(task: str) -> Any:
        return SVC(gamma="auto", kernel="linear") if task == MULTILABEL else SVC(gamma="auto")

    def adaboost(task: str) -> Any:
        n_estimators = 100 if task == MULTILABEL else 1000
        return AdaBoostClassifier(n_estimators=n_estimators, random_state=DEFAULT_SEED)

    def xgboost(task: str) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost is not installed. Add it with: pip install -e \".[boost]\" "
                "(or: uv sync --extra boost)"
            ) from error

        return XGBClassifier(max_depth=15, random_state=DEFAULT_SEED)

    return {
        "random-forest": random_forest,
        "svm": svm,
        "xgboost": xgboost,
        "adaboost": adaboost,
    }


#: Names accepted by ``--model``.
MODEL_NAMES: Tuple[str, ...] = ("random-forest", "svm", "xgboost", "adaboost")


def build_classifier(name: str, task: str = BINARY) -> Any:
    """Build a ``StandardScaler`` + estimator pipeline."""

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    factories = _estimator_factories()
    if name not in factories:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(factories)}")

    return make_pipeline(StandardScaler(), factories[name](task))


@dataclass
class TrainedModel:
    """A fitted pipeline together with what is needed to reuse it."""

    pipeline: Any
    task: str
    model_name: str
    feature_columns: List[str]
    classes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict labels, decoding class names for the multilabel task."""
        predictions = self.pipeline.predict(X)
        if self.task == MULTILABEL and self.classes and np.issubdtype(np.asarray(predictions).dtype, np.number):
            return np.array([self.classes[int(index)] for index in predictions])
        return predictions

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        """Class probabilities when the estimator exposes them, else None."""
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        try:
            return self.pipeline.predict_proba(X)
        except (AttributeError, NotImplementedError):  # pragma: no cover
            return None


def prepare_binary(
    features: pd.DataFrame,
    balance: bool = True,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Features and labels for the binary interaction task."""

    data = features.copy()
    data["interaction"] = data["label"].apply(lambda value: 1 if value else 0)

    if balance:
        data = balance_features(data, seed=seed)

    X = data[embedding_columns(data)]
    y = data["interaction"]

    return X, y


def prepare_multilabel(
    features: pd.DataFrame,
    phage_embedding_size: int = PHAGE_EMBEDDING_SIZE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Features and labels for the host identification task.

    Only the interacting pairs are used, and only the phage half of the
    embedding is given to the model, since the bacteria half would leak the
    answer.
    """

    interacting = features[features["label"].astype(bool)]
    if interacting.empty:
        raise ValueError("No interacting pair in the feature table - cannot train the multilabel model")

    all_columns = embedding_columns(interacting)
    # The feature vector is the phage embedding followed by the bacteria one.
    # PHAGE_EMBEDDING_SIZE describes the full-size ESM-C table; on a narrower
    # table (a subset, or a test fixture) fall back to splitting in half, so the
    # phage half stays the phage half and predictions line up with training.
    width = phage_embedding_size if len(all_columns) > phage_embedding_size else len(all_columns) // 2
    if width == 0:
        raise ValueError("The feature table holds no embedding column")

    X = interacting[all_columns[:width]]
    y = interacting["bacteria_id"]

    return X, y


def train(
    features: pd.DataFrame,
    model_name: str = "random-forest",
    task: str = BINARY,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    balance: bool = True,
) -> TrainedModel:
    """Fit one classifier and score it on a held-out split.

    Parameters
    ----------
    features
        The flat feature table from ``smartamr build-features``.
    model_name
        One of :data:`MODEL_NAMES`.
    task
        ``"binary"`` for phage-bacteria interaction, ``"multilabel"`` for host
        identification.
    test_size
        Fraction held out for scoring.
    seed
        Random seed of the split.
    balance
        Downsample the majority class (binary task only).

    Returns
    -------
    TrainedModel
        The fitted pipeline and its metrics.
    """

    from sklearn.model_selection import train_test_split

    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {', '.join(TASKS)}")

    if task == BINARY:
        X, y = prepare_binary(features, balance=balance, seed=seed)
        stratify = None
    else:
        X, y = prepare_multilabel(features)
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    classes: List[str] = []
    y_train_fit, y_test_eval = y_train, y_test

    # XGBoost needs numeric targets; the class names are kept to decode later.
    if task == MULTILABEL:
        classes = sorted(y.unique().tolist())
        if model_name == "xgboost":
            class_to_number = {name: index for index, name in enumerate(classes)}
            y_train_fit = y_train.map(class_to_number)
            y_test_eval = y_test.map(class_to_number)

    pipeline = build_classifier(model_name, task=task)
    print(f"Training {model_name} ({task}) on {X_train.shape[0]} pairs, {X_train.shape[1]} features")
    pipeline.fit(X_train, y_train_fit)

    model = TrainedModel(
        pipeline=pipeline,
        task=task,
        model_name=model_name,
        feature_columns=list(X.columns.astype(str)),
        classes=classes,
    )
    model.metrics = evaluate(model, X_test, y_test_eval)

    return model


def evaluate(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Score a fitted model: accuracy, report and confusion matrix."""

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = model.pipeline.predict(X)

    return {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "report_text": classification_report(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "labels": sorted({str(label) for label in pd.unique(pd.Series(y))}),
    }


def save_model(model: TrainedModel, path: str | Path) -> Path:
    """Persist a trained model, with joblib when available."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": model.pipeline,
        "task": model.task,
        "model_name": model.model_name,
        "feature_columns": model.feature_columns,
        "classes": model.classes,
        "metrics": model.metrics,
    }

    try:
        import joblib

        joblib.dump(payload, out_path)
    except ImportError:  # pragma: no cover - joblib ships with scikit-learn
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)

    return out_path


def load_model(path: str | Path) -> TrainedModel:
    """Load a model saved by :func:`save_model`."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one with 'smartamr train' first."
        )

    try:
        import joblib

        payload = joblib.load(model_path)
    except ImportError:  # pragma: no cover
        with model_path.open("rb") as handle:
            payload = pickle.load(handle)

    return TrainedModel(
        pipeline=payload["pipeline"],
        task=payload["task"],
        model_name=payload["model_name"],
        feature_columns=payload["feature_columns"],
        classes=payload.get("classes", []),
        metrics=payload.get("metrics", {}),
    )
