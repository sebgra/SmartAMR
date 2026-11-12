"""The analyses the classification notebooks ran, as reusable functions.

Ported from ``notebooks/binary_classification.ipynb`` and
``notebooks/multilabel_classification.ipynb``:

- ROC and DET curves per estimator, and all estimators on one pair of axes
- the silhouette analysis, including the random-embedding baseline
- the four estimators trained on one split so their scores are comparable

Everything returns a tidy DataFrame, so the dashboard and the figure export
draw the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from SmartAMR.models.classifiers import (
    BINARY,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    MODEL_NAMES,
    MULTILABEL,
    TrainedModel,
    build_classifier,
    evaluate,
    prepare_binary,
    prepare_multilabel,
)


def decision_scores(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Continuous scores for the positive class.

    ``predict_proba`` when the estimator has it, ``decision_function``
    otherwise. This is what scikit-learn's own ROC display does, and it is why
    an SVC without ``probability=True`` still plots a curve.
    """

    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):  # pragma: no cover
            pass

    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(X))

    # Last resort: the hard prediction, which gives a single-threshold curve.
    return np.asarray(pipeline.predict(X), dtype=float)


def roc_points(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, float]:
    """Points of the ROC curve, and the area under it."""
    from sklearn.metrics import auc, roc_curve

    scores = decision_scores(model.pipeline, X)
    fpr, tpr, _thresholds = roc_curve(y, scores)

    frame = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    frame["Model"] = model.model_name

    return frame, float(auc(fpr, tpr))


def det_points(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Points of the detection error tradeoff curve.

    False-negative rate against false-positive rate, which is what
    ``DetCurveDisplay`` plots.
    """

    from sklearn.metrics import det_curve

    scores = decision_scores(model.pipeline, X)
    fpr, fnr, _thresholds = det_curve(y, scores)

    frame = pd.DataFrame({"fpr": fpr, "fnr": fnr})
    frame["Model"] = model.model_name

    return frame


def probability_frame(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Score given to every held-out pair, with its true class.

    Shows how well the two classes separate, which a single accuracy number
    hides.
    """

    scores = decision_scores(model.pipeline, X)
    labels = pd.Series(y).map({1: "Infects", 0: "Does not infect"}).fillna(pd.Series(y).astype(str))

    return pd.DataFrame({"score": scores, "True class": labels.values, "Model": model.model_name})


@dataclass
class Experiment:
    """Several estimators fitted on one split, with their curves."""

    task: str
    models: Dict[str, TrainedModel] = field(default_factory=dict)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    roc: pd.DataFrame = field(default_factory=pd.DataFrame)
    det: pd.DataFrame = field(default_factory=pd.DataFrame)
    probabilities: pd.DataFrame = field(default_factory=pd.DataFrame)
    auc: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    failures: Dict[str, str] = field(default_factory=dict)


def run_experiment(
    features: pd.DataFrame,
    model_names: Sequence[str] = MODEL_NAMES,
    task: str = BINARY,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    balance: bool = True,
) -> Experiment:
    """Fit every chosen estimator on the same split and collect its curves.

    One split shared by all estimators is what makes the comparison fair; the
    notebooks did the same.
    """

    from sklearn.model_selection import train_test_split

    if task == BINARY:
        X, y = prepare_binary(features, balance=balance, seed=seed)
        stratify = None
    else:
        X, y = prepare_multilabel(features)
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    experiment = Experiment(
        task=task,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        n_features=int(X_train.shape[1]),
    )

    classes: List[str] = sorted(pd.Series(y).unique().tolist()) if task == MULTILABEL else []

    score_rows: List[Dict[str, Any]] = []
    roc_frames: List[pd.DataFrame] = []
    det_frames: List[pd.DataFrame] = []
    probability_frames: List[pd.DataFrame] = []

    for name in model_names:
        y_train_fit, y_test_eval = y_train, y_test

        if task == MULTILABEL and name == "xgboost":
            class_to_number = {label: index for index, label in enumerate(classes)}
            y_train_fit = y_train.map(class_to_number)
            y_test_eval = y_test.map(class_to_number)

        try:
            pipeline = build_classifier(name, task=task)
            pipeline.fit(X_train, y_train_fit)
        except (ImportError, ValueError) as error:
            experiment.failures[name] = str(error)
            continue

        model = TrainedModel(
            pipeline=pipeline,
            task=task,
            model_name=name,
            feature_columns=list(X.columns.astype(str)),
            classes=classes,
        )
        model.metrics = evaluate(model, X_test, y_test_eval)
        experiment.models[name] = model

        row = {
            "Model": name,
            "Task": task,
            "Accuracy": model.metrics["accuracy"],
            "Test pairs": model.metrics["n_test"],
        }

        if task == BINARY:
            try:
                roc, area = roc_points(model, X_test, y_test_eval)
                roc_frames.append(roc)
                det_frames.append(det_points(model, X_test, y_test_eval))
                probability_frames.append(probability_frame(model, X_test, y_test_eval))
                experiment.auc[name] = area
                row["AUC"] = area
            except (ValueError, AttributeError) as error:  # pragma: no cover
                experiment.failures[name] = f"curve failed: {error}"

        score_rows.append(row)

    experiment.scores = pd.DataFrame(score_rows)
    experiment.roc = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
    experiment.det = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    experiment.probabilities = (
        pd.concat(probability_frames, ignore_index=True) if probability_frames else pd.DataFrame()
    )

    return experiment


def silhouette_analysis(
    X: pd.DataFrame,
    labels: Sequence[str],
    n_clusters: int | None = None,
    seed: int = DEFAULT_SEED,
    max_samples: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Silhouette of the embeddings, against a random-embedding baseline.

    Ported from the silhouette cells of both classification notebooks: the
    samples are scored against the true labels and against KMeans clusters,
    and the same is done on random vectors so the numbers have a floor to beat.

    Returns
    -------
    tuple
        Per-sample silhouettes (columns ``silhouette``, ``Label``, ``Source``)
        and the average score of each source.
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.preprocessing import StandardScaler

    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    label_series = pd.Series(list(labels), index=frame.index)

    if len(frame) > max_samples:
        frame = frame.sample(max_samples, random_state=seed)
        label_series = label_series.loc[frame.index]

    if label_series.nunique() < 2:
        raise ValueError("The silhouette needs at least two classes")

    values = StandardScaler().fit_transform(frame.values)
    n_clusters = n_clusters or int(label_series.nunique())

    rng = np.random.default_rng(seed)
    random_values = 2 * rng.random(values.shape) - 1

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(values)
    random_clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(random_values)

    rows = [
        pd.DataFrame(
            {
                "silhouette": silhouette_samples(values, label_series.values),
                "Label": label_series.astype(str).values,
                "Source": "True labels",
            }
        ),
        pd.DataFrame(
            {
                "silhouette": silhouette_samples(values, cluster_labels),
                "Label": [f"Cluster {c}" for c in cluster_labels],
                "Source": "KMeans clusters",
            }
        ),
        pd.DataFrame(
            {
                "silhouette": silhouette_samples(random_values, random_clusters),
                "Label": [f"Cluster {c}" for c in random_clusters],
                "Source": "Random baseline",
            }
        ),
    ]

    averages = {
        "True labels": float(silhouette_score(values, label_series.values)),
        "KMeans clusters": float(silhouette_score(values, cluster_labels)),
        "Random baseline": float(silhouette_score(random_values, random_clusters)),
    }

    return pd.concat(rows, ignore_index=True), averages


def rank_within_group(frame: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    """Sort by value within each group and add the within-group rank.

    The silhouette plot draws one sorted run per cluster, which needs this.
    """

    ordered = frame.sort_values([group, value], ascending=[True, False]).copy()
    ordered["rank"] = ordered.groupby(group).cumcount()
    return ordered
