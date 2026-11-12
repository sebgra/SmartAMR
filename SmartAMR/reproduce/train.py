"""Train the notebooks' two models on the reproduced feature table.

The project has two prediction tasks, and the notebooks set them up differently
in ways that matter:

**Binary -- does this phage infect this host?**
    From ``notebooks/binary_classification.ipynb``. Features are the full
    concatenated vector (phage block + host block), the target is the
    interaction label. The classes are strongly imbalanced (~11% positive), and
    the notebook handled that by downsampling the majority class to the size of
    the minority class before splitting.

**Multiclass -- which host does this phage infect?**
    From ``notebooks/multilabel_classification.ipynb``. Only interacting pairs
    are used, features are the **phage block alone** (using the host block
    would leak the answer, since the host vector identifies the host), and the
    target is the host name. Despite the notebook's filename this is
    single-label multiclass: each row has exactly one host.

Four estimators are trained per task, as the notebooks did: random forest, SVM,
XGBoost and AdaBoost, each behind a ``StandardScaler``.

Two deliberate changes from the notebooks
-----------------------------------------
*Splitting.* The binary notebook downsampled and then split, which lets the
same phage appear in both train and test with different hosts. Since a phage's
embedding is identical across its nine rows, that leaks: the model can memorise
the phage rather than learn the interaction. Splitting here is **grouped by
phage** by default, so every row of a phage lands on one side of the split.
Pass ``group_by_phage=False`` for the notebooks' row-wise behaviour.

*Scoring.* The notebooks read accuracy off a balanced test set. Accuracy on a
balanced set is reported too, but ROC AUC and per-class F1 are recorded because
they survive a change of class balance.

A note on the AdaBoost result
-----------------------------
The project's ROC figure shows AdaBoost below the diagonal (AUC < 0.5), which
usually means its decision scores are sign-flipped relative to the positive
class. :func:`evaluate` always takes the positive column of
``predict_proba`` and never ``decision_function``, so a below-chance AUC here
is a real result for the estimator rather than a plotting artefact.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from SmartAMR.config import Settings
from SmartAMR.reproduce.features import embedding_columns, phage_columns

RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_TASK = "binary"
MULTICLASS_TASK = "multiclass"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """Metrics for one estimator on one task."""

    task: str
    model: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    roc_auc: float | None = None
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0
    features: int = 0
    fit_seconds: float = 0.0
    error: str = ""
    #: ROC curve points, for the app to draw without refitting.
    roc_curve: dict[str, list[float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error == ""


@dataclass
class TaskResult:
    """Every estimator's result for one task, plus how the data was split."""

    task: str
    rows: int
    positives: int | None
    classes: list[str]
    features: int
    split: str
    models: list[ModelResult] = field(default_factory=list)

    def best(self) -> ModelResult | None:
        """The estimator with the highest macro F1, ignoring failures."""
        scored = [result for result in self.models if result.ok]
        return max(scored, key=lambda result: result.f1_macro) if scored else None


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def estimator_factories(threads: int = 1) -> dict[str, Callable[[], Any]]:
    """The four estimators the notebooks compared, each scaled.

    ``threads`` caps the parallelism of the estimators that support it, so the
    resource budget reaches scikit-learn and XGBoost too.
    """

    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    return {
        "random-forest": lambda: make_pipeline(
            StandardScaler(),
            RandomForestClassifier(max_depth=5, random_state=0, n_jobs=threads),
        ),
        # A plain SVC, with no probability estimates. ROC comes from its
        # decision_function instead -- see the note in `evaluate`.
        #
        # Neither probability route is usable here, and both fail the same way.
        # `SVC(probability=True)` (deprecated in scikit-learn 1.9) and
        # `CalibratedClassifierCV` both fit their sigmoid on an internal
        # cross-validation that is NOT group-aware. Every phage in this dataset
        # contributes a positive row and a negative row that differ only in the
        # bacteria half of the vector, so a non-grouped split puts a phage's two
        # rows in different folds. The sigmoid then learns a NEGATIVE slope, and
        # the probabilities come out perfectly anti-correlated (r = -1.0) with
        # the SVM's own decision_function: AUC 0.342 and accuracy 0.332, from a
        # model whose raw decision_function scores AUC 0.658 and accuracy 0.674.
        "svm": lambda: make_pipeline(
            StandardScaler(),
            SVC(gamma="auto", kernel="linear", random_state=RANDOM_STATE),
        ),
        "xgboost": lambda: make_pipeline(
            StandardScaler(),
            XGBClassifier(
                max_depth=6,
                random_state=RANDOM_STATE,
                n_jobs=threads,
                eval_metric="logloss",
                tree_method="hist",
            ),
        ),
        "adaboost": lambda: make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    task: str,
    name: str,
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: list[str],
    train_rows: int,
    fit_seconds: float,
) -> ModelResult:
    """Score a fitted pipeline on the held-out split."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )

    predicted = pipeline.predict(x_test)

    result = ModelResult(
        task=task,
        model=name,
        accuracy=float(accuracy_score(y_test, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
        f1_macro=float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        classes=classes,
        train_rows=train_rows,
        test_rows=len(y_test),
        features=x_test.shape[1],
        fit_seconds=fit_seconds,
    )

    per_class = f1_score(y_test, predicted, average=None, labels=classes, zero_division=0)
    result.per_class_f1 = {
        str(label): float(score) for label, score in zip(classes, per_class)
    }
    result.confusion = confusion_matrix(y_test, predicted, labels=classes).tolist()

    # ROC AUC: binary gets a curve, multiclass gets the one-vs-rest average.
    #
    # An estimator with no predict_proba is scored from decision_function,
    # which is what the notebooks' RocCurveDisplay did for the SVM. ROC only
    # needs a ranking, so an uncalibrated margin is a perfectly valid score --
    # and for this SVM a far more trustworthy one than a fitted sigmoid.
    try:
        if not hasattr(pipeline, "predict_proba") and hasattr(pipeline, "decision_function"):
            margins = pipeline.decision_function(x_test)
            if len(classes) == 2:
                result.roc_auc = float(roc_auc_score(y_test, margins))
                false_positive, true_positive, _ = roc_curve(
                    y_test, margins, pos_label=classes[1]
                )
                step = max(1, len(false_positive) // 200)
                result.roc_curve = {
                    "fpr": [float(value) for value in false_positive[::step]],
                    "tpr": [float(value) for value in true_positive[::step]],
                }
            else:
                result.roc_auc = float(
                    roc_auc_score(y_test, margins, multi_class="ovr", average="macro")
                )
            return result

        probabilities = pipeline.predict_proba(x_test)
        if len(classes) == 2:
            positive = probabilities[:, list(pipeline.classes_).index(classes[1])]
            result.roc_auc = float(roc_auc_score(y_test, positive))
            false_positive, true_positive, _ = roc_curve(
                y_test, positive, pos_label=classes[1]
            )
            # Thin the curve: a few hundred points draw identically to a few
            # thousand and keep the metrics file small.
            step = max(1, len(false_positive) // 200)
            result.roc_curve = {
                "fpr": [float(value) for value in false_positive[::step]],
                "tpr": [float(value) for value in true_positive[::step]],
            }
        else:
            result.roc_auc = float(
                roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
            )
    except Exception:  # noqa: BLE001 - a missing predict_proba is not fatal
        # ROC is best-effort: the accuracy and F1 above are already computed
        # from hard predictions and stand on their own.
        result.roc_auc = None

    return result


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
    group_by_phage: bool,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, str]:
    """Split into train/test, optionally keeping each phage on one side.

    Returns ``(x_train, x_test, y_train, y_test, description)``.
    """

    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    features = frame[feature_columns]
    values = target.to_numpy()

    if group_by_phage:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_index, test_index = next(
            splitter.split(features, values, groups=frame["phage_id"].to_numpy())
        )
        return (
            features.iloc[train_index],
            features.iloc[test_index],
            values[train_index],
            values[test_index],
            "grouped by phage (no phage in both splits)",
        )

    stratify_on = values if stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_on,
    )
    return x_train, x_test, y_train, y_test, "row-wise" + (" stratified" if stratify else "")


def downsample_majority(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Cut the non-interacting rows down to the number of interacting ones.

    This is what the binary notebook did, via ``sklearn.utils.resample``.
    """

    positives = frame[frame["label"]]
    negatives = frame[~frame["label"]]
    if positives.empty or negatives.empty:
        return frame
    keep = min(len(positives), len(negatives))
    return pd.concat(
        [
            positives.sample(n=keep, random_state=random_state),
            negatives.sample(n=keep, random_state=random_state),
        ]
    ).sample(frac=1.0, random_state=random_state)


# ---------------------------------------------------------------------------
# The two tasks
# ---------------------------------------------------------------------------
def train_binary(
    frame: pd.DataFrame,
    threads: int = 1,
    group_by_phage: bool = True,
    balance: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[TaskResult, dict[str, Any]]:
    """Does this phage infect this host? Full feature vector, binary target."""
    working = downsample_majority(frame) if balance else frame
    columns = embedding_columns(working)

    x_train, x_test, y_train, y_test, split = split_frame(
        working,
        columns,
        working["label"].astype(int),
        group_by_phage=group_by_phage,
        stratify=True,
    )
    classes = [0, 1]

    task = TaskResult(
        task=BINARY_TASK,
        rows=len(working),
        positives=int(working["label"].sum()),
        classes=[str(label) for label in classes],
        features=len(columns),
        split=split + (" after downsampling" if balance else ""),
    )
    log(
        f"binary: {len(working)} rows ({task.positives} positive), "
        f"{len(columns)} features, split {split}"
    )

    fitted: dict[str, Any] = {}
    for name, factory in estimator_factories(threads).items():
        pipeline = factory()
        started = time.time()
        try:
            pipeline.fit(x_train, y_train)
        except Exception as error:  # noqa: BLE001 - keep the other three
            log(f"  {name}: FAILED {type(error).__name__}: {error}")
            task.models.append(
                ModelResult(
                    task=BINARY_TASK,
                    model=name,
                    accuracy=0.0,
                    balanced_accuracy=0.0,
                    f1_macro=0.0,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue

        elapsed = time.time() - started
        result = evaluate(
            BINARY_TASK, name, pipeline, x_test, y_test, classes, len(y_train), elapsed
        )
        task.models.append(result)
        fitted[name] = pipeline
        auc = f"{result.roc_auc:.3f}" if result.roc_auc is not None else "n/a"
        log(
            f"  {name}: acc {result.accuracy:.3f} F1 {result.f1_macro:.3f} "
            f"AUC {auc} ({elapsed:.1f}s)"
        )

    return task, fitted


def train_multiclass(
    frame: pd.DataFrame,
    threads: int = 1,
    group_by_phage: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[TaskResult, dict[str, Any]]:
    """Which host does this phage infect? Phage features only, host as target.

    Only interacting pairs take part. The bacteria block is excluded: it is
    constant per host, so including it would hand the model the answer.
    """

    interacting = frame[frame["label"]].reset_index(drop=True)
    columns = phage_columns(interacting)

    # Classes with a single member cannot be split; drop them and say so.
    counts = interacting["bacteria_id"].value_counts()
    too_rare = counts[counts < 2]
    if len(too_rare):
        log(f"  dropping {len(too_rare)} host(s) with <2 interacting phages: {list(too_rare.index)}")
        interacting = interacting[~interacting["bacteria_id"].isin(too_rare.index)]

    target = interacting["bacteria_id"].astype(str)
    classes = sorted(target.unique())

    x_train, x_test, y_train, y_test, split = split_frame(
        interacting, columns, target, group_by_phage=group_by_phage, stratify=True
    )

    task = TaskResult(
        task=MULTICLASS_TASK,
        rows=len(interacting),
        positives=None,
        classes=classes,
        features=len(columns),
        split=split,
    )
    log(
        f"multiclass: {len(interacting)} interacting rows, {len(classes)} hosts, "
        f"{len(columns)} phage features, split {split}"
    )

    fitted: dict[str, Any] = {}
    for name, factory in estimator_factories(threads).items():
        pipeline = factory()
        # XGBoost needs integer class labels; wrap it so every estimator in
        # this loop takes and returns the host names.
        from sklearn.preprocessing import LabelEncoder

        needs_encoding = name == "xgboost"
        encoder = LabelEncoder().fit(classes) if needs_encoding else None

        started = time.time()
        try:
            pipeline.fit(x_train, encoder.transform(y_train) if encoder else y_train)
        except Exception as error:  # noqa: BLE001
            log(f"  {name}: FAILED {type(error).__name__}: {error}")
            task.models.append(
                ModelResult(
                    task=MULTICLASS_TASK,
                    model=name,
                    accuracy=0.0,
                    balanced_accuracy=0.0,
                    f1_macro=0.0,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue
        elapsed = time.time() - started

        scored = _DecodingPipeline(pipeline, encoder) if encoder else pipeline
        result = evaluate(
            MULTICLASS_TASK, name, scored, x_test, y_test, classes, len(y_train), elapsed
        )
        task.models.append(result)
        fitted[name] = scored
        log(
            f"  {name}: acc {result.accuracy:.3f} F1 {result.f1_macro:.3f} ({elapsed:.1f}s)"
        )

    return task, fitted


class _DecodingPipeline:
    """Wraps a label-encoded estimator so it speaks host names again.

    XGBoost requires ``0..n-1`` integer targets. Rather than scoring it in a
    different label space from the other three estimators -- which would make
    the confusion matrices non-comparable -- predictions are decoded back here.
    """

    def __init__(self, pipeline: Any, encoder: Any) -> None:
        self.pipeline = pipeline
        self.encoder = encoder

    @property
    def classes_(self) -> np.ndarray:
        return self.encoder.inverse_transform(self.pipeline.classes_)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.encoder.inverse_transform(self.pipeline.predict(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_models(settings: Settings, task: str, fitted: dict[str, Any]) -> list[Path]:
    """Persist each fitted pipeline as ``<task>-<model>.joblib``."""
    import joblib

    settings.ensure(settings.models_dir)
    written: list[Path] = []
    for name, pipeline in fitted.items():
        path = settings.model_path(f"{task}-{name}")
        joblib.dump(pipeline, path)
        written.append(path)
    return written


def save_metrics(settings: Settings, tasks: list[TaskResult], extra: dict | None = None) -> Path:
    """Write every task's metrics as one JSON file for the app to read."""
    settings.ensure(settings.metrics_path.parent)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "tasks": [
            {
                **{key: value for key, value in asdict(task).items() if key != "models"},
                "models": [asdict(model) for model in task.models],
            }
            for task in tasks
        ],
    }
    if extra:
        payload.update(extra)
    settings.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return settings.metrics_path


def load_metrics(settings: Settings) -> dict:
    """Read the metrics file, or ``{}`` when no training has run."""
    path = Path(settings.metrics_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
