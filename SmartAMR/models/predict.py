"""Predict whether a phage infects a bacteria.

One entry point shared by the CLI, the dashboard and the Flask backend.

Until a model is trained and saved, prediction falls back to the random
placeholder that ``web_interface/main.py`` used (``random.choice([1, 0])``).
The fallback is never silent: every result says whether it came from a real
model or from the placeholder, through :attr:`Prediction.source`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from SmartAMR.fasta import iter_fasta
from SmartAMR.models.classifiers import BINARY, TrainedModel, load_model

MODEL_SOURCE = "model"
PLACEHOLDER_SOURCE = "placeholder"

PLACEHOLDER_WARNING = (
    "No trained model available - results are RANDOM placeholders. "
    "Train one with 'smartamr train' to get real predictions."
)


@dataclass
class Prediction:
    """One phage-bacteria evaluation."""

    phage: str
    bacteria: str
    interaction: int
    score: float | None = None
    source: str = PLACEHOLDER_SOURCE

    @property
    def is_placeholder(self) -> bool:
        return self.source == PLACEHOLDER_SOURCE

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def placeholder_prediction(phage: str = "", bacteria: str = "", seed: int | None = None) -> Prediction:
    """Random stand-in result, kept from the original web backend."""
    rng = random.Random(seed) if seed is not None else random
    return Prediction(
        phage=phage,
        bacteria=bacteria,
        interaction=rng.choice([1, 0]),
        score=None,
        source=PLACEHOLDER_SOURCE,
    )


def resolve_model(model: TrainedModel | str | Path | None) -> TrainedModel | None:
    """Accept a model, a path to one, or None, and return a model or None."""

    if model is None:
        return None
    if isinstance(model, TrainedModel):
        return model

    path = Path(model)
    if not path.exists():
        print(f"Warning: no model at {path} - {PLACEHOLDER_WARNING}")
        return None

    return load_model(path)


def _feature_frame(vector: np.ndarray, model: TrainedModel) -> pd.DataFrame:
    """Shape one concatenated embedding like the training features.

    The vector is truncated or zero-padded to the width the model was fitted
    on, so a prediction never fails on an off-by-a-few embedding length.
    """

    width = len(model.feature_columns)
    flat = np.asarray(vector, dtype=float).ravel()

    if flat.shape[0] != width:
        print(
            f"Warning: embedding has {flat.shape[0]} value(s) but the model expects {width}; "
            "adjusting. Check that the embeddings come from the same ESM-C model as training."
        )

    if flat.shape[0] < width:
        flat = np.pad(flat, (0, width - flat.shape[0]))
    elif flat.shape[0] > width:
        flat = flat[:width]

    return pd.DataFrame([flat], columns=model.feature_columns)


def predict_pair(
    phage_embedding: Sequence[float] | np.ndarray,
    bacteria_embedding: Sequence[float] | np.ndarray | None = None,
    model: TrainedModel | str | Path | None = None,
    phage: str = "",
    bacteria: str = "",
) -> Prediction:
    """Predict one pair from its embeddings.

    Parameters
    ----------
    phage_embedding
        Flattened phage embedding.
    bacteria_embedding
        Flattened bacteria embedding. Omitted for a multilabel model, which
        reads the phage half only.
    model
        A trained model, a path to one, or None to use the placeholder.
    phage, bacteria
        Names carried through to the result.

    Returns
    -------
    Prediction
    """

    trained = resolve_model(model)
    if trained is None:
        return placeholder_prediction(phage=phage, bacteria=bacteria)

    if trained.task == BINARY:
        if bacteria_embedding is None:
            raise ValueError("The binary model needs both a phage and a bacteria embedding")
        vector = np.append(np.asarray(phage_embedding).ravel(), np.asarray(bacteria_embedding).ravel())
    else:
        vector = np.asarray(phage_embedding).ravel()

    X = _feature_frame(vector, trained)
    prediction = trained.predict(X)[0]

    score: float | None = None
    probabilities = trained.predict_proba(X)
    if probabilities is not None:
        score = float(np.max(probabilities[0]))

    if trained.task == BINARY:
        return Prediction(
            phage=phage,
            bacteria=bacteria,
            interaction=int(prediction),
            score=score,
            source=MODEL_SOURCE,
        )

    # Multilabel: the prediction is the predicted host name.
    return Prediction(
        phage=phage,
        bacteria=str(prediction),
        interaction=1,
        score=score,
        source=MODEL_SOURCE,
    )


def evaluate_sequences(
    bacteria_sequence: str,
    phage_sequence: str,
    model: TrainedModel | str | Path | None = None,
    esm_model: Any | None = None,
    phage: str = "",
    bacteria: str = "",
) -> Prediction:
    """Predict one pair straight from two protein sequences.

    Both a trained classifier and a loaded ESM-C model are required to get a
    real prediction, since the classifier reads embeddings rather than
    sequences. Without them the placeholder is returned.
    """

    trained = resolve_model(model)

    if trained is None or esm_model is None:
        return placeholder_prediction(phage=phage, bacteria=bacteria)

    from SmartAMR.embedding.esmc import mean_embedding

    phage_embedding = mean_embedding(esm_model, phage_sequence)
    bacteria_embedding = mean_embedding(esm_model, bacteria_sequence)

    return predict_pair(
        phage_embedding=phage_embedding,
        bacteria_embedding=bacteria_embedding,
        model=trained,
        phage=phage,
        bacteria=bacteria,
    )


def screen_database(
    query_name: str,
    query_sequence: str,
    database: str | Path,
    query_is_phage: bool = True,
    model: TrainedModel | str | Path | None = None,
    esm_model: Any | None = None,
) -> List[Prediction]:
    """Evaluate one sequence against every record of a FASTA database.

    This is the loop the Flask endpoint ran when only one side of the pair was
    provided: a phage is screened against the bacteria database, or the other
    way round.
    """

    results: List[Prediction] = []

    for name, sequence in iter_fasta(database):
        if query_is_phage:
            results.append(
                evaluate_sequences(
                    bacteria_sequence=sequence,
                    phage_sequence=query_sequence,
                    model=model,
                    esm_model=esm_model,
                    phage=query_name,
                    bacteria=name,
                )
            )
        else:
            results.append(
                evaluate_sequences(
                    bacteria_sequence=query_sequence,
                    phage_sequence=sequence,
                    model=model,
                    esm_model=esm_model,
                    phage=name,
                    bacteria=query_name,
                )
            )

    return results


def predictions_to_frame(predictions: Iterable[Prediction]) -> pd.DataFrame:
    """Collect predictions into a DataFrame for display or export."""
    return pd.DataFrame([prediction.as_dict() for prediction in predictions])
