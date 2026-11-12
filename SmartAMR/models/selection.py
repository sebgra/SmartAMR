"""Pick phages for a target bacteria, and find the phages nearest a new one.

Two questions the project exists to answer:

``select_phages``
    Given a bacteria to kill, which phages of the corpus are predicted to
    infect it, best first.
``nearest_phages``
    Given a brand new phage, which known phages sit closest to it in embedding
    space, and what are those known to infect.

Both read the phage embeddings. Those come from
``smartamr embed --kind phage``, but the feature table already carries the same
vectors, so :func:`phage_embeddings_from_features` recovers them when the HDF5
file was never written.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from SmartAMR.config import PHAGE_EMBEDDING_SIZE
from SmartAMR.models.classifiers import BINARY, MULTILABEL, TrainedModel
from SmartAMR.models.predict import resolve_model


def phage_embeddings_from_features(
    features: pd.DataFrame,
    phage_embedding_size: int = PHAGE_EMBEDDING_SIZE,
) -> pd.DataFrame:
    """Recover one embedding per phage from the paired feature table.

    Each row holds the phage embedding followed by the bacteria one, and a
    phage appears once per bacteria, so the first block of columns is taken
    and the duplicates dropped.
    """

    from SmartAMR.embedding.features import embedding_columns

    columns = embedding_columns(features)
    width = phage_embedding_size if len(columns) > phage_embedding_size else len(columns) // 2
    if width == 0:
        raise ValueError("The feature table holds no embedding column")

    unique = features.drop_duplicates(subset="phage_id")

    return pd.DataFrame(
        {
            "phage_id": unique["phage_id"].astype(str).values,
            "embeddings": list(unique[columns[:width]].to_numpy(dtype=float)),
        }
    )


def known_hosts(features: pd.DataFrame) -> Dict[str, List[str]]:
    """The hosts each phage of the corpus is recorded as infecting."""
    interacting = features[features["label"].astype(bool)]
    grouped = interacting.groupby("phage_id")["bacteria_id"].unique()

    return {str(phage): sorted(str(host) for host in hosts) for phage, hosts in grouped.items()}


def _stack(embeddings: pd.DataFrame) -> np.ndarray:
    """Embeddings as one 2D array, padded to the widest row."""
    rows = [np.asarray(value, dtype=float).ravel() for value in embeddings["embeddings"]]
    width = max(row.shape[0] for row in rows)

    return np.vstack([np.pad(row, (0, width - row.shape[0])) for row in rows])


def select_phages(
    target_bacteria: str,
    phage_embeddings: pd.DataFrame,
    model: TrainedModel | str | None = None,
    bacteria_embeddings: pd.DataFrame | None = None,
    top_n: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Rank the phages predicted to infect ``target_bacteria``.

    Parameters
    ----------
    target_bacteria
        Name of the bacteria to find phages for.
    phage_embeddings
        Frame with ``phage_id`` and ``embeddings``.
    model
        A trained model, or a path to one.
    bacteria_embeddings
        Needed by a binary model, which scores pairs.
    top_n
        Keep only the best N candidates.
    threshold
        Mark candidates at or above this probability as recommended.

    Returns
    -------
    pandas.DataFrame
        ``phage_id``, ``probability`` and ``recommended``, best first.
    """

    trained = resolve_model(model)
    if trained is None:
        raise ValueError("Selecting phages needs a trained model.")

    if phage_embeddings.empty:
        raise ValueError("No phage embedding available to rank.")

    X = _stack(phage_embeddings)
    phage_ids = phage_embeddings["phage_id"].astype(str).tolist()

    if trained.task == MULTILABEL:
        if target_bacteria not in trained.classes:
            raise ValueError(
                f"{target_bacteria!r} is not one of the model's classes: {', '.join(trained.classes)}"
            )

        frame = _as_model_frame(X, trained)
        probabilities = trained.predict_proba(frame)
        if probabilities is None:
            predictions = trained.predict(frame)
            scores = np.array([1.0 if str(p) == target_bacteria else 0.0 for p in predictions])
        else:
            index = list(trained.classes).index(target_bacteria)
            scores = probabilities[:, index]

    elif trained.task == BINARY:
        if bacteria_embeddings is None or bacteria_embeddings.empty:
            raise ValueError(
                "A binary model scores pairs, so it needs the bacteria embeddings. "
                "Run 'smartamr embed --kind bacteria', or use a multilabel model."
            )

        match = bacteria_embeddings[bacteria_embeddings["bacteria_id"].astype(str) == target_bacteria]
        if match.empty:
            raise ValueError(f"No embedding for {target_bacteria!r} in the bacteria table")

        bacteria_vector = np.asarray(match.iloc[0]["embeddings"], dtype=float).ravel()
        paired = np.hstack([X, np.tile(bacteria_vector, (X.shape[0], 1))])

        frame = _as_model_frame(paired, trained)
        probabilities = trained.predict_proba(frame)
        scores = (
            probabilities[:, -1]
            if probabilities is not None
            else np.asarray(trained.predict(frame), dtype=float)
        )
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"Unknown task {trained.task!r}")

    ranked = pd.DataFrame({"phage_id": phage_ids, "probability": scores})
    ranked = ranked.sort_values("probability", ascending=False).reset_index(drop=True)

    if threshold is not None:
        ranked["recommended"] = ranked["probability"] >= threshold

    return ranked.head(top_n) if top_n else ranked


def _as_model_frame(values: np.ndarray, model: TrainedModel) -> pd.DataFrame:
    """Shape a stack of vectors like the table the model was fitted on."""
    width = len(model.feature_columns)
    rows = values

    if rows.shape[1] < width:
        rows = np.pad(rows, ((0, 0), (0, width - rows.shape[1])))
    elif rows.shape[1] > width:
        rows = rows[:, :width]

    return pd.DataFrame(rows, columns=model.feature_columns)


def nearest_phages(
    embedding: np.ndarray,
    phage_embeddings: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Phages of the corpus closest to ``embedding``, by cosine similarity.

    Useful for a brand new phage: the phages it resembles are a first, model
    free hint at what it might infect.
    """

    if phage_embeddings.empty:
        raise ValueError("No phage embedding available to compare against.")

    known = _stack(phage_embeddings)
    query = np.asarray(embedding, dtype=float).ravel()

    width = known.shape[1]
    if query.shape[0] < width:
        query = np.pad(query, (0, width - query.shape[0]))
    elif query.shape[0] > width:
        query = query[:width]

    norms = np.linalg.norm(known, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = np.nan
    similarity = known @ query / norms

    ranked = pd.DataFrame(
        {"phage_id": phage_embeddings["phage_id"].astype(str).values, "similarity": similarity}
    )

    return ranked.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def annotate_hosts(ranked: pd.DataFrame, hosts: Dict[str, List[str]]) -> pd.DataFrame:
    """Add a ``known hosts`` column to a ranking of phages."""
    annotated = ranked.copy()
    annotated["known hosts"] = [
        ", ".join(hosts.get(str(phage), [])) or "unknown" for phage in annotated["phage_id"]
    ]
    return annotated
