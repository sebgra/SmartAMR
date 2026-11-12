"""Screen a newly uploaded phage against the bacteria of the project.

This is the "upload a phage and treat it" path: take a proteome the user just
provided, embed it with ESM-C, and rank the bacteria it is likely to infect.

Two model shapes are supported, and they need different inputs:

``multilabel``
    Reads the phage embedding alone and returns a probability per host. This is
    the natural fit here, since a brand new phage has no bacteria counterpart.
``binary``
    Scores one (phage, bacteria) pair, so it also needs the bacteria
    embeddings produced by ``smartamr embed --kind bacteria``.

With no model, or no ESM-C, the result is the usual labelled placeholder rather
than a silent guess.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from SmartAMR.models.classifiers import BINARY, MULTILABEL, TrainedModel
from SmartAMR.models.predict import (
    MODEL_SOURCE,
    PLACEHOLDER_SOURCE,
    placeholder_prediction,
    resolve_model,
)

FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna", ".txt"}


@dataclass
class ScreenResult:
    """Ranked hosts for one phage proteome."""

    phage_id: str
    ranked: pd.DataFrame
    source: str = PLACEHOLDER_SOURCE
    n_proteins: int = 0
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_placeholder(self) -> bool:
        return self.source == PLACEHOLDER_SOURCE

    @property
    def best_host(self) -> str | None:
        if self.ranked.empty:
            return None
        return str(self.ranked.iloc[0]["bacteria"])


def count_proteins(proteome: str | Path) -> int:
    """Number of records in a proteome, FASTA or CSV."""
    path = Path(proteome)

    if path.suffix.lower() == ".csv":
        return len(pd.read_csv(path))

    from SmartAMR.fasta import iter_fasta

    return sum(1 for _ in iter_fasta(path))


def embedding_cache_path(proteome: str | Path, cache_dir: str | Path) -> Path:
    """Where the embedding of this proteome is cached.

    The file content decides the name, so re-uploading the same proteome under
    a different filename still hits the cache, and editing it misses.
    """

    digest = hashlib.sha256(Path(proteome).read_bytes()).hexdigest()[:16]
    return Path(cache_dir) / f"{Path(proteome).stem}-{digest}.npy"


def embed_proteome(
    proteome: str | Path,
    esm_model: Any,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Mean ESM-C embedding of a phage proteome, cached on disk."""

    path = Path(proteome)

    cache_file = None
    if cache_dir is not None:
        cache_file = embedding_cache_path(path, cache_dir)
        if cache_file.exists():
            return np.load(cache_file)

    from SmartAMR.embedding.esmc import embeddings_from_csv, embeddings_from_fasta

    if path.suffix.lower() == ".csv":
        _phage_id, embedding = embeddings_from_csv(path, esm_model)
    else:
        embedding = embeddings_from_fasta(path, esm_model)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embedding)

    return embedding


def rank_hosts_multilabel(model: TrainedModel, phage_embedding: np.ndarray) -> pd.DataFrame:
    """Probability per host from a multilabel model."""

    from SmartAMR.models.predict import _feature_frame

    X = _feature_frame(np.asarray(phage_embedding).ravel(), model)

    probabilities = model.predict_proba(X)
    if probabilities is None:
        # The estimator gives no probabilities (SVC without probability=True),
        # so fall back to the single predicted class.
        predicted = str(model.predict(X)[0])
        return pd.DataFrame({"bacteria": [predicted], "probability": [float("nan")]})

    classes = model.classes or [str(index) for index in range(probabilities.shape[1])]
    ranked = pd.DataFrame({"bacteria": classes, "probability": probabilities[0]})

    return ranked.sort_values("probability", ascending=False).reset_index(drop=True)


def rank_hosts_binary(
    model: TrainedModel,
    phage_embedding: np.ndarray,
    bacteria_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Score the phage against every bacteria with a binary model."""

    from SmartAMR.models.predict import _feature_frame

    rows: List[Dict[str, Any]] = []
    phage_flat = np.asarray(phage_embedding).ravel()

    for _, row in bacteria_embeddings.iterrows():
        vector = np.append(phage_flat, np.asarray(row["embeddings"]).ravel())
        X = _feature_frame(vector, model)

        interaction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)
        probability = float(probabilities[0][-1]) if probabilities is not None else float("nan")

        rows.append(
            {
                "bacteria": str(row["bacteria_id"]),
                "probability": probability,
                "interaction": interaction,
            }
        )

    ranked = pd.DataFrame(rows)
    return ranked.sort_values("probability", ascending=False).reset_index(drop=True)


def screen_phage(
    proteome: str | Path,
    model: TrainedModel | str | Path | None = None,
    esm_model: Any | None = None,
    bacteria_embeddings: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
    phage_id: str | None = None,
) -> ScreenResult:
    """Rank the bacteria a newly uploaded phage is likely to infect.

    Parameters
    ----------
    proteome
        FASTA or CSV holding the phage's proteins.
    model
        A trained model or a path to one. None gives a placeholder.
    esm_model
        A loaded ESM-C model. None gives a placeholder.
    bacteria_embeddings
        Needed only by a binary model.
    cache_dir
        Where to cache the computed embedding.
    phage_id
        Name for the result; defaults to the file stem.

    Returns
    -------
    ScreenResult
    """

    path = Path(proteome)
    name = phage_id or path.stem
    n_proteins = count_proteins(path)

    trained = resolve_model(model)

    if trained is None or esm_model is None:
        hosts = (
            list(bacteria_embeddings["bacteria_id"].astype(str))
            if bacteria_embeddings is not None and not bacteria_embeddings.empty
            else ["unknown"]
        )
        ranked = pd.DataFrame(
            {
                "bacteria": hosts,
                "probability": [float("nan")] * len(hosts),
                "interaction": [placeholder_prediction().interaction for _ in hosts],
            }
        )
        return ScreenResult(phage_id=name, ranked=ranked, source=PLACEHOLDER_SOURCE, n_proteins=n_proteins)

    embedding = embed_proteome(path, esm_model, cache_dir=cache_dir)

    if trained.task == MULTILABEL:
        ranked = rank_hosts_multilabel(trained, embedding)
    elif trained.task == BINARY:
        if bacteria_embeddings is None or bacteria_embeddings.empty:
            raise ValueError(
                "A binary model scores pairs, so it needs the bacteria embeddings. "
                "Run 'smartamr embed --kind bacteria', or screen with a multilabel model."
            )
        ranked = rank_hosts_binary(trained, embedding, bacteria_embeddings)
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"Unknown task {trained.task!r}")

    return ScreenResult(
        phage_id=name,
        ranked=ranked,
        source=MODEL_SOURCE,
        n_proteins=n_proteins,
        embedding=embedding,
    )


def screen_proteomes(
    proteomes: List[str | Path],
    model: TrainedModel | str | Path | None = None,
    esm_model: Any | None = None,
    bacteria_embeddings: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> List[ScreenResult]:
    """Screen several phage proteomes, loading the model once."""

    trained = resolve_model(model)

    return [
        screen_phage(
            proteome,
            model=trained,
            esm_model=esm_model,
            bacteria_embeddings=bacteria_embeddings,
            cache_dir=cache_dir,
        )
        for proteome in proteomes
    ]


def results_to_frame(results: List[ScreenResult], top_n: int | None = None) -> pd.DataFrame:
    """Flatten screening results into one table, best host first per phage."""

    frames: List[pd.DataFrame] = []

    for result in results:
        ranked = result.ranked if top_n is None else result.ranked.head(top_n)
        frame = ranked.copy()
        frame.insert(0, "phage_id", result.phage_id)
        frame["proteins"] = result.n_proteins
        frame["source"] = result.source
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["phage_id", "bacteria", "probability", "proteins", "source"])

    return pd.concat(frames, ignore_index=True)
