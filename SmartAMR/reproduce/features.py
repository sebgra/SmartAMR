"""Pair phage and bacteria embeddings into the labelled feature table.

One row per ``(phage, host)`` pair: the phage's vector concatenated with the
host's vector, plus the label saying whether that phage's metadata lists that
host as an isolation host. This is the table both training tasks read.

Layout of the result, which the training code and the app both depend on:

======================  ==========================================
``phage_id``            INPHARED accession
``bacteria_id``         host name
``taxon_id``            host's UniProt taxon
``label``               True when the pair interacts
``p0 ... p{n-1}``       the phage embedding
``b0 ... b{m-1}``       the host embedding
======================  ==========================================

Named columns rather than the original's bare integers ``0..4198``: the
notebooks had to slice the phage block out with hardcoded positions
(``iloc[:, 3:3+3863]``), and those positions silently moved whenever the
embedding width changed. A prefix makes the two blocks addressable by name, so
:func:`phage_columns` and :func:`bacteria_columns` replace the magic numbers.

The table is written as Parquet, not CSV. With 960+960 float columns a CSV is
~15x larger, loses the float precision it round-trips through text, and takes
seconds to parse on every app page load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from SmartAMR.config import Settings
from SmartAMR.reproduce.embed import EmbeddingStore

PHAGE_PREFIX = "p"
BACTERIA_PREFIX = "b"
METADATA_COLUMNS = ("phage_id", "bacteria_id", "taxon_id", "label")


@dataclass(frozen=True)
class FeatureReport:
    """Shape and provenance of a built feature table."""

    rows: int
    positives: int
    phage_dim: int
    bacteria_dim: int
    phages: int
    hosts: int
    dropped_missing_phage: int
    dropped_missing_bacteria: int
    pooling: str

    @property
    def negatives(self) -> int:
        return self.rows - self.positives

    @property
    def positive_rate(self) -> float:
        return self.positives / self.rows if self.rows else 0.0

    def summary(self) -> str:
        lines = [
            f"rows              {self.rows}",
            f"  interacting     {self.positives} ({self.positive_rate:.1%})",
            f"  non-interacting {self.negatives}",
            f"phages            {self.phages}",
            f"hosts             {self.hosts}",
            f"features          {self.phage_dim} phage + {self.bacteria_dim} bacteria "
            f"= {self.phage_dim + self.bacteria_dim}",
            f"pooling           {self.pooling}",
        ]
        if self.dropped_missing_phage or self.dropped_missing_bacteria:
            lines.append(
                f"dropped           {self.dropped_missing_phage} pairs without a phage "
                f"embedding, {self.dropped_missing_bacteria} without a host embedding"
            )
        return "\n".join(lines)


def phage_columns(frame: pd.DataFrame) -> list[str]:
    """The phage embedding columns of a feature table, in order."""
    return [column for column in frame.columns if _is_block_column(column, PHAGE_PREFIX)]


def bacteria_columns(frame: pd.DataFrame) -> list[str]:
    """The bacteria embedding columns of a feature table, in order."""
    return [column for column in frame.columns if _is_block_column(column, BACTERIA_PREFIX)]


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Every embedding column: the phage block then the bacteria block."""
    return phage_columns(frame) + bacteria_columns(frame)


def _is_block_column(column: str, prefix: str) -> bool:
    """True for ``<prefix><digits>``, which is how the blocks are named."""
    return (
        isinstance(column, str)
        and len(column) > len(prefix)
        and column.startswith(prefix)
        and column[len(prefix) :].isdigit()
    )


def _pad_to_width(vectors: dict[str, np.ndarray], width: int) -> dict[str, np.ndarray]:
    """Zero-pad every vector to ``width``.

    Only ``legacy`` pooling needs this: its vectors are as long as the longest
    protein of each organism, so they differ between organisms and have to be
    squared off before they can share a feature space.
    """

    return {
        key: (vector if vector.shape[0] == width else np.pad(vector, (0, width - vector.shape[0])))
        for key, vector in vectors.items()
    }


def build_features(
    settings: Settings,
    pairs: pd.DataFrame | None = None,
    phage_store: EmbeddingStore | None = None,
    bacteria_store: EmbeddingStore | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Join the pair list with both embedding stores into one table.

    Pairs whose phage or host has no embedding are dropped and counted, rather
    than filled with zeros: a zero vector is a statement about the organism
    that the data does not support.
    """

    pairs = pd.read_csv(settings.phage_pairs) if pairs is None else pairs
    phage_store = EmbeddingStore(settings.phage_embedding_store) if phage_store is None else phage_store
    bacteria_store = (
        EmbeddingStore(settings.bacteria_embedding_store) if bacteria_store is None else bacteria_store
    )

    if not len(phage_store):
        raise FileNotFoundError(
            f"No phage embeddings at {settings.phage_embedding_store}. Run the embed stage first."
        )
    if not len(bacteria_store):
        raise FileNotFoundError(
            f"No bacteria embeddings at {settings.bacteria_embedding_store}. Run the embed stage first."
        )

    pooling = str(phage_store.meta.get("pooling", "residue"))

    phage_vectors = dict(phage_store.vectors)
    bacteria_vectors = dict(bacteria_store.vectors)
    if pooling == "legacy":
        phage_width = max(vector.shape[0] for vector in phage_vectors.values())
        bacteria_width = max(vector.shape[0] for vector in bacteria_vectors.values())
        phage_vectors = _pad_to_width(phage_vectors, phage_width)
        bacteria_vectors = _pad_to_width(bacteria_vectors, bacteria_width)

    phage_dim = next(iter(phage_vectors.values())).shape[0]
    bacteria_dim = next(iter(bacteria_vectors.values())).shape[0]

    pairs = pairs.copy()
    pairs["phage_id"] = pairs["phage_id"].astype(str)
    pairs["taxon_id"] = pairs["taxon_id"].astype(str)

    has_phage = pairs["phage_id"].isin(phage_vectors)
    has_bacteria = pairs["taxon_id"].isin(bacteria_vectors)
    kept = pairs.loc[has_phage & has_bacteria].reset_index(drop=True)

    if kept.empty:
        raise ValueError(
            "No pair had both a phage and a host embedding. The embedding stores "
            "and the pair list disagree on their keys."
        )

    # Build the two float blocks as single arrays. Stacking once per block and
    # handing the result to the DataFrame constructor keeps this O(rows) in
    # allocations; assigning 1920 columns one at a time does not.
    phage_block = np.stack([phage_vectors[key] for key in kept["phage_id"]], axis=0)
    bacteria_block = np.stack([bacteria_vectors[key] for key in kept["taxon_id"]], axis=0)

    frame = pd.concat(
        [
            kept[list(METADATA_COLUMNS)].reset_index(drop=True),
            pd.DataFrame(
                phage_block.astype(np.float32),
                columns=[f"{PHAGE_PREFIX}{index}" for index in range(phage_dim)],
            ),
            pd.DataFrame(
                bacteria_block.astype(np.float32),
                columns=[f"{BACTERIA_PREFIX}{index}" for index in range(bacteria_dim)],
            ),
        ],
        axis=1,
    )
    frame["label"] = frame["label"].astype(bool)

    report = FeatureReport(
        rows=len(frame),
        positives=int(frame["label"].sum()),
        phage_dim=phage_dim,
        bacteria_dim=bacteria_dim,
        phages=int(frame["phage_id"].nunique()),
        hosts=int(frame["bacteria_id"].nunique()),
        dropped_missing_phage=int((~has_phage).sum()),
        dropped_missing_bacteria=int((has_phage & ~has_bacteria).sum()),
        pooling=pooling,
    )
    return frame, report


def write_features(settings: Settings, frame: pd.DataFrame) -> Path:
    """Write the feature table as Parquet and return its path."""
    path = Path(settings.reproduced_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_features(settings: Settings) -> pd.DataFrame:
    """Read the built feature table."""
    path = Path(settings.reproduced_features)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature table at {path}. Run `smartamr reproduce features` first."
        )
    return pd.read_parquet(path)
