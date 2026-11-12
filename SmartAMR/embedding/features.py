"""Pair phage and bacteria embeddings into the training feature table.

Logic lifted from the last part of ``notebooks/ESMC_embedding.ipynb``: every
phage is paired with every bacteria, the two embeddings are concatenated, and
the label says whether the inphared metadata records that pair as a real
phage-host relationship.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from SmartAMR.config import INPHARED_ACCESSION_COLUMN, INPHARED_HOST_COLUMN
from SmartAMR.embedding.esmc import pad_arrays_to_max_shape


def find_label(metadata: pd.DataFrame, phage_id: str, bacteria_id: str) -> bool:
    """True when the metadata lists ``bacteria_id`` as a host of ``phage_id``."""
    phage_metadata = metadata[metadata[INPHARED_ACCESSION_COLUMN] == phage_id]
    pair_metadata = phage_metadata[
        phage_metadata[INPHARED_HOST_COLUMN].str.contains(bacteria_id, na=False)
    ]
    return pair_metadata.shape[0] > 0


def flatten_embeddings(df: pd.DataFrame, column: str = "embeddings") -> pd.DataFrame:
    """Add padded and flattened copies of an embedding column."""
    df = df.copy()
    df["padded_embeddings"] = pad_arrays_to_max_shape(df[column].values)
    df["flat_embeddings"] = [embedding.flatten() for embedding in df["padded_embeddings"].values]
    return df


def build_features(
    phage_embeddings: pd.DataFrame,
    bacteria_embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build the paired feature table.

    Parameters
    ----------
    phage_embeddings
        Frame with ``phage_id`` and ``embeddings`` columns.
    bacteria_embeddings
        Frame with ``bacteria_id`` and ``embeddings`` columns.
    metadata
        The inphared metadata, used to label each pair.

    Returns
    -------
    pandas.DataFrame
        One row per (phage, bacteria) pair, with ``phage_id``, ``bacteria_id``,
        ``embedding`` (the concatenated vector) and a boolean ``label``.
    """

    phages = flatten_embeddings(phage_embeddings)
    bacterias = flatten_embeddings(bacteria_embeddings)

    phage_ids = phages["phage_id"].unique()
    bacteria_ids = bacterias["bacteria_id"].unique()

    rows = {"phage_id": [], "bacteria_id": [], "embedding": [], "label": []}

    for phage_id, bacteria_id in itertools.product(phage_ids, bacteria_ids):
        viral_embedding = phages[phages["phage_id"] == phage_id]["flat_embeddings"].values[0]
        bacteria_embedding = bacterias[bacterias["bacteria_id"] == bacteria_id][
            "flat_embeddings"
        ].values[0]

        rows["phage_id"].append(phage_id)
        rows["bacteria_id"].append(bacteria_id)
        rows["embedding"].append(np.append(viral_embedding, bacteria_embedding))
        rows["label"].append(find_label(metadata, phage_id, bacteria_id))

    return pd.DataFrame(rows)


def explode_features(features: pd.DataFrame) -> pd.DataFrame:
    """Spread the ``embedding`` vectors into one column per dimension.

    This is the flat layout the classifiers read (``final_features.csv``).
    """

    exploded = features["embedding"].apply(pd.Series)
    return pd.concat([features.drop("embedding", axis=1), exploded], axis=1)


def balance_features(features: pd.DataFrame, n_per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Draw the same number of positive and negative pairs.

    With ``n_per_class`` left as None, the larger class is downsampled to the
    size of the smaller one.
    """

    positives = features[features["label"]]
    negatives = features[~features["label"]]

    if n_per_class is None:
        n_per_class = min(len(positives), len(negatives))

    n_per_class = min(n_per_class, len(positives), len(negatives))
    if n_per_class == 0:
        raise ValueError("One of the two classes is empty - cannot balance the dataset")

    return pd.concat(
        [
            negatives.sample(n_per_class, random_state=seed),
            positives.sample(n_per_class, random_state=seed),
        ]
    )


def save_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write the exploded feature table as CSV and return its path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    explode_features(features).to_csv(out_path, index=False)
    return out_path


def save_features_hdf(
    features: pd.DataFrame,
    path: str | Path,
    batch_size: int = 10,
    key_prefix: str = "features",
) -> Path:
    """Write the feature table to HDF5 in chunks, as the notebook did."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = features.groupby(np.arange(features.shape[0]) // batch_size)
    for index, chunk in groups:
        chunk.to_hdf(out_path, key=f"{key_prefix}_{index}", complib="blosc:lz4", mode="a")

    return out_path


def load_features(path: str | Path) -> pd.DataFrame:
    """Read the flat feature table produced by :func:`save_features`."""
    features_path = Path(path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature table not found at {features_path}. Run 'smartamr build-features' first."
        )
    return pd.read_csv(features_path)


def embedding_columns(features: pd.DataFrame) -> List[str]:
    """Names of the columns holding the embedding dimensions."""
    metadata_columns = {"phage_id", "bacteria_id", "label", "interaction"}
    return [column for column in features.columns if column not in metadata_columns]
