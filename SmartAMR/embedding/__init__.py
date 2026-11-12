"""ESM-C protein embeddings and the paired feature table built from them."""

from __future__ import annotations

from SmartAMR.embedding.esmc import (
    batch_embed,
    embed_proteomes,
    embed_sequence,
    embeddings_from_csv,
    embeddings_from_fasta,
    load_embeddings,
    load_model,
    mean_embedding,
    pad_arrays_to_max_shape,
    save_embeddings,
)
from SmartAMR.embedding.features import (
    balance_features,
    build_features,
    embedding_columns,
    explode_features,
    find_label,
    load_features,
    save_features,
)

__all__ = [
    "balance_features",
    "batch_embed",
    "build_features",
    "embed_proteomes",
    "embed_sequence",
    "embedding_columns",
    "embeddings_from_csv",
    "embeddings_from_fasta",
    "explode_features",
    "find_label",
    "load_embeddings",
    "load_features",
    "load_model",
    "mean_embedding",
    "pad_arrays_to_max_shape",
    "save_embeddings",
    "save_features",
]
