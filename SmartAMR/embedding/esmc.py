"""Protein embeddings with the ESM-C protein language model.

Logic lifted from ``notebooks/ESMC_embedding.ipynb``. ``esm`` is imported lazily
so the rest of the package and the CLI stay usable without the model installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO

from SmartAMR.config import DEFAULT_ESMC_MODEL

#: Hidden layer requested from the model. ESM-C 6B only allows one layer per
#: call, which is why it is explicit here.
DEFAULT_HIDDEN_LAYER = 100


def _require_esm() -> Any:
    """Import the ``esm`` SDK, with an actionable error when it is missing."""
    try:
        import esm  # noqa: F401
        from esm.sdk import api
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'esm' package is required for embedding. Install it with: pip install esm"
        ) from error
    return api


def embedding_config(hidden_layer: int = DEFAULT_HIDDEN_LAYER) -> Any:
    """Build the ``LogitsConfig`` used for every embedding call."""
    api = _require_esm()
    return api.LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=True,
        ith_hidden_layer=hidden_layer,
    )


def load_model(model_name: str = DEFAULT_ESMC_MODEL, device: str = "cuda") -> Any:
    """Load a pretrained ESM-C model onto ``device``.

    Falls back to CPU with a message when CUDA is requested but unavailable.
    """

    _require_esm()
    from esm.models.esmc import ESMC

    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA not available - falling back to CPU")
                device = "cpu"
        except ImportError:  # pragma: no cover - torch ships with esm
            device = "cpu"

    print(f"Loading {model_name} on {device}")
    return ESMC.from_pretrained(model_name).to(device)


def embed_sequence(model: Any, sequence: str, config: Any | None = None) -> Any:
    """Embed a single sequence and return the raw model output."""
    api = _require_esm()
    config = config if config is not None else embedding_config()

    protein = api.ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, config)

    if isinstance(output, api.ESMProteinError):
        raise RuntimeError(f"ESM failed on a sequence of length {len(sequence)}: {output}")

    return output


def batch_embed(model: Any, sequences: Sequence[str], config: Any | None = None) -> List[Any]:
    """Embed several sequences in parallel; the SDK batches the calls itself."""
    api = _require_esm()
    config = config if config is not None else embedding_config()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_sequence, model, sequence, config) for sequence in sequences]

        results: List[Any] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - one failure must not sink the batch
                results.append(api.ESMProteinError(500, str(error)))

    return results


def mean_embedding(model: Any, sequence: str, config: Any | None = None) -> np.ndarray:
    """Per-residue embedding of one sequence, averaged over the feature axis."""
    output = embed_sequence(model, sequence, config)
    return np.mean(output.embeddings.cpu().squeeze(0).numpy(), axis=1)


def pad_arrays_to_max_shape(list_of_arrays: Sequence[np.ndarray], pad_value: float = 0) -> List[np.ndarray]:
    """Pad 1D arrays with ``pad_value`` so they all match the longest one.

    Parameters
    ----------
    list_of_arrays
        Arrays to pad.
    pad_value
        Value used for the padding.

    Returns
    -------
    list of numpy.ndarray
        The padded arrays, all of the same length.
    """

    arrays = list(list_of_arrays)
    if not arrays:
        return []

    max_rows = max(array.shape[0] for array in arrays)

    return [
        np.pad(array, pad_width=(0, max_rows - array.shape[0]), mode="constant", constant_values=pad_value)
        for array in arrays
    ]


def embeddings_from_fasta(
    proteome_fasta: str | Path,
    model: Any,
    config: Any | None = None,
) -> np.ndarray:
    """Average embedding of every protein of a FASTA proteome."""

    config = config if config is not None else embedding_config()

    embeddings: Dict[str, np.ndarray] = {}
    for protein in SeqIO.parse(str(proteome_fasta), "fasta"):
        embeddings[protein.id] = mean_embedding(model, str(protein.seq), config)

    if not embeddings:
        raise ValueError(f"No sequence found in {proteome_fasta}")

    padded = pad_arrays_to_max_shape(list(embeddings.values()))
    return np.mean(padded, axis=0)


def embeddings_from_csv(
    proteome: str | Path,
    model: Any,
    config: Any | None = None,
) -> tuple[str, np.ndarray]:
    """Average embedding of a phage proteome stored as CSV.

    The CSV is expected to carry the ``phage_id``, ``protein_id`` and ``seq``
    columns produced by the phage proteome export.
    """

    config = config if config is not None else embedding_config()

    df = pd.read_csv(proteome)
    for column in ("phage_id", "protein_id", "seq"):
        if column not in df.columns:
            raise ValueError(f"{proteome} is missing the {column!r} column")

    phage_id = str(df["phage_id"].iloc[0])
    embeddings = [mean_embedding(model, sequence, config) for sequence in df["seq"]]

    padded = pad_arrays_to_max_shape(embeddings)
    return phage_id, np.mean(padded, axis=0)


def embed_proteomes(
    proteomes: Sequence[str | Path],
    model: Any,
    id_column: str,
    config: Any | None = None,
) -> pd.DataFrame:
    """Embed several proteomes into a two column ``(id, embeddings)`` frame.

    CSV inputs carry their own id; FASTA inputs are keyed by file stem.
    """

    config = config if config is not None else embedding_config()
    rows: Dict[str, List[Any]] = {id_column: [], "embeddings": []}

    for proteome in proteomes:
        path = Path(proteome)
        print(f"Embedding {path.name}")

        if path.suffix.lower() == ".csv":
            identifier, embedding = embeddings_from_csv(path, model, config)
        else:
            identifier, embedding = path.stem, embeddings_from_fasta(path, model, config)

        rows[id_column].append(identifier)
        rows["embeddings"].append(embedding)

    return pd.DataFrame(rows)


def save_embeddings(df: pd.DataFrame, path: str | Path, key: str) -> Path:
    """Write an embedding frame to HDF5."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(out_path, key=key)
    return out_path


def load_embeddings(path: str | Path, key: str) -> pd.DataFrame:
    """Read back an embedding frame written by :func:`save_embeddings`."""
    return pd.read_hdf(Path(path), key=key)
