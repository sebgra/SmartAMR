"""ESM-C embeddings for phage and bacterial proteomes.

One organism -- a phage or a bacterial host -- becomes one fixed-length vector:
embed each of its proteins, then average the proteins together. The work is
resumable, because embedding a few hundred proteomes takes minutes and a run
that dies halfway should not start over: results are appended to an ``.npz``
store keyed by accession or taxon, and anything already in the store is skipped.

Pooling
-------
Turning a protein's per-residue embedding matrix of shape ``(L, 960)`` into one
vector can be done two ways, and the choice matters enough to be explicit.

``"residue"`` (the default)
    Average over the **residue** axis, giving a 960-dim vector per protein --
    the standard way protein language model embeddings are pooled. Every
    protein maps into the same 960-dim space regardless of length, so proteins
    and organisms are directly comparable.

``"legacy"``
    What ``refactored/ESMC_embedding.py`` did: ``np.mean(embeddings, axis=1)``
    on the squeezed ``(L, 960)`` matrix averages over the **feature** axis,
    producing one scalar per residue, i.e. a vector of length ``L``. Those are
    then zero-padded to the longest protein in the dataset. It is kept for
    comparison, but it has two properties worth knowing before using it:
    feature ``i`` means "mean activation at residue position i", so the
    features are position-indexed rather than semantic; and because the padding
    width is the longest protein *in this run*, the feature space changes when
    the dataset changes, which is where the project's magic 3863 and 4199
    column counts came from.

Both are implemented; ``"residue"`` is the default because it is the one whose
features mean the same thing across runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from SmartAMR.config import DEFAULT_ESMC_MODEL, Settings
from SmartAMR.reproduce.hosts import HOSTS, Host
from SmartAMR.reproduce.proteomes import iter_fasta
from SmartAMR.resources import ResourceLimits

#: ESM-C 300M's embedding width.
EMBEDDING_DIM = 960

#: Residues past this are dropped from a protein before embedding. Attention is
#: quadratic in length, so one pathological sequence can blow the GPU budget
#: that hundreds of normal proteins fit inside. Bacterial and phage proteins are
#: almost all under 1500 residues.
MAX_RESIDUES = 4096

#: How often the resumable store is flushed to disk, in organisms.
CHECKPOINT_EVERY = 25


def _merged_meta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Keep a cached run's tallies instead of overwriting them with zeros.

    A re-run whose organisms are all already in the store embeds nothing, so
    its counters are all 0. Writing those over the previous run's figures would
    make the store claim it was built from no proteins at all. The cumulative
    counters therefore keep whichever value is larger, while the descriptive
    fields take the current run's answer.
    """

    merged = dict(previous)
    merged.update(current)
    for key in ("proteins_embedded", "truncated_proteins", "skipped_proteins"):
        merged[key] = max(int(current.get(key) or 0), int(previous.get(key) or 0))
    return merged

POOLING_CHOICES = ("residue", "legacy")


@dataclass
class EmbedProgress:
    """Running totals for one embedding pass."""

    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    proteins: int = 0
    truncated: int = 0
    #: Individual proteins skipped after a repeated failure (usually OOM).
    skipped_proteins: int = 0
    last_error: str = ""
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def rate(self) -> float:
        """Proteins embedded per second."""
        return self.proteins / self.elapsed if self.elapsed > 0 else 0.0

    def line(self, label: str) -> str:
        return (
            f"[{self.done + self.skipped}/{self.total}] {label} "
            f"| {self.proteins} proteins @ {self.rate:.1f}/s "
            f"| {self.elapsed / 60:.1f} min"
        )


# ---------------------------------------------------------------------------
# The resumable store
# ---------------------------------------------------------------------------
class EmbeddingStore:
    """A dict of ``key -> vector`` persisted as a single ``.npz``.

    ``.npz`` rather than HDF5 so the package needs no PyTables, and rather than
    one file per organism so a few hundred small writes do not become a few
    hundred inodes. Saving is atomic: a temporary file is renamed into place,
    so a crash mid-flush leaves the previous checkpoint intact.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.vectors: dict[str, np.ndarray] = {}
        self.meta: dict[str, Any] = {}
        if self.path.exists():
            self.load()

    def load(self) -> None:
        with np.load(self.path, allow_pickle=False) as archive:
            for key in archive.files:
                if key == "__meta__":
                    continue
                self.vectors[key] = archive[key]
        meta_path = self.path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.meta = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".npz.partial")
        # Write through an open handle, not a path: given a path whose name does
        # not end in ".npz", np.savez_compressed appends the extension itself,
        # so the file would land at "<name>.partial.npz" and the rename below
        # would have nothing to move. A handle is written verbatim.
        with open(partial, "wb") as handle:
            np.savez_compressed(handle, **self.vectors)
        partial.replace(self.path)
        if self.meta:
            self.path.with_suffix(".meta.json").write_text(
                json.dumps(self.meta, indent=2), encoding="utf-8"
            )

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]

    def add(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = np.asarray(vector, dtype=np.float32)

    def keys(self) -> Iterable[str]:
        return self.vectors.keys()

    def width(self) -> int:
        """Length of the stored vectors, 0 when the store is empty."""
        for vector in self.vectors.values():
            return int(vector.shape[0])
        return 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(limits: ResourceLimits, model_name: str = DEFAULT_ESMC_MODEL) -> tuple[Any, str]:
    """Load ESM-C on the device the caps allow. Returns ``(model, device)``.

    The weights are cast to bfloat16 on GPU: it halves the ~1.3 GB of fp32
    parameters, which is what lets the model fit in a capped share of a card
    that another process is already using. bfloat16 keeps fp32's exponent
    range, so this does not introduce the overflow risk that fp16 would, and
    the embeddings are averaged immediately afterwards.
    """

    import torch
    from esm.models.esmc import ESMC

    device = limits.device("cuda")
    model = ESMC.from_pretrained(model_name).to(device).eval()
    if device.startswith("cuda"):
        model = model.to(torch.bfloat16)
    return model, device


def _logits_config() -> Any:
    from esm.sdk.api import LogitsConfig

    return LogitsConfig(sequence=True, return_embeddings=True)


def embed_protein(model: Any, sequence: str, config: Any, retry_on_oom: bool = True) -> np.ndarray:
    """Per-residue embedding matrix of one protein, as ``(L, 960)`` float32.

    Attention memory grows with the square of the length, so within a capped
    share of a card a long protein can fail where its neighbours fit. On an
    out-of-memory error the cached blocks are released and the protein is tried
    once more; that alone recovers most cases, because the failure is usually
    fragmentation rather than a genuine shortfall. If it fails again the caller
    sees the error and can skip just that protein.
    """

    import torch
    from esm.sdk.api import ESMProtein, ESMProteinError

    try:
        with torch.no_grad():
            output = model.logits(model.encode(ESMProtein(sequence=sequence)), config)
    except torch.OutOfMemoryError:
        if not retry_on_oom:
            raise
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model.logits(model.encode(ESMProtein(sequence=sequence)), config)

    if isinstance(output, ESMProteinError):
        raise RuntimeError(f"ESM-C failed on a {len(sequence)}-residue protein: {output}")

    # (1, L+2, 960) including the BOS/EOS tokens the tokenizer adds; drop them
    # so a short protein's vector is not dominated by two special positions.
    embeddings = output.embeddings.squeeze(0)
    if embeddings.shape[0] > 2:
        embeddings = embeddings[1:-1]
    return embeddings.to(torch.float32).cpu().numpy()


def pool_protein(matrix: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce a ``(L, 960)`` matrix to one vector, per the chosen pooling."""
    if pooling == "residue":
        return matrix.mean(axis=0)
    if pooling == "legacy":
        return matrix.mean(axis=1)
    raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLING_CHOICES}")


def pool_organism(vectors: Sequence[np.ndarray], pooling: str) -> np.ndarray:
    """Average an organism's per-protein vectors into one vector.

    Under ``residue`` pooling every vector is already 960 long, so this is a
    plain mean. Under ``legacy`` pooling the vectors have the length of their
    protein, so they are zero-padded to the longest one first -- which is what
    the original script's ``pad_arrays_to_max_shape`` did.
    """

    if not vectors:
        raise ValueError("no proteins to pool")
    if pooling == "residue":
        return np.mean(np.stack(vectors, axis=0), axis=0)

    width = max(vector.shape[0] for vector in vectors)
    padded = [np.pad(vector, (0, width - vector.shape[0])) for vector in vectors]
    return np.mean(np.stack(padded, axis=0), axis=0)


def embed_proteome(
    model: Any,
    config: Any,
    path: Path,
    pooling: str,
    limits: ResourceLimits,
    progress: EmbedProgress,
    max_residues: int = MAX_RESIDUES,
) -> np.ndarray | None:
    """Embed every protein in a FASTA file and pool them into one vector.

    Returns ``None`` when the file holds no usable sequence.
    """

    vectors: list[np.ndarray] = []
    for _header, sequence_lines in iter_fasta(Path(path)):
        sequence = "".join(sequence_lines).strip().upper()
        if not sequence:
            continue
        if len(sequence) > max_residues:
            sequence = sequence[:max_residues]
            progress.truncated += 1
        try:
            matrix = embed_protein(model, sequence, config)
        except Exception as error:  # noqa: BLE001
            # Skip the one protein rather than lose the organism: a proteome of
            # dozens of proteins still yields a usable mean without a single
            # pathological sequence, and the count is reported.
            progress.skipped_proteins += 1
            progress.last_error = f"{type(error).__name__} on a {len(sequence)}-residue protein"
            continue
        vectors.append(pool_protein(matrix, pooling))
        progress.proteins += 1

    if not vectors:
        return None
    # Guard after a whole proteome rather than after each protein: the check
    # reads /proc and a phage can hold hundreds of proteins.
    limits.memory_guard(f"embedding {Path(path).name}")
    return pool_organism(vectors, pooling)


# ---------------------------------------------------------------------------
# Passes over the two kinds of input
# ---------------------------------------------------------------------------
def embed_phages(
    settings: Settings,
    limits: ResourceLimits,
    selection: "Any",
    pooling: str = "residue",
    model: Any | None = None,
    config: Any | None = None,
    log: Callable[[str], None] = print,
    force: bool = False,
) -> EmbeddingStore:
    """Embed every selected phage's proteome, resuming an interrupted run."""
    store = EmbeddingStore(settings.phage_embedding_store)
    if force:
        store.vectors.clear()

    rows = [
        (str(row.phage_id), Path(str(row.proteome_path)))
        for row in selection.itertuples()
        if str(row.proteome_path)
    ]
    progress = EmbedProgress(total=len(rows))

    if model is None:
        model, device = load_model(limits)
        log(f"ESM-C loaded on {device}")
    config = config if config is not None else _logits_config()

    for phage_id, path in rows:
        if phage_id in store and not force:
            progress.skipped += 1
            continue
        if not path.exists():
            progress.failed += 1
            log(f"  missing proteome for {phage_id}: {path}")
            continue
        try:
            vector = embed_proteome(model, config, path, pooling, limits, progress)
        except Exception as error:  # noqa: BLE001 - one phage must not sink the pass
            progress.failed += 1
            log(f"  failed {phage_id}: {type(error).__name__}: {error}")
            continue

        if vector is None:
            progress.failed += 1
            log(f"  no sequences in {path.name}")
            continue

        store.add(phage_id, vector)
        progress.done += 1

        if progress.done % CHECKPOINT_EVERY == 0:
            store.save()
            log(progress.line(f"phages (checkpoint, {len(store)} stored)"))

    store.meta = _merged_meta(
        store.meta,
        {
            "pooling": pooling,
            "model": DEFAULT_ESMC_MODEL,
            "dim": store.width(),
            "proteins_embedded": progress.proteins,
            "truncated_proteins": progress.truncated,
            "skipped_proteins": progress.skipped_proteins,
            "failed": progress.failed,
        },
    )
    store.save()
    log(progress.line(f"phages done ({len(store)} stored, {progress.failed} failed)"))
    return store


def embed_bacteria(
    settings: Settings,
    limits: ResourceLimits,
    hosts: Iterable[Host] = HOSTS,
    pooling: str = "residue",
    use_filtered: bool = True,
    model: Any | None = None,
    config: Any | None = None,
    log: Callable[[str], None] = print,
    force: bool = False,
) -> EmbeddingStore:
    """Embed each bacterial host's proteome into one vector.

    With ``use_filtered`` the resistance-protein subset is embedded, which is
    what the original pipeline's filter stage was for; otherwise the whole
    downloaded proteome is used.
    """

    store = EmbeddingStore(settings.bacteria_embedding_store)
    if force:
        store.vectors.clear()

    hosts = list(hosts)
    progress = EmbedProgress(total=len(hosts))

    if model is None:
        model, device = load_model(limits)
        log(f"ESM-C loaded on {device}")
    config = config if config is not None else _logits_config()

    for host in hosts:
        key = host.taxon_key
        if key in store and not force:
            progress.skipped += 1
            continue

        if use_filtered:
            path = settings.filtered_proteomes / f"proteome_taxon_{host.uniprot_taxon}.selected.fasta"
        else:
            path = settings.uniprot_proteomes / f"proteome_taxon_{host.uniprot_taxon}.fasta"

        if not path.exists():
            progress.failed += 1
            log(f"  missing proteome for {host.inphared_name}: {path}")
            continue

        try:
            vector = embed_proteome(model, config, path, pooling, limits, progress)
        except Exception as error:  # noqa: BLE001
            progress.failed += 1
            log(f"  failed {host.inphared_name}: {type(error).__name__}: {error}")
            continue

        if vector is None:
            progress.failed += 1
            continue

        store.add(key, vector)
        progress.done += 1
        log(progress.line(f"{host.inphared_name}"))

    store.meta = _merged_meta(
        store.meta,
        {
            "pooling": pooling,
            "model": DEFAULT_ESMC_MODEL,
            "dim": store.width(),
            "filtered": use_filtered,
            "proteins_embedded": progress.proteins,
            "truncated_proteins": progress.truncated,
            "skipped_proteins": progress.skipped_proteins,
            "failed": progress.failed,
        },
    )
    store.save()
    log(progress.line(f"bacteria done ({len(store)} stored, {progress.failed} failed)"))
    return store
