"""FASTA reading helpers shared by the CLI, the dashboard and the web backend.

This replaces the hand-rolled parsers that lived in ``web_interface/main.py``.
The behaviour is unchanged: records are keyed by the full header line (minus the
leading ``>``) and sequence lines are concatenated without separators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

Record = Tuple[str, str]


def iter_fasta(path: str | Path) -> Iterator[Record]:
    """Yield ``(name, sequence)`` for every record in a FASTA file.

    Parameters
    ----------
    path
        Path to a FASTA file.

    Yields
    ------
    tuple of (str, str)
        The header line without its ``>`` prefix, and the concatenated sequence.
    """

    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    name = ""
    chunks: List[str] = []

    with fasta_path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if chunks:
                    yield name, "".join(chunks)
                name = line[1:]
                chunks = []
            elif line:
                chunks.append(line)

    if chunks:
        yield name, "".join(chunks)


def read_fasta(path: str | Path) -> Dict[str, str]:
    """Load a FASTA file into a ``{name: sequence}`` dictionary."""
    return dict(iter_fasta(path))


def retrieve_sequences(path: str | Path, name: str) -> List[str]:
    """Return every sequence whose header matches ``name`` exactly."""
    return [sequence for header, sequence in iter_fasta(path) if header == name]


def retrieve_sequence(path: str | Path, name: str) -> str | None:
    """Return the first sequence whose header matches ``name``, or None."""
    matches = retrieve_sequences(path, name)
    return matches[0] if matches else None


def names(path: str | Path) -> List[str]:
    """List the record names of a FASTA file, in file order."""
    return [header for header, _ in iter_fasta(path)]


def write_fasta(records: Iterable[Record], path: str | Path, line_width: int = 60) -> Path:
    """Write ``(name, sequence)`` pairs to a FASTA file and return its path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            if line_width and line_width > 0:
                for start in range(0, len(sequence), line_width):
                    handle.write(sequence[start : start + line_width] + "\n")
            else:
                handle.write(sequence + "\n")

    return out_path


def parse_fasta_text(text: str) -> List[Record]:
    """Parse FASTA content already held in memory (e.g. an uploaded file).

    A plain sequence with no header is accepted and returned under the name
    ``"unnamed"``, which is what the web form allows users to paste.
    """

    records: List[Record] = []
    name = ""
    chunks: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if chunks:
                records.append((name, "".join(chunks)))
            name = line[1:]
            chunks = []
        else:
            chunks.append(line)

    if chunks:
        records.append((name or "unnamed", "".join(chunks)))

    return records
