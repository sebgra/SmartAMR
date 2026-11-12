"""Fetch the bacterial proteomes and subset them to resistance proteins.

Two stages the original kept as a Python script plus a Bash/awk pair:

:func:`download_proteomes`
    What ``refactored/bacteria_proteome_downloader.py`` did -- stream each
    host's proteome out of the UniProt REST API -- with the taxon-id correction
    recorded in :mod:`SmartAMR.reproduce.hosts`, a retry, and a guard that
    refuses to overwrite a good file with a short one.

:func:`filter_proteomes`
    What ``scripts/filter_bacterian_proteomes.sh`` and
    ``scripts/extract_proteins.awk`` did -- keep only records whose header
    mentions one of the protein types in ``selected_bacterian_proteins.txt``.
    The awk implementation is correct (its ``keep == 1`` rule re-runs on each
    sequence line, so whole records are emitted); it is reimplemented here only
    to drop the hardcoded ``/mnt/d/...`` paths and to change two behaviours
    deliberately:

    * awk's ``$0 ~ name`` treats each name as a **regular expression**, so a
      name containing ``.``, ``(`` or ``+`` would match more than intended.
      Matching here is plain substring containment.
    * awk's match is case-sensitive, so ``OmpA`` would miss a header spelling
      it ``ompA``. Matching here is case-insensitive, which keeps more of the
      intended records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from SmartAMR.config import Settings
from SmartAMR.reproduce.hosts import HOSTS, Host

#: UniProt's streaming endpoint, as the original script used.
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

#: A proteome smaller than this is treated as a failed download, not a result.
#: The original used 100 bytes, which let the 3.7 KB eight-protein E. coli file
#: through as a success; a real bacterial proteome is hundreds of KB at least.
MIN_PROTEOME_BYTES = 20_000

#: A proteome with fewer records than this is reported as suspicious.
MIN_PROTEOME_RECORDS = 200

DOWNLOAD_TIMEOUT_SECONDS = 300
DOWNLOAD_ATTEMPTS = 3


@dataclass
class ProteomeReport:
    """Outcome for one host's proteome."""

    host: str
    taxon: int
    path: Path | None
    records: int = 0
    bytes_written: int = 0
    skipped: bool = False
    error: str = ""
    #: True for a filtered subset, where a small record count is the point and
    #: the :data:`MIN_PROTEOME_RECORDS` floor must not be applied.
    is_subset: bool = False

    @property
    def ok(self) -> bool:
        return self.error == "" and self.records > 0

    @property
    def suspicious(self) -> bool:
        """A *download* that succeeded but looks too small to be a proteome."""
        if self.is_subset:
            return False
        return self.ok and self.records < MIN_PROTEOME_RECORDS


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------
def iter_fasta(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(header, sequence_lines)`` for each record in a FASTA file.

    Written by hand rather than with Bio.SeqIO so that the filter stage has no
    dependency beyond the standard library, and so a malformed record cannot
    abort the whole file.
    """

    header: str | None = None
    sequence: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, sequence
                header = line.rstrip("\n")
                sequence = []
            elif header is not None:
                stripped = line.strip()
                if stripped:
                    sequence.append(stripped)
    if header is not None:
        yield header, sequence


def count_records(path: Path) -> int:
    """Number of FASTA records in a file, 0 when it is missing."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def read_protein_names(path: Path) -> list[str]:
    """Protein-type names to keep, one per line, blanks and ``#`` ignored."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Protein list not found at {path}; it ships as "
            "config/selected_bacterian_proteins.txt"
        )
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = raw.strip()
        if name and not name.startswith("#"):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _stream_proteome(taxon: int, destination: Path) -> int:
    """Stream one proteome to ``destination``. Returns bytes written."""
    import requests  # imported here so the module loads without it

    params = {
        "query": f"organism_id:{taxon}",
        "format": "fasta",
        "compressed": "false",
    }
    # Write to a temporary sibling and move it into place, so an interrupted
    # download can never be mistaken for a complete proteome by a later run.
    partial = destination.with_suffix(destination.suffix + ".partial")
    written = 0
    with requests.get(
        UNIPROT_STREAM_URL, params=params, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)

    if written < MIN_PROTEOME_BYTES:
        partial.unlink(missing_ok=True)
        return written

    partial.replace(destination)
    return written


def download_proteomes(
    settings: Settings,
    hosts: Iterable[Host] = HOSTS,
    force: bool = False,
) -> list[ProteomeReport]:
    """Download each host's proteome from UniProt into ``uniprot_proteomes``.

    Existing, plausible files are kept unless ``force``; a download is retried
    up to :data:`DOWNLOAD_ATTEMPTS` times with a short backoff.
    """

    settings.ensure(settings.uniprot_proteomes)
    reports: list[ProteomeReport] = []

    for host in hosts:
        destination = settings.uniprot_proteomes / f"proteome_taxon_{host.uniprot_taxon}.fasta"
        report = ProteomeReport(host=host.inphared_name, taxon=host.uniprot_taxon, path=destination)

        if destination.exists() and destination.stat().st_size >= MIN_PROTEOME_BYTES and not force:
            report.records = count_records(destination)
            report.bytes_written = destination.stat().st_size
            report.skipped = True
            reports.append(report)
            continue

        last_error = ""
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                written = _stream_proteome(host.uniprot_taxon, destination)
            except Exception as error:  # noqa: BLE001 - network, DNS, HTTP
                last_error = f"{type(error).__name__}: {error}"
            else:
                if destination.exists():
                    report.bytes_written = written
                    report.records = count_records(destination)
                    last_error = ""
                    break
                last_error = (
                    f"only {written} bytes returned for organism_id:{host.uniprot_taxon} "
                    f"(under the {MIN_PROTEOME_BYTES} byte floor)"
                )
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2 * attempt)

        report.error = last_error
        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
def header_matches(header: str, names: Iterable[str]) -> bool:
    """True when a protein-type name appears in the FASTA header.

    Case-insensitive substring matching over the whole header, which is how
    the awk script tested it: UniProt headers carry the protein name in a
    free-text description, so the name cannot be read from a fixed field.
    """

    lowered = header.lower()
    return any(name.lower() in lowered for name in names)


def filter_proteome(source: Path, destination: Path, names: Iterable[str]) -> int:
    """Write the records of ``source`` whose header matches. Returns the count."""
    names = list(names)
    kept = 0
    partial = destination.with_suffix(destination.suffix + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        for header, sequence in iter_fasta(source):
            if not sequence or not header_matches(header, names):
                continue
            handle.write(f"{header}\n")
            # Keep the original line wrapping; it carries no meaning but makes
            # the output diffable against the source.
            for line in sequence:
                handle.write(f"{line}\n")
            kept += 1
    partial.replace(destination)
    return kept


def filter_proteomes(
    settings: Settings,
    hosts: Iterable[Host] = HOSTS,
) -> list[ProteomeReport]:
    """Subset every downloaded proteome to the resistance-related proteins."""
    names = read_protein_names(settings.selected_proteins)
    settings.ensure(settings.filtered_proteomes)

    reports: list[ProteomeReport] = []
    for host in hosts:
        source = settings.uniprot_proteomes / f"proteome_taxon_{host.uniprot_taxon}.fasta"
        destination = (
            settings.filtered_proteomes / f"proteome_taxon_{host.uniprot_taxon}.selected.fasta"
        )
        report = ProteomeReport(
            host=host.inphared_name,
            taxon=host.uniprot_taxon,
            path=destination,
            is_subset=True,
        )

        if not source.exists():
            report.error = f"no downloaded proteome at {source.name}"
            reports.append(report)
            continue

        try:
            report.records = filter_proteome(source, destination, names)
            report.bytes_written = destination.stat().st_size
        except OSError as error:
            report.error = f"{type(error).__name__}: {error}"
        reports.append(report)

    return reports


def format_reports(reports: Iterable[ProteomeReport], title: str) -> str:
    """Render proteome reports as a fixed-width table."""
    lines = [title, f"{'host':<28}{'taxon':>9}{'records':>9}{'size':>10}  status"]
    for report in reports:
        size = f"{report.bytes_written / 1024:.0f} KB" if report.bytes_written else "-"
        if report.error:
            status = f"FAILED: {report.error}"
        elif report.suspicious:
            status = f"suspicious (<{MIN_PROTEOME_RECORDS} records)"
        elif report.skipped:
            status = "cached"
        else:
            status = "ok"
        lines.append(
            f"{report.host:<28}{report.taxon:>9}{report.records:>9}{size:>10}  {status}"
        )
    return "\n".join(lines)
