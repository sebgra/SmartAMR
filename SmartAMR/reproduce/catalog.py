"""Choose the phages to work with, and pair them with the bacterial hosts.

``refactored/phages_selection.py`` sampled up to 100 accessions per host from
the INPHARED metadata. That selection ignored whether the phage's proteome was
actually on disk, and in practice most of it was not: of the 900 accessions it
picked, 62 had a ``.faa`` file locally. The rest would be silently dropped much
later, during feature building, after the expensive embedding stage had already
run.

So selection here is constrained to phages whose proteome exists, which makes
the chosen set the set that can actually be embedded.

Two stages, matching the two files the original wrote:

:func:`select_phages`
    Up to ``per_host`` phages per host, sampled with a fixed seed, restricted
    to accessions with a local proteome.

:func:`build_pairs`
    The cross product of the selected phages with all nine hosts. This is where
    negatives come from: a phage paired with a host its metadata does not list
    is a non-interacting pair. The original scripts wrote only the matched
    pairs and then computed the label by re-checking the match, which can only
    ever return True -- the labels had to come from a cross product, and the
    shape of the project's own feature table (360 phages x 9 hosts = 3240 rows,
    361 positive) confirms it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from SmartAMR.config import (
    INPHARED_ACCESSION_COLUMN,
    INPHARED_HOST_COLUMN,
    Settings,
)
from SmartAMR.reproduce.hosts import HOSTS, Host

#: The original scripts' sample size per host, and their seed.
DEFAULT_PER_HOST = 100
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SelectionReport:
    """What :func:`select_phages` found, per host, for honest reporting."""

    host: str
    in_metadata: int
    with_proteome: int
    selected: int


def read_metadata(settings: Settings) -> pd.DataFrame:
    """Load the INPHARED metadata table.

    Only the columns the pipeline uses are kept: the file has ~30 columns and
    28k rows, and the genome statistics are read separately by the stats module.
    """

    path = Path(settings.inphared_metadata)
    if not path.exists():
        raise FileNotFoundError(
            f"INPHARED metadata not found at {path}. It ships with the repository "
            f"as {path.name}; set SMARTAMR_DATA_DIR if it lives elsewhere."
        )
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = {INPHARED_ACCESSION_COLUMN, INPHARED_HOST_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected column(s): {sorted(missing)}")
    return frame


def available_proteomes(genomes_dir: Path) -> dict[str, Path]:
    """Map accession -> proteome path for every phage proteome on disk.

    The INPHARED genome database stores one directory per accession holding
    ``<accession>.faa`` among other files. Both that layout and a flat
    directory of ``.faa`` files are accepted, because
    ``refactored/scripts/flatten_inphared.sh`` produced the flat form.
    """

    genomes_dir = Path(genomes_dir)
    if not genomes_dir.is_dir():
        return {}

    found: dict[str, Path] = {}
    # Flat layout first so that, if both exist, the per-accession directory
    # (checked second, written by the authoritative extraction) wins.
    for path in genomes_dir.glob("*.faa"):
        found[path.stem] = path
    for path in genomes_dir.glob("*/*.faa"):
        found[path.stem] = path
    return found


def host_matches(frame: pd.DataFrame, host: Host) -> pd.Series:
    """Boolean mask of metadata rows whose isolation host matches ``host``.

    Substring, case-insensitive, as the original scripts did: the host column
    is free text, so "Escherichia coli C" and "Escherichia coli K-12" both
    count as E. coli.
    """

    column = frame[INPHARED_HOST_COLUMN].fillna("")
    return column.str.contains(host.inphared_name, case=False, na=False, regex=False)


def select_phages(
    settings: Settings,
    per_host: int = DEFAULT_PER_HOST,
    seed: int = DEFAULT_SEED,
    require_proteome: bool = True,
) -> tuple[pd.DataFrame, list[SelectionReport]]:
    """Sample up to ``per_host`` phages for each host.

    Returns the selection as a frame of ``phage_id, host, taxon_id,
    proteome_path`` plus a per-host report of how the numbers were arrived at.

    With ``require_proteome`` (the default) only accessions with a proteome on
    disk can be chosen, so every selected phage is embeddable.
    """

    frame = read_metadata(settings)
    proteomes = available_proteomes(settings.phage_genomes) if require_proteome else {}

    if require_proteome and not proteomes:
        raise FileNotFoundError(
            f"No phage proteomes (.faa) found under {settings.phage_genomes}. "
            "Extract the INPHARED genome database there, or pass "
            "require_proteome=False to select without them."
        )

    rows: list[dict] = []
    reports: list[SelectionReport] = []
    # One seeded generator for the whole selection, so the result depends on
    # the host order (fixed in HOSTS) and not on which hosts happen to be run.
    rng = random.Random(seed)

    for host in HOSTS:
        matched = frame.loc[host_matches(frame, host), INPHARED_ACCESSION_COLUMN]
        accessions = sorted(set(matched.dropna().astype(str)))

        if require_proteome:
            candidates = [accession for accession in accessions if accession in proteomes]
        else:
            candidates = accessions

        chosen = sorted(rng.sample(candidates, min(per_host, len(candidates))))
        for accession in chosen:
            rows.append(
                {
                    "phage_id": accession,
                    "host": host.inphared_name,
                    "taxon_id": host.taxon_key,
                    "proteome_path": str(proteomes.get(accession, "")),
                }
            )

        reports.append(
            SelectionReport(
                host=host.inphared_name,
                in_metadata=len(accessions),
                with_proteome=len(candidates),
                selected=len(chosen),
            )
        )

    selection = pd.DataFrame(rows, columns=["phage_id", "host", "taxon_id", "proteome_path"])
    return selection, reports


def build_pairs(settings: Settings, selection: pd.DataFrame) -> pd.DataFrame:
    """Cross the selected phages with all nine hosts and label each pair.

    ``label`` is True when the phage's INPHARED metadata lists that host, so
    each phage contributes one or a few positives and eight or so negatives.
    """

    frame = read_metadata(settings)
    phage_ids = sorted(set(selection["phage_id"].astype(str)))

    # Pre-compute, per host, the set of accessions it matches. Doing this once
    # per host beats re-querying the 28k-row table once per pair.
    matched_by_host: dict[str, set[str]] = {}
    for host in HOSTS:
        accessions = frame.loc[host_matches(frame, host), INPHARED_ACCESSION_COLUMN]
        matched_by_host[host.inphared_name] = set(accessions.dropna().astype(str))

    rows = [
        {
            "phage_id": phage_id,
            "bacteria_id": host.inphared_name,
            "taxon_id": host.taxon_key,
            "label": phage_id in matched_by_host[host.inphared_name],
        }
        for phage_id in phage_ids
        for host in HOSTS
    ]
    return pd.DataFrame(rows, columns=["phage_id", "bacteria_id", "taxon_id", "label"])


def write_catalog(
    settings: Settings,
    per_host: int = DEFAULT_PER_HOST,
    seed: int = DEFAULT_SEED,
    require_proteome: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[SelectionReport]]:
    """Run the selection, write both tables, and return them with the report.

    Writes ``selected_phages.csv`` and ``phage_bacteria_pairs.csv`` into the
    output directory's ``catalog/`` folder.
    """

    selection, reports = select_phages(
        settings, per_host=per_host, seed=seed, require_proteome=require_proteome
    )
    pairs = build_pairs(settings, selection)

    settings.ensure(settings.catalog_dir)
    selection.to_csv(settings.selected_phages, index=False)
    pairs.to_csv(settings.phage_pairs, index=False)
    return selection, pairs, reports


def format_report(reports: Iterable[SelectionReport]) -> str:
    """Render the per-host selection report as a fixed-width table."""
    lines = [f"{'host':<28}{'metadata':>10}{'proteome':>10}{'selected':>10}"]
    totals = [0, 0, 0]
    for report in reports:
        lines.append(
            f"{report.host:<28}{report.in_metadata:>10}"
            f"{report.with_proteome:>10}{report.selected:>10}"
        )
        totals[0] += report.in_metadata
        totals[1] += report.with_proteome
        totals[2] += report.selected
    lines.append(f"{'total':<28}{totals[0]:>10}{totals[1]:>10}{totals[2]:>10}")
    return "\n".join(lines)
