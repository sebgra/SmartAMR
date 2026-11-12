"""Look up AMR genes, their coordinates and their sequences in a genome.

Moved from ``SmartAMR/utils.py``, same behaviour. Each function works on the
per-genome files downloaded by :mod:`SmartAMR.data.scrapper`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from Bio import SeqIO

from SmartAMR.config import CONTIG_SUFFIX, FEATURES_SUFFIX, METADATA_FILENAME, SP_GENES_SUFFIX

# Kept as module level names for backwards compatibility with the notebooks.
METADATA = METADATA_FILENAME
SP_GENES_ROOT = SP_GENES_SUFFIX
FEATURES_ROOT = FEATURES_SUFFIX

DEFAULT_GENE_TYPE = "Antibiotic Resistance"

GenePosition = Tuple[str, int, int]


def extract_AMR_genes_names(  # noqa: N802 - name kept for compatibility
    file_path: str | Path | None = None,
    gene_type: str = DEFAULT_GENE_TYPE,
    genome_id: str | None = None,
) -> List[str]:
    """Names of the genes of a given property in a ``.spgene.tab`` file.

    Equivalent to ``cut -f 8 XXXX.PATRIC.spgene.tab``.
    """

    data_path: Path = Path(file_path) / f"{genome_id}{SP_GENES_SUFFIX}"

    df: pd.DataFrame = pd.read_csv(data_path, sep="\t")
    df = df[df["property"] == gene_type]

    return df["gene"].tolist()


def extract_genes_positions(
    file_path: str | Path | None = None,
    genome_id: str | None = None,
    gene_name: str | List[str] | None = None,
) -> Union[GenePosition, Dict[str, GenePosition]]:
    """Genomic coordinates of one gene or a list of genes.

    Equivalent to grepping ``XXXX.PATRIC.features.tab``. Returns a single
    ``(accession, start, end)`` tuple for a string, or a dict keyed by gene
    name for a list.
    """

    data_path: Path = Path(file_path) / f"{genome_id}{FEATURES_SUFFIX}"
    df: pd.DataFrame = pd.read_csv(data_path, sep="\t")

    if isinstance(gene_name, str):
        value = df[df["gene"] == gene_name]
        if value.empty:
            raise KeyError(f"Gene {gene_name!r} not found in {data_path}")
        return (
            str(value["accession"].values[0]),
            int(value["start"].values[0]),
            int(value["end"].values[0]),
        )

    if isinstance(gene_name, list):
        value = df[df["gene"].isin(gene_name)]
        return {
            str(row["gene"]): (str(row["accession"]), int(row["start"]), int(row["end"]))
            for _, row in value.iterrows()
        }

    raise TypeError("gene_name must be a string or a list of strings")


def extract_gene_sequence(
    file_path: str | Path | None = None,
    genome_id: str | None = None,
    gene_name: str | List[str] | None = None,
) -> Dict[str, Tuple[str, str]]:
    """Sequences of one gene or a list of genes, read from the ``.fna`` contigs.

    Returns a ``{gene: (contig, sequence)}`` dictionary. The contigs are read
    once and indexed, instead of re-parsing the genome per gene.
    """

    genome_file: Path = Path(file_path) / f"{genome_id}{CONTIG_SUFFIX}"
    contigs = {record.id: record.seq for record in SeqIO.parse(genome_file, "fasta")}

    if isinstance(gene_name, str):
        contig, start, stop = extract_genes_positions(
            file_path=file_path, genome_id=genome_id, gene_name=gene_name
        )
        if contig not in contigs:
            raise KeyError(f"Contig {contig!r} not found in {genome_file}")
        return {gene_name: (contig, str(contigs[contig][start:stop]).upper())}

    if isinstance(gene_name, list):
        positions = extract_genes_positions(
            file_path=file_path, genome_id=genome_id, gene_name=gene_name
        )
        sequences: Dict[str, Tuple[str, str]] = {}
        for gene, (contig, start, stop) in positions.items():
            if contig not in contigs:
                print(f"Warning: contig {contig!r} of gene {gene!r} not in {genome_file} - skipped")
                continue
            sequences[gene] = (contig, str(contigs[contig][start:stop]).upper())
        return sequences

    raise TypeError("gene_name must be a string or a list of strings")
