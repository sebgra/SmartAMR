"""Per-genome lookups: AMR genes, their coordinates and sequences, codon usage."""

from __future__ import annotations

from SmartAMR.genomics.amr import (
    extract_AMR_genes_names,
    extract_gene_sequence,
    extract_genes_positions,
)
from SmartAMR.genomics.codon import CodonTable, fetch_codon_table

__all__ = [
    "CodonTable",
    "extract_AMR_genes_names",
    "extract_gene_sequence",
    "extract_genes_positions",
    "fetch_codon_table",
]
