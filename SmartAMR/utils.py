"""Deprecated: moved to :mod:`SmartAMR.genomics.amr`.

Kept as a re-export so existing imports keep working. ``SmartAMR.genomics.amr``
is the maintained copy -- same behaviour, with the ``.gff``/``.spgene.tab``
parsing documented and typed.
"""

from __future__ import annotations

import warnings

from SmartAMR.genomics.amr import (
    extract_AMR_genes_names,
    extract_gene_sequence,
    extract_genes_positions,
)

warnings.warn(
    "SmartAMR.utils is deprecated: import from SmartAMR.genomics.amr instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "extract_AMR_genes_names",
    "extract_gene_sequence",
    "extract_genes_positions",
]
