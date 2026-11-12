"""Deprecated: moved to :mod:`SmartAMR.data.scrapper` and :mod:`SmartAMR.genomics.codon`.

This module used to hold both the BV-BRC downloaders and the Kazusa codon-table
scraper. Both moved when the package was split into subpackages, and keeping a
second copy here meant two implementations drifting apart -- the downloaders in
:mod:`SmartAMR.data.scrapper` since grew a shared ``_GenomeFileDownloader``
base, a ``download_genomes`` helper and a ``DOWNLOADERS`` registry that this
file never had.

It survives as a re-export so that ``bacteria_dataset.py`` and
``notebooks/bacteria_dataset.ipynb``, which import ``SmartAMR.scrapper``,
keep working. Import from the new locations in new code.
"""

from __future__ import annotations

import warnings

from SmartAMR.data.scrapper import (
    DOWNLOADERS,
    Downloader,
    GenomeAnnotationDownloader,
    GenomeContigDownloader,
    GenomeFeatureDownloader,
    GenomeFeatureSeqDownloader,
    GenomePathwayDownloader,
    GenomeProteinSeqDownloader,
    GenomeRNASeqDownloader,
    GenomeSpecialityGeneDownloader,
    GenomeSubSystemDownloader,
    GlobalDownloader,
    MetadataUpdater,
    download_genomes,
)
from SmartAMR.genomics.codon import CodonTable, fetch_codon_table

warnings.warn(
    "SmartAMR.scrapper is deprecated: import the downloaders from "
    "SmartAMR.data.scrapper and CodonTable from SmartAMR.genomics.codon.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DOWNLOADERS",
    "CodonTable",
    "Downloader",
    "GenomeAnnotationDownloader",
    "GenomeContigDownloader",
    "GenomeFeatureDownloader",
    "GenomeFeatureSeqDownloader",
    "GenomePathwayDownloader",
    "GenomeProteinSeqDownloader",
    "GenomeRNASeqDownloader",
    "GenomeSpecialityGeneDownloader",
    "GenomeSubSystemDownloader",
    "GlobalDownloader",
    "MetadataUpdater",
    "download_genomes",
    "fetch_codon_table",
]
