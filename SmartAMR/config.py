"""Central configuration: where data lives and which files the pipeline expects.

Every path can be overridden by an environment variable so the same code runs
on a laptop and on a compute node without editing source. The notebooks used to
hardcode absolute paths such as ``/mnt/data/data/inphared/``; those now come
from ``SMARTAMR_DATA_DIR`` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Remote endpoints (BV-BRC / PATRIC and Kazusa codon usage tables).
PATRIC_FTP_BASE_URL = "ftp://ftp.bvbrc.org"
PATRIC_FTP_AMR_METADATA_URL = f"{PATRIC_FTP_BASE_URL}/RELEASE_NOTES/PATRIC_genomes_AMR.txt"
PATRIC_FTP_GENOMES_URL = f"{PATRIC_FTP_BASE_URL}/genomes/"
PATRIC_FTP_GENOMES_METADATA_URL = f"{PATRIC_FTP_BASE_URL}/RELEASE_NOTES/genome_metadata"
KAZUSA_URL = "https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species="

# Suffixes of the per-genome files published by BV-BRC.
CONTIG_SUFFIX = ".fna"
PROTEIN_SUFFIX = ".PATRIC.faa"
FEATURES_SUFFIX = ".PATRIC.features.tab"
FEATURE_SEQ_SUFFIX = ".PATRIC.ffn"
RNA_SEQ_SUFFIX = ".PATRIC.frn"
ANNOTATION_SUFFIX = ".PATRIC.gff"
PATHWAY_SUFFIX = ".PATRIC.pathway.tab"
SP_GENES_SUFFIX = ".PATRIC.spgene.tab"
SUBSYSTEM_SUFFIX = ".PATRIC.subsystem.tab"
SELECTED_PROTEIN_SUFFIX = ".PATRIC.selected.faa"

ALL_GENOME_SUFFIXES = (
    CONTIG_SUFFIX,
    PROTEIN_SUFFIX,
    FEATURES_SUFFIX,
    FEATURE_SEQ_SUFFIX,
    RNA_SEQ_SUFFIX,
    ANNOTATION_SUFFIX,
    PATHWAY_SUFFIX,
    SP_GENES_SUFFIX,
    SUBSYSTEM_SUFFIX,
)

METADATA_FILENAME = "PATRIC_genomes_AMR.txt"
INPHARED_METADATA_FILENAME = "14Apr2025_data_excluding_refseq.tsv"
INPHARED_PROTEIN_FILENAME = "14Apr2025_vConTACT2_proteins.faa"

# Column of the inphared metadata that holds the isolation host. The name is
# verbose in the source file and the pipeline depends on it, so it lives here.
INPHARED_HOST_COLUMN = "Isolation Host (beware inconsistent and nonsense values)"
INPHARED_ACCESSION_COLUMN = "Accession"

# Embedding layout used when the paired feature vectors are built. A phage
# embedding occupies the first PHAGE_EMBEDDING_SIZE columns of the feature
# block, the bacteria embedding follows.
PHAGE_EMBEDDING_SIZE = 3863
FEATURE_METADATA_COLUMNS = ("phage_id", "bacteria_id", "label")

DEFAULT_ESMC_MODEL = "esmc_300m"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass
class Settings:
    """Resolved locations for inputs and outputs of the pipeline."""

    root: Path = field(default_factory=lambda: _env_path("SMARTAMR_ROOT", Path.cwd()))
    data_dir: Path | None = None
    config_dir: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()
        self.data_dir = Path(
            self.data_dir if self.data_dir is not None else _env_path("SMARTAMR_DATA_DIR", self.root / "data")
        ).expanduser()
        self.config_dir = Path(
            self.config_dir if self.config_dir is not None else _env_path("SMARTAMR_CONFIG_DIR", self.root / "config")
        ).expanduser()
        self.output_dir = Path(
            self.output_dir if self.output_dir is not None else _env_path("SMARTAMR_OUTPUT_DIR", self.root / "output")
        ).expanduser()

    # --- Reference tables shipped with the repository -------------------
    @property
    def bacteria_table(self) -> Path:
        """CSV mapping bacteria names to their BV-BRC genome id."""
        return self.config_dir / "bacterias.csv"

    @property
    def selected_proteins(self) -> Path:
        """Protein names expected to be involved in viral resistance."""
        return self.config_dir / "selected_bacterian_proteins.txt"

    @property
    def selected_viruses(self) -> Path:
        return self.config_dir / "selected_virus.csv"

    @property
    def virus_list(self) -> Path:
        return self.config_dir / "viruses.list"

    # --- Downloaded and generated data ---------------------------------
    def _data_file(self, filename: str) -> Path:
        """Path of a downloaded file, falling back to the project root.

        Large downloads often land next to the code rather than in ``data/``,
        which is where this project kept them. The ``data/`` copy wins; the
        root is only used when it is the one that exists.
        """

        in_data_dir = self.data_dir / filename
        if in_data_dir.exists():
            return in_data_dir

        at_root = self.root / filename
        return at_root if at_root.exists() else in_data_dir

    @property
    def inphared_metadata(self) -> Path:
        return self._data_file(INPHARED_METADATA_FILENAME)

    @property
    def amr_metadata(self) -> Path:
        return self._data_file(METADATA_FILENAME)

    @property
    def phage_protein_fasta(self) -> Path:
        """The inphared protein FASTA, the source of the phage proteomes."""
        return self._data_file(INPHARED_PROTEIN_FILENAME)

    @property
    def bacteria_proteomes(self) -> Path:
        return self.data_dir / "bacteria_proteomes"

    @property
    def phage_proteomes(self) -> Path:
        return self.data_dir / "phage_proteomes"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "embeddings"

    @property
    def phage_embeddings(self) -> Path:
        return self.embeddings_dir / "phage_embeddings.h5"

    @property
    def bacteria_embeddings(self) -> Path:
        return self.embeddings_dir / "bacteria_embeddings.h5"

    @property
    def features(self) -> Path:
        return self.output_dir / "features" / "final_features.csv"

    # --- Reproduction pipeline (SmartAMR.reproduce) ---------------------
    # These locate the raw INPHARED genome database and the artefacts the
    # reproduction stages write. They are kept apart from the properties above
    # so the demo pipeline's paths stay untouched.
    @property
    def phage_genomes(self) -> Path:
        """INPHARED genome database: one directory per accession, or flat .faa.

        Defaults to ``GenomesDB`` beside the code, which is where the project's
        own ``.gitignore`` expects the extracted archive to land.
        """
        return _env_path("SMARTAMR_GENOMES_DIR", self.root / "GenomesDB")

    @property
    def uniprot_proteomes(self) -> Path:
        """Raw bacterial proteomes as downloaded from UniProt."""
        return self.data_dir / "uniprot_proteomes"

    @property
    def filtered_proteomes(self) -> Path:
        """Bacterial proteomes subset to the resistance-related proteins."""
        return self.data_dir / "uniprot_proteomes_filtered"

    @property
    def catalog_dir(self) -> Path:
        return self.output_dir / "catalog"

    @property
    def selected_phages(self) -> Path:
        return self.catalog_dir / "selected_phages.csv"

    @property
    def phage_pairs(self) -> Path:
        return self.catalog_dir / "phage_bacteria_pairs.csv"

    @property
    def phage_embedding_store(self) -> Path:
        """Per-phage embeddings, as a resumable ``.npz`` keyed by accession."""
        return self.embeddings_dir / "phage_embeddings.npz"

    @property
    def bacteria_embedding_store(self) -> Path:
        """Per-host embeddings, as a resumable ``.npz`` keyed by taxon id."""
        return self.embeddings_dir / "bacteria_embeddings.npz"

    @property
    def reproduced_features(self) -> Path:
        """The feature table built from real ESM-C embeddings."""
        return self.output_dir / "features" / "reproduced_features.parquet"

    @property
    def metrics_path(self) -> Path:
        """Training metrics for every model and task, as JSON."""
        return self.output_dir / "models" / "metrics.json"

    @property
    def run_manifest(self) -> Path:
        """What the last reproduction run did, for the app to display."""
        return self.output_dir / "run_manifest.json"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    def model_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.joblib"

    def ensure(self, *paths: Path) -> None:
        """Create the given directories if they are missing."""
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


#: Module level default, used when no explicit Settings is passed around.
settings = Settings()
