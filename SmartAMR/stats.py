"""Summary tables describing the project's corpus and its trained models.

Each function returns a tidy DataFrame, so the dashboard and the figure export
draw exactly the same numbers from exactly the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from SmartAMR.config import (
    INPHARED_ACCESSION_COLUMN,
    INPHARED_HOST_COLUMN,
    Settings,
)

GENOME_LENGTH_COLUMN = "Genome Length (bp)"
GC_COLUMN = "molGC (%)"
CDS_COLUMN = "Number CDS"
FAMILY_COLUMN = "Family"

#: Metadata columns the project graphs read, kept small so the 11 MB table
#: does not have to be loaded in full.
METADATA_COLUMNS = (
    INPHARED_ACCESSION_COLUMN,
    INPHARED_HOST_COLUMN,
    GENOME_LENGTH_COLUMN,
    GC_COLUMN,
    CDS_COLUMN,
    FAMILY_COLUMN,
)


def read_metadata(path: str | Path, columns: tuple[str, ...] = METADATA_COLUMNS) -> pd.DataFrame:
    """Read the inphared metadata, only the columns the graphs need."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"inphared metadata not found at {metadata_path}. "
            "Set SMARTAMR_DATA_DIR or pass --metadata."
        )

    available = pd.read_csv(metadata_path, sep="\t", nrows=0).columns
    usecols = [column for column in columns if column in available]

    return pd.read_csv(metadata_path, sep="\t", usecols=usecols)


def host_counts(metadata: pd.DataFrame, drop_unspecified: bool = False) -> pd.DataFrame:
    """Phage count per isolation host."""
    counts = (
        metadata[INPHARED_HOST_COLUMN]
        .fillna("Unspecified")
        .value_counts()
        .rename_axis("Host")
        .reset_index(name="Phages")
    )
    if drop_unspecified:
        counts = counts[counts["Host"] != "Unspecified"]
    return counts


def selected_accessions(selected_viruses: str | Path) -> Dict[str, List[str]]:
    """Accessions selected per bacteria, read from the selection CSV."""
    path = Path(selected_viruses)
    if not path.exists():
        return {}

    table = pd.read_csv(path)
    return {column: table[column].dropna().astype(str).tolist() for column in table.columns}


def phages_per_bacteria(selected_viruses: str | Path) -> pd.DataFrame:
    """Distinct phages selected for each bacteria."""
    selection = selected_accessions(selected_viruses)

    rows = [
        {"Bacteria": bacteria, "Phages": len(set(accessions))}
        for bacteria, accessions in selection.items()
    ]
    return pd.DataFrame(rows).sort_values("Phages", ascending=False).reset_index(drop=True)


def selected_metadata(metadata: pd.DataFrame, selected_viruses: str | Path) -> pd.DataFrame:
    """Metadata rows of the phages the project selected.

    A ``Bacteria`` column records which selection each row came from; a phage
    chosen for two bacteria appears once per bacteria.
    """

    selection = selected_accessions(selected_viruses)
    if not selection:
        return metadata.assign(Bacteria="all")

    frames = []
    for bacteria, accessions in selection.items():
        subset = metadata[metadata[INPHARED_ACCESSION_COLUMN].isin(set(accessions))].copy()
        subset["Bacteria"] = bacteria
        frames.append(subset)

    if not frames:
        return metadata.iloc[0:0].assign(Bacteria=pd.Series(dtype=str))

    return pd.concat(frames, ignore_index=True)


def family_counts(metadata: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Phage count per taxonomic family."""
    if FAMILY_COLUMN not in metadata.columns:
        return pd.DataFrame(columns=["Family", "Phages"])

    counts = (
        metadata[FAMILY_COLUMN]
        .fillna("Unclassified")
        .value_counts()
        .rename_axis("Family")
        .reset_index(name="Phages")
    )
    return counts.head(top_n)


def corpus_summary(settings: Settings) -> Dict[str, int]:
    """Headline counts of the corpus, skipping anything not on disk."""
    summary: Dict[str, int] = {}

    if settings.bacteria_table.exists():
        summary["Bacteria"] = len(pd.read_csv(settings.bacteria_table))

    if settings.selected_proteins.exists():
        from SmartAMR.data.proteins import read_protein_names

        summary["Resistance proteins"] = len(read_protein_names(settings.selected_proteins))

    selection = selected_accessions(settings.selected_viruses)
    if selection:
        summary["Selected phages"] = len({a for values in selection.values() for a in values})

    if settings.features.exists():
        features = pd.read_csv(settings.features, usecols=["label"])
        summary["Feature pairs"] = len(features)
        summary["Interacting pairs"] = int(features["label"].astype(bool).sum())

    return summary


def model_scores(models_dir: str | Path) -> pd.DataFrame:
    """Accuracy of every saved model, with its task and estimator."""
    directory = Path(models_dir)
    if not directory.exists():
        return pd.DataFrame(columns=["Model", "Task", "Accuracy", "Test pairs", "Path"])

    from SmartAMR.models.classifiers import load_model

    rows = []
    for path in sorted(directory.glob("*.joblib")) + sorted(directory.glob("*.pkl")):
        try:
            model = load_model(path)
        except Exception as error:  # noqa: BLE001 - a bad file must not sink the page
            print(f"Warning: could not read {path.name}: {error}")
            continue

        metrics = model.metrics or {}
        if "accuracy" not in metrics:
            continue

        rows.append(
            {
                "Model": model.model_name,
                "Task": model.task,
                "Accuracy": float(metrics["accuracy"]),
                "Test pairs": int(metrics.get("n_test", 0)),
                "Path": str(path),
            }
        )

    return pd.DataFrame(rows)


def per_class_f1(model) -> pd.DataFrame:
    """F1 per class from a trained model's stored report."""
    report = (model.metrics or {}).get("report", {})

    rows = [
        {"Class": str(name), "F1": float(values["f1-score"])}
        for name, values in report.items()
        if isinstance(values, dict) and "f1-score" in values and name != "accuracy"
    ]
    return pd.DataFrame(rows)
