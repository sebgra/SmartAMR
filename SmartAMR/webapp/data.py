"""Cached loaders for everything the app displays.

Every loader is wrapped in ``st.cache_data`` and keyed on the file's path and
modification time, so re-running the pipeline in another terminal invalidates
the cache without a restart. Each returns an empty value rather than raising
when its artefact is missing: the pages decide how to present an absent stage,
and a missing file is an expected state, not an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from SmartAMR.config import Settings
from SmartAMR.reproduce import features as feature_tools
from SmartAMR.reproduce.hosts import HOSTS

#: How long a cached artefact is trusted before it is re-read, in seconds.
CACHE_TTL = 300


def settings() -> Settings:
    """The resolved project settings. Cheap, so not cached."""
    return Settings()


def _stamp(path: Path | str) -> float:
    """Modification time of a path, 0 when it does not exist.

    Used as a cache key so an artefact rewritten by a pipeline run in another
    process is picked up on the next page render.
    """

    path = Path(path)
    try:
        if path.is_dir():
            return max((child.stat().st_mtime for child in path.iterdir()), default=0.0)
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# JSON artefacts
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _read_json(path: str, _stamp_key: float) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def metrics() -> dict:
    """Training metrics for every task and estimator."""
    path = settings().metrics_path
    return _read_json(str(path), _stamp(path))


def manifest() -> dict:
    """What the last reproduction run did."""
    path = settings().run_manifest
    return _read_json(str(path), _stamp(path))


def embedding_meta(kind: str) -> dict:
    """Sidecar metadata of an embedding store (``"phage"`` or ``"bacteria"``)."""
    store = (
        settings().phage_embedding_store if kind == "phage" else settings().bacteria_embedding_store
    )
    path = Path(store).with_suffix(".meta.json")
    return _read_json(str(path), _stamp(path))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _read_csv(path: str, _stamp_key: float) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading feature table...")
def _read_parquet(path: str, _stamp_key: float) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def selected_phages() -> pd.DataFrame:
    """The catalog's chosen phages."""
    path = settings().selected_phages
    return _read_csv(str(path), _stamp(path))


def pairs() -> pd.DataFrame:
    """The phage-host pair list with interaction labels."""
    path = settings().phage_pairs
    return _read_csv(str(path), _stamp(path))


def features() -> pd.DataFrame:
    """The full feature table. Large -- prefer :func:`feature_summary`."""
    path = settings().reproduced_features
    return _read_parquet(str(path), _stamp(path))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _read_feature_columns(path: str, _stamp_key: float) -> list[str]:
    """Column names of the feature table, read from the Parquet footer.

    The prediction page needs the names of the two embedding blocks, not the
    data. Reading the schema costs a few KB where loading the frame costs tens
    of megabytes of float columns the page never looks at.
    """

    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001 - fall back to a full read
        frame = _read_parquet(path, _stamp_key)
        return list(frame.columns)


def feature_columns() -> list[str]:
    """Column names of the feature table, without loading its data."""
    path = settings().reproduced_features
    if not Path(path).exists():
        return []
    return _read_feature_columns(str(path), _stamp(path))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _summarise_features(path: str, _stamp_key: float) -> dict:
    """Shape and class balance without holding the whole frame in the page."""
    frame = _read_parquet(path, _stamp_key)
    if frame.empty:
        return {}
    phage_dim = len(feature_tools.phage_columns(frame))
    bacteria_dim = len(feature_tools.bacteria_columns(frame))
    return {
        "rows": int(len(frame)),
        "positives": int(frame["label"].sum()),
        "phages": int(frame["phage_id"].nunique()),
        "hosts": int(frame["bacteria_id"].nunique()),
        "phage_dim": phage_dim,
        "bacteria_dim": bacteria_dim,
        "per_host": (
            frame.groupby("bacteria_id")["label"].agg(["sum", "count"]).reset_index().to_dict("list")
        ),
    }


def feature_summary() -> dict:
    """Row counts, class balance and per-host totals of the feature table."""
    path = settings().reproduced_features
    return _summarise_features(str(path), _stamp(path))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def inphared_stats(path: str, _stamp_key: float, accessions: tuple[str, ...]) -> pd.DataFrame:
    """Genome statistics for the selected phages, from the INPHARED metadata."""
    from SmartAMR.config import INPHARED_ACCESSION_COLUMN

    wanted = {
        INPHARED_ACCESSION_COLUMN: "phage_id",
        "Genome Length (bp)": "genome_length",
        "molGC (%)": "gc_content",
        "Number CDS": "cds_count",
        "Family": "family",
        "Host": "host_taxon",
    }
    try:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    except (OSError, ValueError):
        return pd.DataFrame()

    available = {source: target for source, target in wanted.items() if source in frame.columns}
    frame = frame[list(available)].rename(columns=available)
    frame["phage_id"] = frame["phage_id"].astype(str)
    if accessions:
        frame = frame[frame["phage_id"].isin(set(accessions))]
    return frame.reset_index(drop=True)


def phage_stats() -> pd.DataFrame:
    """Genome statistics restricted to the phages in the catalog."""
    catalog = selected_phages()
    if catalog.empty:
        return pd.DataFrame()
    path = settings().inphared_metadata
    accessions = tuple(sorted(catalog["phage_id"].astype(str)))
    return inphared_stats(str(path), _stamp(path), accessions)


# ---------------------------------------------------------------------------
# Embeddings and projections
# ---------------------------------------------------------------------------
@dataclass
class Stores:
    """Both embedding stores, loaded together."""

    phage: dict[str, np.ndarray]
    bacteria: dict[str, np.ndarray]

    @property
    def ready(self) -> bool:
        return bool(self.phage) and bool(self.bacteria)


@st.cache_resource(show_spinner=False)
def _load_stores(phage_path: str, bacteria_path: str, _stamp_key: str) -> Stores:
    """Load both ``.npz`` stores.

    ``cache_resource`` rather than ``cache_data``: these are read-only numpy
    arrays shared across sessions, so copying them per session would waste
    hundreds of MB for no benefit.
    """

    from SmartAMR.reproduce.embed import EmbeddingStore

    phage = EmbeddingStore(Path(phage_path)).vectors if Path(phage_path).exists() else {}
    bacteria = (
        EmbeddingStore(Path(bacteria_path)).vectors if Path(bacteria_path).exists() else {}
    )
    return Stores(phage=phage, bacteria=bacteria)


def stores() -> Stores:
    """The phage and bacteria embedding stores."""
    config = settings()
    key = f"{_stamp(config.phage_embedding_store)}:{_stamp(config.bacteria_embedding_store)}"
    return _load_stores(
        str(config.phage_embedding_store), str(config.bacteria_embedding_store), key
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner="Projecting embeddings...")
def projection(_stamp_key: str, components: int = 2) -> pd.DataFrame:
    """PCA of the phage embeddings, labelled with each phage's host.

    One row per interacting pair, so a phage listed against two hosts appears
    in both panels -- which is what the faceted scatter is showing.
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    store = stores()
    pair_table = pairs()
    if not store.phage or pair_table.empty:
        return pd.DataFrame()

    interacting = pair_table[pair_table["label"].astype(bool)].copy()
    interacting["phage_id"] = interacting["phage_id"].astype(str)
    interacting = interacting[interacting["phage_id"].isin(store.phage)]
    if interacting.empty:
        return pd.DataFrame()

    matrix = np.stack([store.phage[key] for key in interacting["phage_id"]], axis=0)
    scaled = StandardScaler().fit_transform(matrix)
    reduced = PCA(n_components=components, random_state=42).fit_transform(scaled)

    frame = interacting[["phage_id", "bacteria_id"]].reset_index(drop=True)
    for index in range(components):
        frame["xy"[index] if index < 2 else f"pc{index}"] = reduced[:, index]
    return frame


def projection_frame(components: int = 2) -> pd.DataFrame:
    config = settings()
    key = f"{_stamp(config.phage_embedding_store)}:{_stamp(config.phage_pairs)}"
    return projection(key, components)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model(path: str, _stamp_key: float) -> Any:
    import joblib

    try:
        return joblib.load(path)
    except Exception:  # noqa: BLE001 - a stale joblib must not break the page
        return None


def model(task: str, name: str) -> Any:
    """Load one trained pipeline, or ``None`` when it is absent."""
    path = settings().model_path(f"{task}-{name}")
    if not Path(path).exists():
        return None
    return _load_model(str(path), _stamp(path))


def available_models(task: str) -> list[str]:
    """Names of the trained estimators for a task, from the metrics file."""
    payload = metrics()
    for entry in payload.get("tasks", []):
        if entry.get("task") == task:
            return [
                model_entry["model"]
                for model_entry in entry.get("models", [])
                if not model_entry.get("error")
            ]
    return []


def task_metrics(task: str) -> dict:
    """The metrics entry for one task, or ``{}``."""
    for entry in metrics().get("tasks", []):
        if entry.get("task") == task:
            return entry
    return {}


def metrics_frame(task: str) -> pd.DataFrame:
    """One row per estimator: the headline scores, for charts and tables."""
    entry = task_metrics(task)
    rows = [
        {
            "model": model_entry.get("model", "?"),
            "accuracy": model_entry.get("accuracy"),
            "balanced_accuracy": model_entry.get("balanced_accuracy"),
            "f1_macro": model_entry.get("f1_macro"),
            "roc_auc": model_entry.get("roc_auc"),
            "fit_seconds": model_entry.get("fit_seconds"),
        }
        for model_entry in entry.get("models", [])
        if not model_entry.get("error")
    ]
    return pd.DataFrame(rows)


def roc_frame(task: str) -> pd.DataFrame:
    """Long-form ROC curve points for every estimator of a task.

    The series column is named ``Model`` because that is what
    ``charts.roc_curves`` facets and colours on.
    """

    entry = task_metrics(task)
    rows: list[dict] = []
    for model_entry in entry.get("models", []):
        curve = model_entry.get("roc_curve") or {}
        for false_positive, true_positive in zip(curve.get("fpr", []), curve.get("tpr", [])):
            rows.append(
                {
                    "Model": model_entry.get("model", "?"),
                    "fpr": float(false_positive),
                    "tpr": float(true_positive),
                }
            )
    return pd.DataFrame(rows)


def host_options() -> list[str]:
    """Host names, for the prediction page's selector."""
    return [host.inphared_name for host in HOSTS]
