"""Run the reproduction stages in order, and record what happened.

The stages are independent and each writes its output to disk, so this is a
thin orchestrator: it resolves the resource caps once, runs the stages that
were asked for, and writes a manifest the app reads to show the provenance of
what it is displaying.

Stage order and what each needs:

===========  ====================================  =========================
``catalog``  INPHARED metadata + phage proteomes   selected_phages.csv, pairs
``fetch``    network                               uniprot_proteomes/
``filter``   uniprot_proteomes/ + protein list     uniprot_proteomes_filtered/
``embed``    both proteome sets, GPU               *_embeddings.npz
``features`` embeddings + pairs                    reproduced_features.parquet
``train``    features                              *.joblib, metrics.json
===========  ====================================  =========================
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from SmartAMR.config import Settings
from SmartAMR.reproduce import catalog, embed, features, proteomes, train
from SmartAMR.reproduce.hosts import HOSTS, corrections
from SmartAMR.resources import ResourceLimits, limits

#: Stage names, in dependency order.
STAGES = ("catalog", "fetch", "filter", "embed", "features", "train")


@dataclass
class RunRecord:
    """What one reproduction run did, written to ``run_manifest.json``."""

    started_at: str = ""
    finished_at: str = ""
    stages: list[str] = field(default_factory=list)
    resource_summary: str = ""
    pooling: str = "residue"
    per_host: int = catalog.DEFAULT_PER_HOST
    notes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": self.stages,
            "resource_summary": self.resource_summary,
            "pooling": self.pooling,
            "per_host": self.per_host,
            "notes": self.notes,
            "counts": self.counts,
            "failures": self.failures,
            "python": platform.python_version(),
        }


def run(
    settings: Settings | None = None,
    stages: Sequence[str] = STAGES,
    pooling: str = "residue",
    per_host: int = catalog.DEFAULT_PER_HOST,
    use_filtered: bool = True,
    group_by_phage: bool = True,
    fraction: float | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> RunRecord:
    """Run the requested stages and return the run record.

    ``fraction`` overrides the resource share for this run; by default the
    ``SMARTAMR_RESOURCE_FRACTION`` environment variable or 50% applies.
    """

    settings = settings or Settings()
    caps: ResourceLimits = limits(fraction)
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")

    record = RunRecord(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        stages=list(stages),
        resource_summary=caps.summary(),
        pooling=pooling,
        per_host=per_host,
    )
    log(caps.summary())

    for host in corrections():
        note = (
            f"{host.inphared_name}: using UniProt taxon {host.uniprot_taxon} "
            f"instead of {host.legacy_uniprot_taxon} ({host.note})"
        )
        record.notes.append(note)
        log(f"note: {note}")

    selection: pd.DataFrame | None = None

    # --- catalog -------------------------------------------------------
    if "catalog" in stages:
        log("\n== catalog: selecting phages")
        selection, pairs, reports = catalog.write_catalog(settings, per_host=per_host)
        log(catalog.format_report(reports))
        record.counts["phages_selected"] = int(selection["phage_id"].nunique())
        record.counts["pairs"] = len(pairs)
        record.counts["positive_pairs"] = int(pairs["label"].sum())

    # --- fetch ---------------------------------------------------------
    if "fetch" in stages:
        log("\n== fetch: downloading bacterial proteomes from UniProt")
        reports = proteomes.download_proteomes(settings, force=force)
        log(proteomes.format_reports(reports, ""))
        record.counts["bacteria_proteomes"] = sum(1 for report in reports if report.ok)
        record.failures.extend(
            f"fetch {report.host}: {report.error}" for report in reports if report.error
        )

    # --- filter --------------------------------------------------------
    if "filter" in stages:
        log("\n== filter: subsetting proteomes to resistance-related proteins")
        reports = proteomes.filter_proteomes(settings)
        log(proteomes.format_reports(reports, ""))
        record.counts["filtered_proteins"] = sum(report.records for report in reports)
        record.failures.extend(
            f"filter {report.host}: {report.error}" for report in reports if report.error
        )

    # --- embed ---------------------------------------------------------
    if "embed" in stages:
        log("\n== embed: ESM-C embeddings")
        if selection is None:
            if not Path(settings.selected_phages).exists():
                raise FileNotFoundError(
                    f"{settings.selected_phages} is missing; run the catalog stage first."
                )
            selection = pd.read_csv(settings.selected_phages)

        model, device = embed.load_model(caps)
        log(f"ESM-C on {device}")
        config = embed._logits_config()

        bacteria_store = embed.embed_bacteria(
            settings,
            caps,
            pooling=pooling,
            use_filtered=use_filtered,
            model=model,
            config=config,
            log=log,
            force=force,
        )
        phage_store = embed.embed_phages(
            settings,
            caps,
            selection,
            pooling=pooling,
            model=model,
            config=config,
            log=log,
            force=force,
        )
        record.counts["phage_embeddings"] = len(phage_store)
        record.counts["bacteria_embeddings"] = len(bacteria_store)
        record.counts["embedding_dim"] = phage_store.width()

    # --- features ------------------------------------------------------
    if "features" in stages:
        log("\n== features: pairing embeddings")
        frame, report = features.build_features(settings)
        path = features.write_features(settings, frame)
        log(report.summary())
        log(f"wrote {path}")
        record.counts["feature_rows"] = report.rows
        record.counts["feature_positives"] = report.positives
        record.counts["feature_columns"] = report.phage_dim + report.bacteria_dim

    # --- train ---------------------------------------------------------
    if "train" in stages:
        log("\n== train: fitting classifiers")
        frame = features.load_features(settings)

        binary_task, binary_models = train.train_binary(
            frame, threads=caps.threads, group_by_phage=group_by_phage, log=log
        )
        multiclass_task, multiclass_models = train.train_multiclass(
            frame, threads=caps.threads, group_by_phage=group_by_phage, log=log
        )

        written = train.save_models(settings, train.BINARY_TASK, binary_models)
        written += train.save_models(settings, train.MULTICLASS_TASK, multiclass_models)
        metrics_path = train.save_metrics(
            settings,
            [binary_task, multiclass_task],
            extra={"resource_summary": caps.summary(), "pooling": pooling},
        )
        log(f"wrote {len(written)} models and {metrics_path}")

        record.counts["models_trained"] = len(written)
        for task in (binary_task, multiclass_task):
            best = task.best()
            if best:
                record.notes.append(
                    f"best {task.task}: {best.model} (F1 {best.f1_macro:.3f})"
                )
            record.failures.extend(
                f"train {task.task}/{model.model}: {model.error}"
                for model in task.models
                if model.error
            )

    record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    settings.ensure(Path(settings.run_manifest).parent)
    Path(settings.run_manifest).write_text(
        json.dumps(record.to_json(), indent=2), encoding="utf-8"
    )
    log(f"\nmanifest: {settings.run_manifest}")
    return record


def load_manifest(settings: Settings) -> dict:
    """Read the last run's manifest, or ``{}`` when nothing has run."""
    path = Path(settings.run_manifest)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def pipeline_state(settings: Settings) -> list[dict]:
    """Per-stage readiness, for the app's status panel."""
    checks = [
        ("catalog", settings.selected_phages, "selected phages"),
        ("fetch", settings.uniprot_proteomes, "bacterial proteomes"),
        ("filter", settings.filtered_proteomes, "filtered proteomes"),
        ("embed", settings.phage_embedding_store, "phage embeddings"),
        ("features", settings.reproduced_features, "feature table"),
        ("train", settings.metrics_path, "trained models"),
    ]
    state = []
    for stage, path, label in checks:
        path = Path(path)
        exists = any(path.iterdir()) if path.is_dir() else path.exists()
        state.append(
            {
                "stage": stage,
                "label": label,
                "path": str(path),
                "done": exists,
                "hosts": len(HOSTS),
            }
        )
    return state
