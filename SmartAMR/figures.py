"""Write the project's graphs to image files.

The same chart builders the dashboard uses, saved as PNG or SVG for a paper, a
slide deck or the README. Anything whose input is missing is skipped with a
note rather than failing the whole export.

Saving images needs ``vl-convert-python``::

    pip install -e ".[figures]"      # or: uv sync --extra figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from SmartAMR import stats
from SmartAMR.config import Settings
from SmartAMR.viz import charts
from SmartAMR.viz.theme import LIGHT, Theme

DEFAULT_FORMATS: Tuple[str, ...] = ("png",)
DEFAULT_SCALE = 2


def build_figures(
    settings: Settings,
    theme: Theme = LIGHT,
    include_projection: bool = True,
    projection_method: str = "pca",
    max_points: int = 500,
    include_experiments: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Build every figure whose inputs are present.

    Returns
    -------
    tuple
        The charts keyed by file stem, and a note per figure that was skipped.
    """

    figures: Dict[str, Any] = {}
    skipped: List[str] = []

    # --- Corpus ---------------------------------------------------------
    if settings.inphared_metadata.exists():
        metadata = stats.read_metadata(settings.inphared_metadata)

        figures["hosts-top15"] = charts.bar_ranking(
            stats.host_counts(metadata, drop_unspecified=True),
            label="Host",
            value="Phages",
            theme=theme,
            title="Phages per isolation host",
            subtitle="Top 15 hosts in the inphared metadata",
            top_n=15,
            value_title="Phages",
        )

        selected = stats.selected_metadata(metadata, settings.selected_viruses)

        if not selected.empty:
            if stats.GENOME_LENGTH_COLUMN in selected.columns:
                figures["phage-genome-length"] = charts.histogram(
                    selected,
                    value=stats.GENOME_LENGTH_COLUMN,
                    theme=theme,
                    title="Genome length of the selected phages",
                    subtitle=f"{len(selected):,} selected phage records",
                    value_title="Genome length (bp)",
                )

            if stats.GC_COLUMN in selected.columns:
                figures["phage-gc-content"] = charts.histogram(
                    selected,
                    value=stats.GC_COLUMN,
                    theme=theme,
                    title="GC content of the selected phages",
                    subtitle="Share of G and C bases",
                    value_title="molGC (%)",
                )

            if stats.CDS_COLUMN in selected.columns:
                figures["phage-cds-count"] = charts.histogram(
                    selected,
                    value=stats.CDS_COLUMN,
                    theme=theme,
                    title="Coding sequences per selected phage",
                    subtitle="How many proteins each phage contributes to its embedding",
                    value_title="Number of CDS",
                )

            families = stats.family_counts(selected)
            if not families.empty:
                figures["phage-families"] = charts.bar_ranking(
                    families,
                    label="Family",
                    value="Phages",
                    theme=theme,
                    title="Families of the selected phages",
                    top_n=10,
                    value_title="Phages",
                )
        else:
            skipped.append("phage distributions: no selected phage found in the metadata")
    else:
        skipped.append(f"host and phage charts: no inphared metadata at {settings.inphared_metadata}")

    per_bacteria = stats.phages_per_bacteria(settings.selected_viruses)
    if not per_bacteria.empty:
        figures["phages-per-bacteria"] = charts.bar_ranking(
            per_bacteria,
            label="Bacteria",
            value="Phages",
            theme=theme,
            title="Phages selected per bacteria",
            subtitle="Distinct accessions drawn for each host",
            top_n=None,
            value_title="Phages",
        )
    else:
        skipped.append("phages per bacteria: run 'smartamr select-viruses'")

    # --- Features -------------------------------------------------------
    if settings.features.exists():
        import pandas as pd

        from SmartAMR.embedding.features import load_features

        features = load_features(settings.features)
        positives = int(features["label"].astype(bool).sum())

        figures["class-balance"] = charts.bar_ranking(
            pd.DataFrame(
                {
                    "Pair": ["Interacting", "Not interacting"],
                    "Count": [positives, len(features) - positives],
                }
            ),
            label="Pair",
            value="Count",
            theme=theme,
            title="Class balance of the feature table",
            top_n=None,
            value_title="Pairs",
        )

        if include_projection:
            try:
                from SmartAMR.embedding.projection import projection_frame
                from SmartAMR.models.classifiers import prepare_multilabel

                X, y = prepare_multilabel(features)
                if len(X) > max_points:
                    sample = X.sample(max_points, random_state=42)
                    X, y = sample, y.loc[sample.index]

                frame = projection_frame(X, labels=y, method=projection_method)
                figures[f"embedding-{projection_method}"] = charts.projection_facets(
                    frame,
                    theme=theme,
                    title=f"{projection_method.upper()} of the phage embeddings",
                    subtitle=f"{len(frame):,} interacting pairs, one panel per host",
                )
            except (ImportError, ValueError) as error:
                skipped.append(f"embedding projection: {error}")

        if include_experiments:
            # Refits every estimator, so this one is opt-in.
            from SmartAMR.models.evaluation import run_experiment, silhouette_analysis

            try:
                experiment = run_experiment(features, task="binary")
            except (ImportError, ValueError) as error:
                experiment = None
                skipped.append(f"experiment figures: {error}")

            if experiment is not None and not experiment.roc.empty:
                figures["roc-curves"] = charts.roc_curves(experiment.roc, theme=theme)
                figures["det-curves"] = charts.det_curves(experiment.det, theme=theme)

                best = max(experiment.auc, key=experiment.auc.get) if experiment.auc else None
                if best:
                    figures["score-separation"] = charts.score_separation(
                        experiment.probabilities,
                        theme=theme,
                        model=best,
                        subtitle=f"{best}: how far apart it puts the two classes",
                    )

            try:
                from SmartAMR.models.classifiers import prepare_binary

                X, y = prepare_binary(features)
                samples, averages = silhouette_analysis(X, y.astype(str))
                figures["silhouette"] = charts.silhouette_plot(samples, averages, theme=theme)
            except (ImportError, ValueError) as error:
                skipped.append(f"silhouette: {error}")
    else:
        skipped.append(f"feature charts: no feature table at {settings.features}")

    # --- Models ---------------------------------------------------------
    scores = stats.model_scores(settings.models_dir)
    if not scores.empty:
        figures["model-accuracy"] = charts.model_comparison(
            scores,
            theme=theme,
            subtitle="Each task has its own test set, so the two panels are not comparable",
        )

        best = scores.sort_values("Accuracy", ascending=False).iloc[0]
        from SmartAMR.models.classifiers import load_model

        model = load_model(best["Path"])
        metrics = model.metrics or {}

        if metrics.get("confusion_matrix"):
            labels = metrics.get("labels") or []
            if model.task == "binary":
                labels = ["Does not infect", "Infects"]
            figures["confusion-matrix"] = charts.confusion_heatmap(
                metrics["confusion_matrix"],
                labels=labels,
                theme=theme,
                title=f"Confusion matrix - {model.model_name} ({model.task})",
                subtitle="Share of each true class, so rows sum to 100%",
            )

        f1 = stats.per_class_f1(model)
        if not f1.empty:
            figures["model-f1"] = charts.metric_bars(
                f1,
                label="Class",
                value="F1",
                theme=theme,
                title=f"F1 per class - {model.model_name}",
                value_title="F1 score",
            )
    else:
        skipped.append("model charts: no trained model found, run 'smartamr train'")

    return figures, skipped


def save_figures(
    settings: Settings,
    outdir: str | Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    theme: Theme = LIGHT,
    scale: int = DEFAULT_SCALE,
    include_projection: bool = True,
    projection_method: str = "pca",
    include_experiments: bool = False,
) -> Tuple[List[Path], List[str]]:
    """Build the figures and write them to ``outdir``.

    Returns the written paths and the notes for anything skipped.
    """

    if any(fmt == "png" for fmt in formats) or any(fmt == "svg" for fmt in formats):
        try:
            import vl_convert  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Saving figures needs vl-convert-python. Install it with: "
                'pip install -e ".[figures]" (or: uv sync --extra figures)'
            ) from error

    figures, skipped = build_figures(
        settings,
        theme=theme,
        include_projection=include_projection,
        projection_method=projection_method,
        include_experiments=include_experiments,
    )

    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for name, chart in figures.items():
        for fmt in formats:
            path = target / f"{name}.{fmt}"
            if fmt == "png":
                chart.save(str(path), scale_factor=scale)
            else:
                chart.save(str(path))
            written.append(path)

    return written, skipped
