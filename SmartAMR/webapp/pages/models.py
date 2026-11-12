"""Models: how the four estimators scored on each of the two tasks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from SmartAMR.viz import charts
from SmartAMR.reproduce.train import BINARY_TASK, MULTICLASS_TASK
from SmartAMR.webapp import data, ui

TASK_LABELS = {
    BINARY_TASK: "Does this phage infect this host?",
    MULTICLASS_TASK: "Which host does this phage infect?",
}

TASK_NOTES = {
    BINARY_TASK: (
        "Both proteome embeddings concatenated, binary target. The negative class "
        "is downsampled to the size of the positive class before splitting, so "
        "accuracy here is accuracy on a balanced test set."
    ),
    MULTICLASS_TASK: (
        "Interacting pairs only, **phage features alone**, host as the target. The "
        "bacterial block is excluded on purpose: it is constant per host, so "
        "including it would hand the model the answer."
    ),
}


def render() -> None:
    ui.header("Models", "Four estimators, two tasks, one held-out split each.")

    metrics = data.metrics()
    if not metrics:
        ui.missing_stage(
            "The models",
            "smartamr reproduce --stages train",
            "Training needs the feature table.",
        )
        return

    st.caption(
        f"Trained {metrics.get('generated_at', 'at an unrecorded time')} · "
        f"test size {metrics.get('test_size', '?')} · seed {metrics.get('random_state', '?')}"
    )

    tabs = st.tabs([TASK_LABELS[BINARY_TASK], TASK_LABELS[MULTICLASS_TASK]])
    with tabs[0]:
        _render_task(BINARY_TASK)
    with tabs[1]:
        _render_task(MULTICLASS_TASK)


def _render_task(task: str) -> None:
    theme = ui.theme()
    entry = data.task_metrics(task)
    frame = data.metrics_frame(task)

    if frame.empty:
        st.info("No estimator finished for this task.")
        return

    st.markdown(TASK_NOTES[task])
    st.caption(f"{entry.get('rows', '?')} rows · {entry.get('features', '?')} features · split: {entry.get('split', '?')}")

    best = frame.loc[frame["f1_macro"].idxmax()]
    ui.tile_row(
        [
            ui.Tile("Best model", str(best["model"]), f"macro F1 {ui.number(best['f1_macro'])}"),
            ui.Tile("Accuracy", ui.number(best["accuracy"])),
            ui.Tile("Balanced accuracy", ui.number(best["balanced_accuracy"])),
            ui.Tile(
                "ROC AUC",
                ui.number(best["roc_auc"]) if pd.notna(best["roc_auc"]) else "—",
                "one-vs-rest" if task == MULTICLASS_TASK else "",
            ),
        ]
    )

    st.divider()

    # --- comparison -----------------------------------------------------
    ui.section("Macro F1 by estimator", "Macro F1 weights every class equally.")
    ranked = frame.sort_values("f1_macro", ascending=False)
    ui.chart(
        charts.metric_bars(
            ranked,
            label="model",
            value="f1_macro",
            theme=theme,
            title="Macro F1",
            subtitle="Higher is better; 0.5 is the line a coin-flip reaches on two balanced classes",
            value_title="Macro F1",
        ),
        key=f"f1-{task}",
    )

    display = ranked.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy",
            "f1_macro": "Macro F1",
            "roc_auc": "ROC AUC",
            "fit_seconds": "Fit (s)",
        }
    ).round(4)
    ui.table_view(display, label="Show every score")

    # --- ROC -------------------------------------------------------------
    roc = data.roc_frame(task)
    if not roc.empty:
        ui.section(
            "ROC curves",
            "True positives against false positives across every threshold.",
        )
        ui.chart(
            charts.roc_curves(
                roc,
                theme=theme,
                title="ROC curves",
                subtitle="The diagonal is chance; a curve below it means the scores are inverted",
            ),
            key=f"roc-{task}",
        )
        auc_table = frame[["model", "roc_auc"]].dropna().sort_values("roc_auc", ascending=False)
        ui.table_view(auc_table.round(4), label="Show the AUC values")

        below_chance = auc_table[auc_table["roc_auc"] < 0.5]
        if len(below_chance):
            names = ", ".join(below_chance["model"])
            st.warning(
                f"**{names}** scored below 0.5 AUC, which is worse than chance. That "
                "usually means the estimator's probabilities are sign-flipped relative "
                "to the positive class rather than that it learned nothing — the same "
                "pattern the project's earlier AdaBoost figure showed."
            )

    # --- confusion -------------------------------------------------------
    st.divider()
    ui.section("Confusion matrix", "Rows are the true class, columns the prediction.")

    models = [entry_model for entry_model in entry.get("models", []) if not entry_model.get("error")]
    names = [entry_model["model"] for entry_model in models]
    default = int(ranked.index.get_loc(ranked.index[0])) if len(ranked) else 0
    chosen = st.selectbox(
        "Estimator",
        names,
        index=names.index(str(best["model"])) if str(best["model"]) in names else default,
        key=f"cm-select-{task}",
    )
    selected = next(entry_model for entry_model in models if entry_model["model"] == chosen)

    matrix = selected.get("confusion") or []
    labels = selected.get("classes") or []
    if matrix and labels:
        pretty = ["Does not infect", "Infects"] if task == BINARY_TASK else labels
        ui.chart(
            charts.confusion_heatmap(
                matrix,
                labels=pretty,
                theme=theme,
                title=f"Confusion matrix — {chosen}",
                subtitle="Shares of each true class, so rows sum to 100%",
            ),
            key=f"cm-{task}",
        )
        ui.table_view(
            pd.DataFrame(matrix, index=pretty, columns=pretty).reset_index(names="true class"),
            label="Show the raw counts",
        )

    per_class = selected.get("per_class_f1") or {}
    if per_class:
        ui.section("F1 per class", f"Where {chosen} is strong and where it is not.")
        class_frame = pd.DataFrame(
            [{"class": key, "f1": value} for key, value in per_class.items()]
        )
        if task == BINARY_TASK:
            class_frame["class"] = class_frame["class"].map(
                {"0": "Does not infect", "1": "Infects"}
            ).fillna(class_frame["class"])
        ui.chart(
            charts.metric_bars(
                class_frame.sort_values("f1", ascending=False),
                label="class",
                value="f1",
                theme=theme,
                title=f"F1 per class — {chosen}",
                subtitle="A class with few test examples gives a noisy score",
                value_title="F1",
            ),
            key=f"perclass-{task}",
        )
        ui.table_view(class_frame.round(4), label="Show the per-class values")
