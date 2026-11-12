"""Pipeline: what has been built, from what, and under which resource caps.

The provenance page. It answers "where did these numbers come from" and, on a
fresh checkout, "what do I run to get them".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from SmartAMR.reproduce.hosts import corrections
from SmartAMR.reproduce.runner import pipeline_state
from SmartAMR.webapp import data, ui

#: Command that produces each stage, shown next to a stage that is missing.
STAGE_COMMANDS = {
    "catalog": "smartamr reproduce --stages catalog",
    "fetch": "smartamr reproduce --stages fetch",
    "filter": "smartamr reproduce --stages filter",
    "embed": "smartamr reproduce --stages embed",
    "features": "smartamr reproduce --stages features",
    "train": "smartamr reproduce --stages train",
}


def render() -> None:
    ui.header(
        "Pipeline",
        "What has been built, from what, and under which resource caps. "
        "Every figure in this app traces back to a run recorded here.",
    )

    summary = data.feature_summary()
    metrics = data.metrics()
    manifest = data.manifest()

    # --- headline numbers ----------------------------------------------
    best_line = "—"
    best_note = "not trained"
    binary = data.task_metrics("binary")
    scored = [m for m in binary.get("models", []) if not m.get("error")]
    if scored:
        best = max(scored, key=lambda entry: entry.get("f1_macro") or 0)
        best_line = ui.number(best.get("f1_macro"))
        best_note = f"{best.get('model')} · macro F1"

    ui.tile_row(
        [
            ui.Tile("Phages", ui.thousands(summary.get("phages")), "with a real proteome"),
            ui.Tile("Hosts", ui.thousands(summary.get("hosts")), "bacterial species"),
            ui.Tile(
                "Labelled pairs",
                ui.thousands(summary.get("rows")),
                f"{ui.thousands(summary.get('positives'))} interacting"
                if summary.get("positives") is not None
                else "",
            ),
            ui.Tile("Best binary model", best_line, best_note),
        ]
    )

    st.divider()

    left, right = st.columns([1.15, 1], gap="large")

    # --- pipeline state ------------------------------------------------
    with left:
        ui.section(
            "Pipeline",
            "Each stage writes its output to disk; the next stage reads it.",
        )
        state = pipeline_state(data.settings())
        for entry in state:
            columns = st.columns([0.32, 0.68])
            columns[0].markdown(ui.pill(entry["stage"], entry["done"]), unsafe_allow_html=True)
            if entry["done"]:
                columns[1].markdown(
                    f'<div class="amr-caption">{entry["label"]}</div>', unsafe_allow_html=True
                )
            else:
                columns[1].markdown(
                    f'<div class="amr-caption">{entry["label"]} — not built</div>',
                    unsafe_allow_html=True,
                )

        pending = [entry["stage"] for entry in state if not entry["done"]]
        if pending:
            st.caption(
                "Build everything that is missing with `smartamr reproduce`, or one "
                f"stage at a time: `{STAGE_COMMANDS[pending[0]]}`"
            )

    # --- provenance ----------------------------------------------------
    with right:
        ui.section("Last run", "Recorded by the pipeline, not inferred here.")
        if manifest:
            rows = [
                ("Finished", manifest.get("finished_at", "—")),
                ("Stages", ", ".join(manifest.get("stages", [])) or "—"),
                ("Pooling", manifest.get("pooling", "—")),
                ("Resources", manifest.get("resource_summary", "—")),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=["", "value"]),
                hide_index=True,
                width='stretch',
            )
            if manifest.get("failures"):
                with st.expander(f"{len(manifest['failures'])} failure(s) in that run"):
                    for failure in manifest["failures"]:
                        st.write(f"- {failure}")
        else:
            st.caption("No run manifest yet.")

        meta = data.embedding_meta("phage")
        if meta:
            ui.section("Embeddings")
            st.caption(
                f"{meta.get('model', 'ESM-C')} · {meta.get('dim', '?')} dimensions · "
                f"{meta.get('pooling', '?')} pooling · "
                f"{ui.thousands(meta.get('proteins_embedded'))} proteins embedded"
            )
            notes = []
            if meta.get("truncated_proteins"):
                notes.append(
                    f"{meta['truncated_proteins']} protein(s) truncated to the length limit"
                )
            if meta.get("skipped_proteins"):
                notes.append(f"{meta['skipped_proteins']} protein(s) skipped after an error")
            if meta.get("failed"):
                notes.append(f"{meta['failed']} phage(s) produced no embedding")
            ui.caption_list(notes)

    # --- what to know about the data -----------------------------------
    st.divider()
    ui.section("Things worth knowing about this dataset")

    notes = [
        "Phage selection is restricted to accessions whose proteome is on disk, so "
        "every selected phage is actually embeddable. Sampling is seeded.",
        "Negative pairs come from crossing each phage with all nine hosts: a pair is "
        "positive only when the phage's INPHARED metadata lists that host.",
        "The isolation-host column of the INPHARED metadata is free text — its own "
        "name warns about inconsistent values — so hosts are matched by substring.",
    ]
    for host in corrections():
        notes.append(
            f"**{host.inphared_name}** uses UniProt taxon {host.uniprot_taxon}, not "
            f"{host.legacy_uniprot_taxon} as the original scripts did: the latter "
            "returns 8 proteins, which is not a proteome."
        )
    if metrics:
        notes.append(
            "Train/test splits are grouped by phage, so no phage appears on both "
            "sides. A phage's embedding is identical across its nine rows, so a "
            "row-wise split would let the model memorise the phage."
        )
    for note in notes:
        st.markdown(f"- {note}")
