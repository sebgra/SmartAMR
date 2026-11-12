"""Dataset: the corpus the models learned from, and its embedding space.

Absorbs what used to be a separate Embeddings page: the selection, the class
balance, the genome statistics, the feature table and the projection of the
embeddings are all answers to "what is in the corpus", so they live together.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import pandas as pd

from SmartAMR.reproduce import features as feature_tools
from SmartAMR.viz import charts
from SmartAMR.webapp import data, ui


def render() -> None:
    ui.header("Dataset", "The selected phages, their hosts, and the labelled pairs.")

    catalog = data.selected_phages()
    pairs = data.pairs()

    if catalog.empty:
        ui.missing_stage(
            "The phage catalog",
            "smartamr reproduce --stages catalog",
            "It selects the phages and builds the labelled pair list.",
        )
        return

    theme = ui.theme()

    ui.tile_row(
        [
            ui.Tile("Phages selected", ui.thousands(catalog["phage_id"].nunique())),
            ui.Tile("Pairs", ui.thousands(len(pairs)) if not pairs.empty else "—"),
            ui.Tile(
                "Interacting",
                ui.thousands(int(pairs["label"].sum())) if not pairs.empty else "—",
                ui.percent(pairs["label"].mean()) if not pairs.empty else "",
            ),
            ui.Tile("Hosts", ui.thousands(catalog["host"].nunique())),
        ]
    )

    st.divider()

    # --- coverage per host ---------------------------------------------
    ui.section(
        "Phages per host",
        "How many phages the catalog holds for each bacterial host.",
    )
    per_host = (
        catalog.groupby("host").size().reset_index(name="phages").sort_values("phages", ascending=False)
    )
    ui.chart(
        charts.bar_ranking(
            per_host,
            label="host",
            value="phages",
            theme=theme,
            title="Phages per bacterial host",
            subtitle="Selection is capped at 100 per host and limited by local proteomes",
            label_title="Host",
            value_title="Phages",
        ),
        key="per-host",
    )
    ui.table_view(per_host)

    # --- class balance --------------------------------------------------
    if not pairs.empty:
        ui.section(
            "Class balance",
            "A pair is positive when the phage's metadata lists that host. "
            "Crossing every phage with all nine hosts makes negatives the majority.",
        )
        balance = (
            pairs["label"]
            .map({True: "Interacts", False: "Does not interact"})
            .value_counts()
            .reset_index()
        )
        balance.columns = ["interaction", "pairs"]
        ui.chart(
            charts.bar_ranking(
                balance,
                label="interaction",
                value="pairs",
                theme=theme,
                title="Interacting against non-interacting pairs",
                subtitle=f"{ui.percent(pairs['label'].mean())} of pairs are positive",
                label_title="",
                value_title="Pairs",
                top_n=None,
            ),
            key="balance",
        )
        ui.table_view(balance)

        with st.expander("Positives per host"):
            by_host = (
                pairs.groupby("bacteria_id")["label"]
                .agg(interacting="sum", pairs="count")
                .reset_index()
            )
            by_host["rate"] = (by_host["interacting"] / by_host["pairs"]).map(ui.percent)
            st.dataframe(by_host, hide_index=True, width='stretch')

    # --- genome statistics ---------------------------------------------
    stats = data.phage_stats()
    if stats.empty:
        st.caption("INPHARED genome statistics are unavailable, so the distributions are skipped.")
        return

    st.divider()
    ui.section(
        "What the selected phages look like",
        "Genome statistics from the INPHARED metadata, restricted to the catalog.",
    )

    left, right = st.columns(2, gap="large")

    if "genome_length" in stats:
        lengths = stats.dropna(subset=["genome_length"]).copy()
        lengths["genome_kb"] = lengths["genome_length"] / 1000
        with left:
            ui.chart(
                charts.histogram(
                    lengths,
                    value="genome_kb",
                    theme=theme,
                    title="Genome length",
                    subtitle="Most phages sit well under 100 kb; a jumbophage tail runs further",
                    value_title="Genome length (kb)",
                ),
                key="genome-length",
            )

    if "gc_content" in stats:
        with right:
            ui.chart(
                charts.histogram(
                    stats.dropna(subset=["gc_content"]),
                    value="gc_content",
                    theme=theme,
                    title="GC content",
                    subtitle="Proportion of G and C bases, which varies with host",
                    value_title="molGC (%)",
                ),
                key="gc",
            )

    left, right = st.columns(2, gap="large")

    if "cds_count" in stats:
        with left:
            ui.chart(
                charts.histogram(
                    stats.dropna(subset=["cds_count"]),
                    value="cds_count",
                    theme=theme,
                    title="Coding sequences per phage",
                    subtitle="Each CDS is one protein that goes into the phage's embedding",
                    value_title="CDS",
                ),
                key="cds",
            )

    if "family" in stats:
        families = (
            stats["family"].fillna("Unclassified").value_counts().reset_index()
        )
        families.columns = ["family", "phages"]
        with right:
            ui.chart(
                charts.bar_ranking(
                    families,
                    label="family",
                    value="phages",
                    theme=theme,
                    title="Phage families",
                    subtitle="Taxonomic family as INPHARED records it",
                    label_title="Family",
                    value_title="Phages",
                    top_n=12,
                ),
                key="families",
            )

    ui.table_view(stats, label="Show the genome statistics table")

    st.divider()
    ui.section("The catalog")
    display = catalog[["phage_id", "host", "taxon_id"]].copy()
    st.dataframe(display, hide_index=True, width='stretch', height=320)
    st.download_button(
        "Download the catalog as CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="smartamr_selected_phages.csv",
        mime="text/csv",
    )

    _render_feature_table()
    _render_embedding_space()


def _render_feature_table() -> None:
    """Browse the built feature table.

    Carried over from the former dashboard's "Features" page. Only the
    metadata columns and a preview of the embedding blocks are shown: the table
    is 1,920 float columns wide and no one reads that in a grid.
    """

    summary = data.feature_summary()
    if not summary:
        return

    st.divider()
    ui.section(
        "The feature table",
        "One row per phage-host pair: the phage embedding, the host embedding, "
        "and the interaction label.",
    )

    ui.tile_row(
        [
            ui.Tile("Rows", ui.thousands(summary.get("rows"))),
            ui.Tile(
                "Interacting",
                ui.thousands(summary.get("positives")),
                ui.percent(
                    (summary.get("positives") or 0) / summary["rows"]
                    if summary.get("rows")
                    else None
                ),
            ),
            ui.Tile(
                "Feature columns",
                ui.thousands(
                    (summary.get("phage_dim") or 0) + (summary.get("bacteria_dim") or 0)
                ),
                f"{summary.get('phage_dim')} phage + {summary.get('bacteria_dim')} host",
            ),
            ui.Tile("Stored as", "Parquet", "float32 columns"),
        ]
    )

    preview_width = st.slider(
        "Embedding columns to preview from each block", 0, 12, 4, 2
    )

    columns = data.feature_columns()
    if not columns:
        return

    schema = pd.DataFrame(columns=columns)
    phage = feature_tools.phage_columns(schema)[:preview_width]
    host = feature_tools.bacteria_columns(schema)[:preview_width]
    wanted = ["phage_id", "bacteria_id", "taxon_id", "label", *phage, *host]

    with st.spinner("Reading the feature table..."):
        frame = pd.read_parquet(data.settings().reproduced_features, columns=wanted)

    st.dataframe(frame.head(500).round(4), hide_index=True, width="stretch", height=360)
    st.caption(
        f"First 500 of {len(frame):,} rows. "
        f"The full table is at `{data.settings().reproduced_features}`."
    )


# ---------------------------------------------------------------------------
# The embedding space (formerly its own page)
# ---------------------------------------------------------------------------
#: Silhouette is O(n^2) in samples; above this the section samples first.
SILHOUETTE_SAMPLE_CAP = 1200


def _render_embedding_space() -> None:
    """The ESM-C embeddings: how they were pooled, and what structure they hold."""
    stores = data.stores()
    if not stores.ready:
        return

    meta = data.embedding_meta("phage")
    bacteria_meta = data.embedding_meta("bacteria")

    st.divider()
    ui.section(
        "The embedding space",
        "Each proteome is one vector: every protein embedded with ESM-C, then "
        "the proteins averaged.",
    )

    ui.tile_row(
        [
            ui.Tile("Phage vectors", ui.thousands(len(stores.phage))),
            ui.Tile("Host vectors", ui.thousands(len(stores.bacteria))),
            ui.Tile("Dimensions", ui.thousands(meta.get("dim") or 0), meta.get("pooling", "")),
            ui.Tile(
                "Proteins embedded",
                ui.thousands(
                    (meta.get("proteins_embedded") or 0)
                    + (bacteria_meta.get("proteins_embedded") or 0)
                ),
            ),
        ]
    )
    st.caption(
        f"Model {meta.get('model', 'esmc_300m')} · pooling `{meta.get('pooling', 'residue')}` · "
        "bacterial side "
        + (
            "filtered to resistance-related proteins"
            if bacteria_meta.get("filtered")
            else "whole proteome"
        )
    )

    with st.expander("How the pooling works, and why it matters"):
        st.markdown(
            "ESM-C returns one vector per residue, shape `(L, 960)`. Collapsing that "
            "to one vector per protein can be done two ways, and the project's own "
            "history contains both:\n\n"
            "- **`residue`** (used here) averages over the residue axis, giving 960 "
            "numbers per protein. Every protein lands in the same space whatever its "
            "length, so proteins and organisms are directly comparable.\n"
            "- **`legacy`** is what `refactored/ESMC_embedding.py` did: it averaged "
            "over the *feature* axis, producing one number per residue position, then "
            "zero-padded to the longest protein in the dataset. Feature *i* then means "
            '"mean activation at position *i*", and the width of the feature space '
            "changes whenever the dataset does — which is where the project's 3863 and "
            "4199 column counts came from.\n\n"
            "Both are implemented; `residue` is the default because its features mean "
            "the same thing from one run to the next."
        )

    projection = data.projection_frame()
    if projection.empty:
        return

    ui.section(
        "Where the phages sit",
        "PCA of the phage embeddings, one panel per host. Grey is every phage; "
        "colour is the phages of that panel's host.",
    )
    ui.chart(
        charts.projection_facets(
            projection,
            theme=ui.theme(),
            x="x",
            y="y",
            facet="bacteria_id",
            title="Phage embeddings, projected to two dimensions",
            subtitle="Panels share the same axes; a host whose phages cluster is separable",
        ),
        key="projection",
    )
    ui.table_view(projection.round(3), label="Show the projected coordinates")

    frame = _silhouette_frame(projection)
    if frame is None:
        return

    knife, averages = frame
    ui.section(
        "Do the host labels form clusters?",
        "Silhouette compares within-host tightness to between-host distance. "
        "Near zero means the labels do not separate in this space.",
    )
    ui.chart(
        charts.silhouette_plot(
            knife,
            averages=averages,
            theme=ui.theme(),
            title="Silhouette, by labelling",
            subtitle=(
                "True hosts against k-means clusters and a random baseline. "
                "A score near 0 means no cluster structure at that labelling."
            ),
        ),
        key="silhouette",
    )
    ui.table_view(
        pd.DataFrame(
            [{"labelling": key, "mean silhouette": round(value, 4)} for key, value in averages.items()]
        ),
        label="Show the mean scores",
    )
    st.markdown(
        "A near-zero score for the true hosts is a real finding, not a bug: host "
        "identity is **not** the dominant axis of variation in these embeddings. "
        "The classifiers can still separate hosts, because a supervised model is "
        "free to use directions that unsupervised clustering ignores."
    )


def _silhouette_frame(projection: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]] | None:
    """Per-sample silhouette for the true hosts, k-means, and random labels.

    Computed on the full embeddings rather than the 2-D projection: the question
    is whether the embedding space separates hosts, and PCA to two dimensions
    would answer a different, easier question.
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import StandardScaler

    stores = data.stores()
    hosts = projection["bacteria_id"].to_numpy()
    accessions = projection["phage_id"].to_numpy()

    unique_hosts = sorted(set(hosts))
    if len(unique_hosts) < 2 or len(accessions) < 10:
        return None

    rng = np.random.default_rng(42)
    if len(accessions) > SILHOUETTE_SAMPLE_CAP:
        keep = rng.choice(len(accessions), SILHOUETTE_SAMPLE_CAP, replace=False)
        accessions, hosts = accessions[keep], hosts[keep]

    matrix = StandardScaler().fit_transform(
        np.stack([stores.phage[key] for key in accessions], axis=0)
    )

    clusters = KMeans(n_clusters=len(unique_hosts), n_init=10, random_state=42).fit_predict(matrix)
    random_labels = rng.integers(0, len(unique_hosts), size=len(accessions))

    parts = []
    averages: dict[str, float] = {}
    for name, labels in (
        ("True hosts", hosts),
        ("K-means", clusters.astype(str)),
        ("Random", random_labels.astype(str)),
    ):
        values = silhouette_samples(matrix, labels)
        averages[name] = float(values.mean())
        # Column names are what charts.silhouette_plot expects: it facets on
        # "Source" and tooltips on "Label".
        parts.append(
            pd.DataFrame(
                {
                    "Source": name,
                    "Label": [str(label) for label in labels],
                    "silhouette": values,
                }
            )
        )

    return pd.concat(parts, ignore_index=True), averages
