"""Analyze: upload one proteome, get its embedding and its predicted partners.

The app's main page, and the only one that needs an input. It replaces the
separate Predict and Finder pages, which asked the same question from two
directions and duplicated most of their machinery:

* a **phage** proteome scores against the nine bacterial hosts -> which
  bacteria it could kill;
* a **bacterium** proteome scores against every embedded phage -> which phages
  to try against it.

Both directions share one pipeline: embed with ESM-C under the resource caps,
pair the vector with the other side's stored vectors, score with the binary
model, then draw the ranking, the score spread, the nearest known relatives and
where the upload lands in the embedding projection.

One asymmetry is deliberate and visible in the UI. The bacterial side of the
training data was **filtered** to the resistance-related proteins in
``config/selected_bacterian_proteins.txt`` before embedding, so an uploaded
bacterial proteome is filtered the same way. Embedding all ~5,000 proteins of a
whole proteome would place it somewhere the model has never seen, and the
prediction would be meaningless rather than merely uncertain.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from SmartAMR.reproduce.hosts import HOSTS, host_by_taxon
from SmartAMR.reproduce.train import BINARY_TASK, MULTICLASS_TASK
from SmartAMR.viz import charts
from SmartAMR.webapp import data, ui
from SmartAMR.webapp.scoring import class_scores, positive_scores, score_is_probability

PHAGE = "Phage"
BACTERIA = "Bacterium"

# Guessing phage vs bacterium from the protein count alone only works at the
# extremes, and the honest thing is to say when it does not. Phages in this
# corpus hold 10-420 proteins; the nine reference bacterial proteomes hold
# 346-6353. Those ranges OVERLAP: the L. lactis proteome has 377 proteins, more
# than most phages. So there are three bands, and the middle one admits it
# cannot tell.
#: At or below this, a proteome is confidently viral.
PHAGE_CEILING = 250
#: At or above this, a proteome is confidently bacterial.
BACTERIA_FLOOR = 1000

#: Candidates listed by default when scoring phages against a bacterium.
DEFAULT_TOP_N = 20

#: Nearest relatives shown.
NEIGHBOURS = 8


def render() -> None:
    ui.header(
        "Analyze",
        "Upload a proteome — a phage or a bacterium — and get its ESM-C embedding, "
        "its predicted partners, and where it sits among what is already known.",
    )

    stores = data.stores()
    if not stores.ready:
        ui.missing_stage(
            "The embeddings",
            "smartamr reproduce",
            "The app scores an upload against the stored vectors, so those have to exist.",
        )
        return

    if not data.available_models(BINARY_TASK):
        ui.missing_stage("The trained models", "smartamr reproduce --stages train")
        return

    columns = data.feature_columns()
    if not columns:
        ui.missing_stage("The feature table", "smartamr reproduce --stages features")
        return

    choice = _input_panel(stores)
    if choice is None:
        return

    kind, vector, title, protein_count = choice

    st.divider()
    if kind == PHAGE:
        _render_phage(vector, title, protein_count, columns)
    else:
        _render_bacteria(vector, title, protein_count, columns)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def _input_panel(stores) -> tuple[str, np.ndarray, str, int] | None:
    """The upload/pick panel. Returns ``(kind, vector, title, proteins)``."""
    with st.container(border=True):
        left, right = st.columns([2, 1.3], gap="large")

        with left:
            upload = st.file_uploader(
                "Proteome (protein FASTA)",
                type=["faa", "fasta", "fa", "txt"],
                help=(
                    "One organism per file: the proteins of a single phage, or of a "
                    "single bacterium. The INPHARED `.faa` layout works as is."
                ),
            )

        with right:
            if upload is not None:
                records = _count_records(upload)
                guess, confident = _guess_kind(records)
                kind = st.radio(
                    "This proteome is a",
                    [PHAGE, BACTERIA],
                    index=0 if guess == PHAGE else 1,
                    horizontal=True,
                    help=(
                        "Pre-selected from the protein count, which only separates "
                        "the two reliably at the extremes. Check it."
                    ),
                )
                if confident:
                    st.caption(f"{records} proteins · looks like a **{guess.lower()}**")
                else:
                    st.caption(
                        f"{records} proteins · **cannot tell from the count** "
                        f"({PHAGE_CEILING}-{BACTERIA_FLOOR} proteins fits either), "
                        "so please confirm the choice above."
                    )
            else:
                kind = st.radio(
                    "This proteome is a", [PHAGE, BACTERIA], horizontal=True, index=0
                )

        if upload is None:
            st.caption(
                "No file yet. You can also score something already in the corpus:"
            )
            return _pick_from_corpus(stores, kind)

    vector, proteins = _embed(upload, kind)
    if vector is None:
        return None
    return kind, vector, upload.name, proteins


def _pick_from_corpus(stores, kind: str) -> tuple[str, np.ndarray, str, int] | None:
    """Fall back to an organism that is already embedded, so the page is usable."""
    if kind == PHAGE:
        options = sorted(stores.phage)
        if not options:
            return None
        accession = st.selectbox("Phage from the catalog", options, index=0)
        return PHAGE, stores.phage[accession], accession, 0

    options = [host.inphared_name for host in HOSTS if host.taxon_key in stores.bacteria]
    if not options:
        return None
    name = st.selectbox("Bacterium from the corpus", options, index=0)
    host = next(entry for entry in HOSTS if entry.inphared_name == name)
    return BACTERIA, stores.bacteria[host.taxon_key], name, 0


def _guess_kind(records: int) -> tuple[str, bool]:
    """``(guess, confident)`` for a proteome of ``records`` proteins.

    Returns ``confident=False`` in the overlapping middle band, where the count
    genuinely does not distinguish a large phage from a small bacterium.
    """

    if records <= PHAGE_CEILING:
        return PHAGE, True
    if records >= BACTERIA_FLOOR:
        return BACTERIA, True
    return PHAGE, False


def _count_records(upload) -> int:
    """Number of FASTA records in an uploaded file."""
    text = upload.getvalue().decode("utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.startswith(">"))


def _embed(upload, kind: str) -> tuple[np.ndarray | None, int]:
    """Embed an uploaded proteome the same way the pipeline did."""
    from SmartAMR.reproduce import embed as embedder
    from SmartAMR.reproduce.proteomes import filter_proteome, iter_fasta, read_protein_names
    from SmartAMR.resources import limits

    settings = data.settings()
    meta = data.embedding_meta("phage" if kind == PHAGE else "bacteria")
    pooling = str(meta.get("pooling", "residue"))
    expected = int(meta.get("dim") or 0)

    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "upload.faa"
        path.write_bytes(upload.getvalue())

        total = sum(1 for _ in iter_fasta(path))
        if total == 0:
            st.error("No FASTA records found in that file.")
            return None, 0

        # The bacterial side of the training data was filtered; match it.
        used = path
        kept = total
        if kind == BACTERIA and meta.get("filtered", True):
            used = Path(workspace) / "filtered.faa"
            try:
                names = read_protein_names(settings.selected_proteins)
            except FileNotFoundError as error:
                st.error(str(error))
                return None, 0
            kept = filter_proteome(path, used, names)
            if kept == 0:
                st.error(
                    "None of this proteome's headers mention a protein from "
                    "`config/selected_bacterian_proteins.txt`, so there is nothing to "
                    "embed on the same basis as the training data. Check that the "
                    "FASTA headers carry protein names."
                )
                return None, 0
            st.caption(
                f"{total} proteins in the file, {kept} kept by the resistance-protein "
                "filter — the same subset the models were trained on."
            )

        caps = limits()
        st.caption(f"Embedding {kept} proteins · {caps.summary()}")

        with st.spinner(f"Embedding {kept} proteins with ESM-C..."):
            model, device = _cached_model()
            progress = embedder.EmbedProgress(total=1)
            vector = embedder.embed_proteome(
                model, embedder._logits_config(), used, pooling, caps, progress
            )

    if vector is None:
        st.error("Every protein in that file failed to embed.")
        return None, 0

    if expected and vector.shape[0] != expected:
        st.error(
            f"The upload embedded to {vector.shape[0]} dimensions but the models "
            f"expect {expected}. Rebuild the embeddings and models with one pooling."
        )
        return None, 0

    notes = []
    if progress.truncated:
        notes.append(f"{progress.truncated} protein(s) truncated to the length limit")
    if progress.skipped_proteins:
        notes.append(f"{progress.skipped_proteins} protein(s) skipped after an error")
    ui.caption_list(notes)

    st.success(f"Embedded on {device}: {progress.proteins} proteins → {vector.shape[0]} dimensions")
    return vector, progress.proteins


@st.cache_resource(show_spinner=False)
def _cached_model() -> tuple[object, str]:
    """Load ESM-C once per server process, under the configured caps."""
    from SmartAMR.reproduce.embed import load_model
    from SmartAMR.resources import limits

    return load_model(limits())


# ---------------------------------------------------------------------------
# A phage was uploaded: which bacteria can it kill?
# ---------------------------------------------------------------------------
def _render_phage(vector: np.ndarray, title: str, proteins: int, columns: list[str]) -> None:
    from SmartAMR.reproduce.features import bacteria_columns, embedding_columns, phage_columns

    stores = data.stores()
    schema = pd.DataFrame(columns=columns)
    phage_names = phage_columns(schema)
    bacteria_names = bacteria_columns(schema)
    all_names = embedding_columns(schema)

    model_name = _model_picker(BINARY_TASK, "pair model", key="phage-model")
    pipeline = data.model(BINARY_TASK, model_name)
    if pipeline is None:
        st.error(f"Could not load the {model_name} model.")
        return

    targets = [host for host in HOSTS if host.taxon_key in stores.bacteria]
    if not targets:
        st.error("No bacterial embeddings to score against.")
        return

    # Build the two blocks as arrays and concatenate once. Assigning 960
    # columns one at a time fragments the frame and pandas warns about it.
    bacteria_block = np.stack([stores.bacteria[host.taxon_key] for host in targets], axis=0)
    phage_block = np.tile(vector, (len(targets), 1))
    features = pd.DataFrame(
        np.hstack([phage_block, bacteria_block]).astype(np.float32),
        columns=phage_names + bacteria_names,
    )[all_names]

    scores = positive_scores(pipeline, features)
    frame = pd.DataFrame({"target": [host.inphared_name for host in targets]})
    ranked = (
        pd.DataFrame({"target": frame["target"], "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    best = ranked.iloc[0]
    ui.tile_row(
        [
            ui.Tile("Uploaded phage", title),
            ui.Tile("Top target", str(best["target"]), f"score {best['score']:.3f}"),
            ui.Tile(
                "Predicted targets",
                ui.thousands(int((ranked["score"] >= 0.5).sum())),
                f"of {len(ranked)} hosts, at 0.5",
            ),
            ui.Tile("Proteins embedded", ui.thousands(proteins) if proteins else "—"),
        ]
    )

    ui.section(
        "Which bacteria could this phage kill?",
        f"The {model_name} model scores the phage together with each host.",
    )
    ui.chart(
        charts.bar_ranking(
            ranked,
            label="target",
            value="score",
            theme=ui.theme(),
            title=f"Bacterial targets of {title}",
            subtitle="0.5 is the decision threshold",
            label_title="Bacterium",
            value_title="Score",
            top_n=None,
        ),
        key="phage-targets",
    )
    ui.table_view(ranked.round(4), label="Show the scores")

    _render_host_model(vector, phage_names)
    _render_neighbours(vector, kind=PHAGE)
    _render_placement(vector, kind=PHAGE, title=title)


def _render_host_model(vector: np.ndarray, phage_names: list[str]) -> None:
    """The second opinion: the multiclass model naming one host."""
    names = data.available_models(MULTICLASS_TASK)
    if not names:
        return

    model_name = _model_picker(MULTICLASS_TASK, "host model", key="host-model")
    pipeline = data.model(MULTICLASS_TASK, model_name)
    if pipeline is None:
        return

    row = pd.DataFrame([dict(zip(phage_names, vector))])
    try:
        predicted = pipeline.predict(row)[0]
        matrix, classes = class_scores(pipeline, row)
    except Exception as error:  # noqa: BLE001
        st.caption(f"The host model could not score this phage: {error}")
        return

    calibrated = score_is_probability(pipeline)
    ranking = (
        pd.DataFrame({"host": classes, "score": matrix[0]})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    ui.section(
        "A second opinion, from the phage alone",
        "This model never sees the bacterium: it reads the phage embedding and "
        "names the host it most resembles.",
    )
    st.markdown(
        f"**{model_name}** predicts **{predicted}** "
        + (
            f"({ranking.iloc[0]['score']:.1%} confidence)."
            if calibrated
            else f"(score {ranking.iloc[0]['score']:.3f})."
        )
    )
    ui.chart(
        charts.bar_ranking(
            ranking,
            label="host",
            value="score",
            theme=ui.theme(),
            title="Host score",
            subtitle=(
                "Probabilities over the nine hosts, summing to 1"
                if calibrated
                else "One-vs-rest margins squashed to 0-1; they do not sum to 1"
            ),
            label_title="Host",
            value_title="Score",
            top_n=None,
        ),
        key="host-model",
    )
    ui.table_view(ranking.round(4), label="Show the scores")


# ---------------------------------------------------------------------------
# A bacterium was uploaded: which phages should we try?
# ---------------------------------------------------------------------------
def _render_bacteria(vector: np.ndarray, title: str, proteins: int, columns: list[str]) -> None:
    from SmartAMR.reproduce.features import bacteria_columns, embedding_columns, phage_columns

    stores = data.stores()
    schema = pd.DataFrame(columns=columns)
    phage_names = phage_columns(schema)
    bacteria_names = bacteria_columns(schema)
    all_names = embedding_columns(schema)

    row = st.columns([2, 1, 1])
    with row[0]:
        model_name = _model_picker(BINARY_TASK, "pair model", key="bact-model")
    top_n = row[1].slider("Candidates to show", 5, 60, DEFAULT_TOP_N, 5)
    threshold = row[2].slider("Recommend above", 0.0, 1.0, 0.5, 0.05)

    pipeline = data.model(BINARY_TASK, model_name)
    if pipeline is None:
        st.error(f"Could not load the {model_name} model.")
        return

    accessions = sorted(stores.phage)
    phage_block = np.stack([stores.phage[key] for key in accessions], axis=0)
    bacteria_block = np.tile(vector, (len(accessions), 1))
    features = pd.DataFrame(
        np.hstack([phage_block, bacteria_block]).astype(np.float32),
        columns=phage_names + bacteria_names,
    )[all_names]

    with st.spinner(f"Scoring {len(accessions):,} phages..."):
        scores = positive_scores(pipeline, features)

    ranked = (
        pd.DataFrame({"phage": accessions, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    ranked["recommended"] = ranked["score"] >= threshold

    best = ranked.iloc[0]
    ui.tile_row(
        [
            ui.Tile("Uploaded bacterium", title),
            ui.Tile("Best candidate", str(best["phage"]), f"score {best['score']:.3f}"),
            ui.Tile(
                f"Above {threshold:.2f}",
                ui.thousands(int(ranked["recommended"].sum())),
                f"of {len(ranked)} phages",
            ),
            ui.Tile("Proteins embedded", ui.thousands(proteins) if proteins else "—"),
        ]
    )

    ui.section(
        "Which phages should we try against it?",
        f"Every embedded phage scored against this proteome by the {model_name} model.",
    )
    shown = ranked.head(int(top_n))
    ui.chart(
        charts.bar_ranking(
            shown,
            label="phage",
            value="score",
            theme=ui.theme(),
            title=f"Phage candidates for {title}",
            subtitle=f"Top {len(shown)} of {len(ranked)}, threshold {threshold:.2f}",
            label_title="Phage",
            value_title="Score",
            top_n=None,
        ),
        key="bact-candidates",
    )

    table = shown.copy()
    table["score"] = table["score"].round(4)
    table["recommended"] = np.where(table["recommended"], "yes", "")
    st.dataframe(table, hide_index=True, width="stretch", height=320)
    st.download_button(
        "Download the full ranking as CSV",
        ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"smartamr_candidates_{title.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    ui.section(
        "How confident is the model overall?",
        "The spread of scores across every phage. A tight cluster below the "
        "threshold means no strong candidate, not a near miss for all of them.",
    )
    ui.chart(
        charts.score_distribution(
            ranked,
            theme=ui.theme(),
            value="score",
            threshold=float(threshold),
            title="Score across every phage",
            subtitle=f"{len(ranked)} candidates",
        ),
        key="bact-spread",
    )

    _render_neighbours(vector, kind=BACTERIA)
    _render_known_hosts(ranked, threshold)


def _render_known_hosts(ranked: pd.DataFrame, threshold: float) -> None:
    """Which recorded hosts the recommended phages are known to infect.

    A useful sanity signal: if the top candidates are all known E. coli phages,
    the uploaded proteome is being read as E. coli-like, whatever its label.
    """

    pairs = data.pairs()
    if pairs.empty:
        return

    recommended = set(ranked.loc[ranked["score"] >= threshold, "phage"])
    if not recommended:
        st.caption("No candidate passes the threshold, so there is nothing to characterise.")
        return

    known = pairs[pairs["label"].astype(bool)].copy()
    known["phage_id"] = known["phage_id"].astype(str)
    hits = known[known["phage_id"].isin(recommended)]
    if hits.empty:
        return

    counts = (
        hits.groupby("bacteria_id").size().reset_index(name="phages").sort_values("phages", ascending=False)
    )

    ui.section(
        "What do the recommended phages normally infect?",
        "Recorded hosts of the candidates above the threshold. A clear winner "
        "suggests which known bacterium this proteome resembles.",
    )
    ui.chart(
        charts.bar_ranking(
            counts,
            label="bacteria_id",
            value="phages",
            theme=ui.theme(),
            title="Recorded hosts of the recommended phages",
            subtitle="From the INPHARED metadata, not from the model",
            label_title="Recorded host",
            value_title="Phages",
        ),
        key="bact-known-hosts",
    )
    ui.table_view(counts, label="Show the counts")


# ---------------------------------------------------------------------------
# Shared graphs
# ---------------------------------------------------------------------------
def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of ``matrix`` to ``vector``."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    # A zero vector has no direction; report 0 rather than dividing by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(norms > 0, matrix @ vector / norms, 0.0)
    return similarity


def _render_neighbours(vector: np.ndarray, kind: str) -> None:
    """The closest known organisms of the same kind, by cosine similarity."""
    stores = data.stores()
    store = stores.phage if kind == PHAGE else stores.bacteria
    if len(store) < 2:
        return

    keys = sorted(store)
    matrix = np.stack([store[key] for key in keys], axis=0)
    similarity = _cosine(matrix, vector)

    if kind == PHAGE:
        labels = keys
        caption = "Known phages whose proteome embedding is closest to the upload."
        pairs = data.pairs()
        hosts: dict[str, str] = {}
        if not pairs.empty:
            known = pairs[pairs["label"].astype(bool)]
            hosts = (
                known.groupby(known["phage_id"].astype(str))["bacteria_id"]
                .agg(lambda values: ", ".join(sorted(set(values))))
                .to_dict()
            )
        frame = pd.DataFrame(
            {
                "organism": labels,
                "similarity": similarity,
                "infects": [hosts.get(key, "") for key in keys],
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "organism": [host_by_taxon(key).inphared_name for key in keys],
                "similarity": similarity,
            }
        )
        caption = "Which of the nine reference proteomes the upload most resembles."

    top = frame.sort_values("similarity", ascending=False).head(NEIGHBOURS).reset_index(drop=True)

    ui.section("Closest known relatives", caption)
    ui.chart(
        charts.bar_ranking(
            top,
            label="organism",
            value="similarity",
            theme=ui.theme(),
            title="Cosine similarity in embedding space",
            subtitle="1.0 would be an identical direction; this is similarity, not a prediction",
            label_title="Organism",
            value_title="Cosine similarity",
            top_n=None,
        ),
        key=f"neighbours-{kind}",
    )
    ui.table_view(top.round(4), label="Show the similarities")


def _render_placement(vector: np.ndarray, kind: str, title: str) -> None:
    """Where the upload lands in the PCA of the known phages."""
    if kind != PHAGE:
        return

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    stores = data.stores()
    keys = sorted(stores.phage)
    if len(keys) < 3:
        return

    matrix = np.stack([stores.phage[key] for key in keys], axis=0)
    scaler = StandardScaler().fit(matrix)
    projector = PCA(n_components=2, random_state=42).fit(scaler.transform(matrix))

    cloud = projector.transform(scaler.transform(matrix))
    point = projector.transform(scaler.transform(vector.reshape(1, -1)))

    known = pd.DataFrame({"phage_id": keys, "x": cloud[:, 0], "y": cloud[:, 1]})
    highlight = pd.DataFrame({"phage_id": [title], "x": point[:, 0], "y": point[:, 1]})

    ui.section(
        "Where the upload sits",
        "The known phages projected to two dimensions, with the upload marked. "
        "Landing inside the cloud means the models have seen proteomes like it.",
    )
    ui.chart(
        charts.projection_with_highlight(
            known,
            theme=ui.theme(),
            highlight=highlight,
            title="Phage embeddings, projected",
            subtitle="PCA fitted on the known phages; axes carry no unit",
        ),
        key="placement",
    )


def _model_picker(task: str, label: str, key: str) -> str:
    """Choose an estimator, defaulting to the best one for the task."""
    names = data.available_models(task)
    frame = data.metrics_frame(task)
    index = 0
    if not frame.empty:
        best = frame.loc[frame["f1_macro"].idxmax(), "model"]
        if best in names:
            index = names.index(best)
    return st.selectbox(
        label.capitalize(),
        names,
        index=index,
        key=key,
        help="Defaults to the estimator with the best macro F1 on this task.",
    )
