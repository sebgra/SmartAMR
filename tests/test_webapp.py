"""Tests for the web app, driven by Streamlit's own test harness.

``AppTest`` runs ``main.py`` in-process with no browser and no server: it
executes the script, collects the elements each page produced, and records any
exception instead of rendering a red box. That makes "does every page render"
a normal test rather than a thing to click through.

Two kinds of test live here:

* The **empty-project** tests point the app at a temporary directory with no
  artefacts in it. Every page must still render and say what is missing. These
  always run, so a broken import or a missing-file crash is caught on any
  machine.
* The **built-project** tests run against the real ``output/`` directory and are
  skipped when the pipeline has not been run. They are the ones that exercise
  the charts, the tables and the model scoring.

The app resolves its paths from ``SMARTAMR_*`` environment variables, which is
what lets a test redirect it without touching the working tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the app needs streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

#: Entry point, relative to the repository root.
APP = Path(__file__).resolve().parent.parent / "SmartAMR" / "webapp" / "main.py"

#: Every page in the sidebar, as the radio labels them.
PAGES = ["Analyze", "Dataset", "Models", "Pipeline"]

#: Key of the sidebar's page radio. Selecting it by key rather than by index
#: matters: the Analyze page has a radio of its own, and it comes before the
#: sidebar in element order, so ``app.radio[0]`` is not the page selector.
PAGE_KEY = "_amr_page"

#: Generous: the first run loads the embedding stores and fits nothing, but a
#: cold import of pandas, altair and sklearn is not instant.
TIMEOUT = 300

PROJECT_ROOT = APP.parent.parent.parent


def _is_built() -> bool:
    """True when the pipeline's artefacts are on disk."""
    return (PROJECT_ROOT / "output" / "models" / "metrics.json").exists()


built_only = pytest.mark.skipif(
    not _is_built(),
    reason="no pipeline output; run `smartamr reproduce` first",
)


def _run(page: str) -> AppTest:
    """Start the app and switch to ``page``."""
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    app.run()
    app.radio(key=PAGE_KEY).set_value(page).run()
    return app


@pytest.fixture()
def empty_project(tmp_path, monkeypatch):
    """Point the app at a directory with no artefacts in it."""
    for variable, value in {
        "SMARTAMR_ROOT": tmp_path,
        "SMARTAMR_DATA_DIR": tmp_path / "data",
        "SMARTAMR_CONFIG_DIR": tmp_path / "config",
        "SMARTAMR_OUTPUT_DIR": tmp_path / "output",
        "SMARTAMR_GENOMES_DIR": tmp_path / "GenomesDB",
    }.items():
        monkeypatch.setenv(variable, str(value))
    yield tmp_path


# ---------------------------------------------------------------------------
# The app starts
# ---------------------------------------------------------------------------
def test_the_app_starts_on_the_analyze_page():
    """Analyze is the landing page: it is the one that takes an input."""
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Analyze"
    # The app's own name lives in the sidebar, not in the page heading.
    assert any("SmartAMR" in block.value for block in app.markdown)


def test_every_page_is_reachable_from_the_sidebar():
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    app.run()
    assert app.radio(key=PAGE_KEY).options == PAGES


# ---------------------------------------------------------------------------
# Nothing built yet: every page must explain itself, not crash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", PAGES)
def test_a_page_renders_with_no_artefacts(page, empty_project):
    app = _run(page)
    assert not app.exception, f"{page} raised: {[e.value for e in app.exception]}"


@pytest.mark.parametrize("page", ["Analyze", "Dataset", "Models"])
def test_a_page_with_no_artefacts_names_the_command_to_run(page, empty_project):
    """The empty state has to be actionable, not just empty."""
    app = _run(page)
    text = " ".join(info.value for info in app.info)
    assert "smartamr reproduce" in text, f"{page} did not say how to build what it needs"


def test_the_pipeline_page_lists_every_stage_as_unbuilt(empty_project):
    app = _run("Pipeline")
    body = " ".join(block.value for block in app.markdown)
    for stage in ("catalog", "fetch", "filter", "embed", "features", "train"):
        assert stage in body


# ---------------------------------------------------------------------------
# Built project: the pages actually draw things
# ---------------------------------------------------------------------------
@built_only
@pytest.mark.parametrize("page", PAGES)
def test_a_page_renders_against_the_real_output(page):
    app = _run(page)
    assert not app.exception, f"{page} raised: {[e.value for e in app.exception]}"


@built_only
@pytest.mark.parametrize(
    ("page", "minimum_charts"),
    [("Analyze", 3), ("Dataset", 6), ("Models", 5)],
)
def test_a_page_draws_its_charts(page, minimum_charts):
    app = _run(page)
    charts = len(app.get("arrow_vega_lite_chart")) + len(app.get("vega_lite_chart"))
    assert charts >= minimum_charts, f"{page} drew {charts} charts"


@built_only
def test_the_pipeline_page_reports_the_resource_cap():
    """The cap a run used is shown, so a figure can be traced to its budget.

    It reaches the page as a sidebar caption and as a row of the "Last run"
    table, neither of which is an ``st.markdown`` element -- so look in the
    captions and in the rendered dataframes.
    """

    app = _run("Pipeline")

    captions = " ".join(caption.value for caption in app.caption)
    tables = " ".join(
        frame.value.to_csv(index=False) for frame in app.dataframe
    )
    body = captions + " " + tables

    assert "Resource cap" in body, "the overview did not report the run's resource cap"
    assert "GPU" in body and "RAM" in body


@built_only
def test_the_models_page_shows_both_tasks():
    app = _run("Models")
    labels = [label for tab in app.tabs for label in [tab.label]]
    assert any("infect this host" in label for label in labels)
    assert any("Which host" in label for label in labels)


@built_only
def test_a_known_phage_scores_its_recorded_host_highly():
    """The round trip the Analyze page performs, asserted without the UI.

    Take a phage whose host the metadata records, score it against all nine
    hosts the way the page does, and check the recorded host ranks near the
    top. The phage was in the training split, so this checks the wiring -- a
    transposed block or a mismatched column order would scramble it -- rather
    than generalisation.
    """

    import numpy as np
    import pandas as pd

    from SmartAMR.reproduce.features import bacteria_columns, embedding_columns, phage_columns
    from SmartAMR.reproduce.hosts import HOSTS
    from SmartAMR.reproduce.train import BINARY_TASK
    from SmartAMR.webapp import data
    from SmartAMR.webapp.scoring import positive_scores

    columns = data.feature_columns()
    models = data.available_models(BINARY_TASK)
    pairs = data.pairs()
    stores = data.stores()
    if not columns or not models or pairs.empty or not stores.ready:
        pytest.skip("pipeline output incomplete")

    positives = pairs[pairs["label"].astype(bool)]
    positives = positives[positives["phage_id"].astype(str).isin(stores.phage)]
    if positives.empty:
        pytest.skip("no embedded phage with a recorded host")

    row = positives.iloc[0]
    accession, recorded = str(row["phage_id"]), str(row["bacteria_id"])
    vector = stores.phage[accession]

    schema = pd.DataFrame(columns=columns)
    names = phage_columns(schema) + bacteria_columns(schema)
    targets = [host for host in HOSTS if host.taxon_key in stores.bacteria]
    features = pd.DataFrame(
        np.hstack(
            [
                np.tile(vector, (len(targets), 1)),
                np.stack([stores.bacteria[host.taxon_key] for host in targets], axis=0),
            ]
        ).astype(np.float32),
        columns=names,
    )[embedding_columns(schema)]

    scores = positive_scores(data.model(BINARY_TASK, models[0]), features)
    order = [targets[index].inphared_name for index in np.argsort(-scores)]

    assert recorded in order[: max(3, len(order) // 3)], (
        f"{accession}'s recorded host {recorded} ranked {order.index(recorded) + 1} "
        f"of {len(order)}"
    )


@built_only
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_every_trained_estimator_can_score(task):
    """Each estimator must rank, whichever scoring interface it exposes.

    The SVM has no ``predict_proba`` on purpose, and an earlier version of the
    finder swallowed that in a bare ``except`` and showed an empty page.
    """

    import pandas as pd

    from SmartAMR.webapp import data
    from SmartAMR.webapp.scoring import class_scores, positive_scores

    columns = data.feature_columns()
    models = data.available_models(task)
    if not columns or not models:
        pytest.skip("nothing trained for this task")

    from SmartAMR.reproduce.features import embedding_columns, phage_columns

    schema = pd.DataFrame(columns=columns)
    names = embedding_columns(schema) if task == "binary" else phage_columns(schema)
    row = pd.DataFrame([{name: 0.0 for name in names}])

    for name in models:
        pipeline = data.model(task, name)
        assert pipeline is not None, f"{task}/{name} did not load"
        if task == "binary":
            scores = positive_scores(pipeline, row)
            assert scores.shape == (1,)
            assert 0.0 <= float(scores[0]) <= 1.0
        else:
            matrix, classes = class_scores(pipeline, row)
            assert matrix.shape == (1, len(classes))
            assert len(classes) >= 2


# ---------------------------------------------------------------------------
# The Analyze page's phage/bacterium guess
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("proteins", "expected", "confident"),
    [
        (1, "Phage", True),
        (250, "Phage", True),
        (251, "Phage", False),      # the overlapping band: admits it cannot tell
        (377, "Phage", False),      # the real L. lactis proteome lands here
        (999, "Phage", False),
        (1000, "Bacterium", True),
        (6353, "Bacterium", True),  # the real A. baumannii proteome
    ],
)
def test_the_kind_guess_knows_when_it_cannot_tell(proteins, expected, confident):
    """Phage and bacterial protein counts overlap, so the guess must say so.

    The nine reference proteomes hold 346-6353 proteins and the phages 10-420:
    a count in between is genuinely ambiguous, and claiming otherwise would
    silently send a bacterial proteome down the phage path.
    """

    from SmartAMR.webapp.pages.analyze import _guess_kind

    guess, is_confident = _guess_kind(proteins)
    assert guess == expected
    assert is_confident is confident


@built_only
def test_a_bacterial_upload_is_filtered_before_embedding():
    """The bacterial side of the training data was filtered; uploads must match.

    Embedding a whole proteome when the models learned from the
    resistance-protein subset would place the vector somewhere they have never
    seen, so the Analyze page applies the same filter. This asserts the filter
    actually selects a small subset of a real proteome.
    """

    from SmartAMR.config import Settings
    from SmartAMR.reproduce.proteomes import count_records, filter_proteome, read_protein_names

    settings = Settings()
    source = settings.uniprot_proteomes / "proteome_taxon_1360.fasta"
    if not source.exists():
        pytest.skip("no downloaded bacterial proteome")

    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        destination = Path(workspace) / "filtered.fasta"
        kept = filter_proteome(
            source, destination, read_protein_names(settings.selected_proteins)
        )

    total = count_records(source)
    assert 0 < kept < total, f"filter kept {kept} of {total}"
