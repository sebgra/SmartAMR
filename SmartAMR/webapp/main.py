"""Entry point of the SmartAMR web app.

Run it with ``smartamr app``, or directly::

    streamlit run SmartAMR/webapp/main.py

The page set is declared here and each page is a plain ``render()`` function, so
a page can be imported and exercised without a Streamlit server.
"""

from __future__ import annotations

import streamlit as st

from SmartAMR.webapp import APP_TAGLINE, APP_TITLE, data, ui
from SmartAMR.webapp.pages import analyze, dataset, models, pipeline

#: Four pages, in the order they answer questions. "Analyze" is the one with an
#: input and comes first; the rest explain what is behind its answer. It
#: replaces the former Predict and Finder pages, which asked the same question
#: from two directions, and Dataset absorbed the former Embeddings page.
PAGES = {
    "Analyze": (analyze.render, ":material/biotech:"),
    "Dataset": (dataset.render, ":material/database:"),
    "Models": (models.render, ":material/model_training:"),
    "Pipeline": (pipeline.render, ":material/dashboard:"),
}


def _sidebar() -> str:
    """Render the sidebar and return the chosen page name."""
    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        st.caption(APP_TAGLINE)

        choice = st.radio(
            "Page",
            list(PAGES),
            label_visibility="collapsed",
            key="_amr_page",
        )

        st.divider()

        manifest = data.manifest()
        if manifest.get("resource_summary"):
            st.caption("**Last run's resource cap**")
            st.caption(manifest["resource_summary"])

        summary = data.feature_summary()
        if summary:
            st.caption(
                f"**Data** · {summary.get('phages', 0)} phages × "
                f"{summary.get('hosts', 0)} hosts = {summary.get('rows', 0)} pairs"
            )

        st.divider()
        if st.button("Reload artefacts", width='stretch'):
            # The loaders key on file mtimes, so a rebuild is normally picked up
            # on its own. This is the escape hatch for a file written with an
            # unchanged timestamp.
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.caption(
            "Everything shown is read from disk. Build or rebuild it with "
            "`smartamr reproduce`."
        )

    return choice


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui.inject_style()

    choice = _sidebar()
    render, _icon = PAGES[choice]
    render()


main()
