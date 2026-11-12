"""Page chrome: CSS and the small layout pieces the app reuses.

Charts carry their own validated palette; this module only dresses the page
around them. Every colour here is a token from
:mod:`SmartAMR.viz.theme`, so light and dark stay consistent.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import streamlit as st

from SmartAMR.viz.theme import Theme, context_grey

#: Icon per page, used in the sidebar.
PAGE_ICONS: Mapping[str, str] = {
    "Pipeline": "◴",
    "Reproduce": "⟳",
    "Project": "◫",
    "Corpus": "◉",
    "Features": "◈",
    "Train": "◆",
    "Experiments": "⬡",
    "Phage finder": "✦",
    "Screen phages": "⬢",
    "Predict": "▲",
}


def inject(theme: Theme) -> None:
    """Add the stylesheet. Call once per run, before anything is drawn."""

    muted = context_grey(theme)

    st.markdown(
        f"""
        <style>
          /* Roomier page, narrower measure for text */
          .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}

          /* Hero */
          .sa-hero {{
            display: flex; align-items: baseline; gap: .75rem;
            flex-wrap: wrap; margin-bottom: .25rem;
          }}
          .sa-hero h1 {{
            font-size: 1.9rem; font-weight: 700; margin: 0;
            letter-spacing: -.02em; color: {theme.text_primary};
          }}
          .sa-hero .sa-sub {{ color: {theme.text_secondary}; font-size: .95rem; }}

          /* Eyebrow above a section */
          .sa-eyebrow {{
            text-transform: uppercase; letter-spacing: .08em;
            font-size: .7rem; font-weight: 600;
            color: {theme.text_muted}; margin: 1.6rem 0 .2rem;
          }}

          /* Pills */
          .sa-pill {{
            display: inline-block; padding: .14rem .55rem; border-radius: 999px;
            font-size: .72rem; font-weight: 600; letter-spacing: .01em;
            border: 1px solid {muted}; color: {theme.text_secondary};
          }}
          .sa-pill-ok {{ border-color: {theme.series(2)}; color: {theme.series(2)}; }}
          .sa-pill-warn {{ border-color: {theme.series_2}; color: {theme.series_2}; }}

          /* Metric cards */
          div[data-testid="stMetric"] {{
            background: transparent;
            border: 1px solid {muted}55;
            border-radius: 12px;
            padding: .8rem 1rem;
          }}
          div[data-testid="stMetricLabel"] p {{
            font-size: .75rem !important; color: {theme.text_secondary} !important;
            text-transform: uppercase; letter-spacing: .05em;
          }}
          div[data-testid="stMetricValue"] {{
            font-size: 1.55rem !important; font-weight: 650;
          }}

          /* Sidebar nav */
          section[data-testid="stSidebar"] .sa-brand {{
            font-size: 1.15rem; font-weight: 700; letter-spacing: -.01em;
            margin-bottom: 0;
          }}
          section[data-testid="stSidebar"] [role="radiogroup"] label {{
            padding: .18rem .35rem; border-radius: 8px;
          }}
          section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {muted}33;
          }}

          /* Empty state */
          .sa-empty {{
            border: 1px dashed {muted}; border-radius: 14px;
            padding: 1.4rem 1.5rem; margin: .5rem 0 1rem;
          }}
          .sa-empty h4 {{
            margin: 0 0 .4rem; font-size: 1rem; font-weight: 650;
            color: {theme.text_primary};
          }}
          .sa-empty p {{ margin: 0 0 .5rem; color: {theme.text_secondary}; font-size: .9rem; }}

          /* Tighten dataframes and tabs */
          div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
          button[data-baseweb="tab"] {{ font-size: .88rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badges: Sequence[tuple[str, str]] = ()) -> None:
    """Page title with an optional row of pills.

    Each badge is ``(text, kind)`` where kind is ``""``, ``"ok"`` or ``"warn"``.
    """

    pills = "".join(
        f'<span class="sa-pill sa-pill-{kind}">{text}</span>' if kind else
        f'<span class="sa-pill">{text}</span>'
        for text, kind in badges
    )
    st.markdown(
        f'<div class="sa-hero"><h1>{title}</h1>{pills}</div>'
        f'<div class="sa-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> None:
    """A small uppercase label above a block."""
    st.markdown(f'<div class="sa-eyebrow">{text}</div>', unsafe_allow_html=True)


def metric_row(items: Mapping[str, object]) -> None:
    """A row of metric cards, one per entry."""
    if not items:
        return
    for column, (label, value) in zip(st.columns(len(items)), items.items()):
        column.metric(label, value)


def empty_state(
    title: str,
    body: str,
    commands: Iterable[str] = (),
    demo_hint: bool = False,
) -> None:
    """Explain why a page is empty and what to run to fill it."""

    st.markdown(
        f'<div class="sa-empty"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )

    commands = list(commands)
    if commands:
        st.code("\n".join(commands), language="bash")

    if demo_hint:
        st.caption(
            "No data to hand? The button below builds a demo table: real phage "
            "accessions and real labels, synthetic embeddings."
        )
