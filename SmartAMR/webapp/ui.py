"""Shared presentation helpers: the app's styling, tiles and empty states.

Everything visual that more than one page needs lives here, so the pages stay
about their subject. The CSS is deliberately small and declares its colours as
tokens that follow Streamlit's own light/dark setting rather than hardcoding a
surface, which is the same discipline the chart theme follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import streamlit as st

from SmartAMR.viz.theme import Theme, active_theme

#: Injected once per session. Keeps the tiles, pills and section headers
#: consistent without a component library.
_STYLE = """
<style>
:root {
  --amr-accent: #2a78d6;
  --amr-good: #1baf7a;
  --amr-warn: #eda100;
  --amr-bad: #e34948;
  --amr-tile-bg: rgba(42, 120, 214, 0.06);
  --amr-tile-border: rgba(42, 120, 214, 0.18);
  --amr-muted: #78766f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --amr-tile-bg: rgba(57, 135, 229, 0.10);
    --amr-tile-border: rgba(57, 135, 229, 0.24);
    --amr-muted: #9b998f;
  }
}

/* Stat tile: a hero number with its label underneath. */
.amr-tile {
  background: var(--amr-tile-bg);
  border: 1px solid var(--amr-tile-border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
  height: 100%;
}
.amr-tile-value {
  font-size: 1.75rem;
  font-weight: 650;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.amr-tile-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--amr-muted);
  margin-top: 0.3rem;
}
.amr-tile-note { font-size: 0.8rem; color: var(--amr-muted); margin-top: 0.25rem; }

/* Status pill, used for pipeline stages. */
.amr-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 550;
  border: 1px solid transparent;
}
.amr-pill-done { background: rgba(27,175,122,0.12); border-color: rgba(27,175,122,0.35); }
.amr-pill-todo { background: rgba(120,118,111,0.12); border-color: rgba(120,118,111,0.3); }

.amr-hero {
  font-size: 0.95rem;
  color: var(--amr-muted);
  margin-top: -0.5rem;
  margin-bottom: 1.2rem;
}
.amr-section { margin-top: 0.5rem; margin-bottom: 0.2rem; font-weight: 620; }
.amr-caption { font-size: 0.82rem; color: var(--amr-muted); }
</style>
"""


def inject_style() -> None:
    """Add the app's CSS once per session."""
    if st.session_state.get("_amr_style_done"):
        return
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.session_state["_amr_style_done"] = True


def theme() -> Theme:
    """The chart theme matching the app's current appearance."""
    return active_theme()


@dataclass(frozen=True)
class Tile:
    """One statistic for :func:`tile_row`."""

    label: str
    value: str
    note: str = ""


def tile_row(tiles: Sequence[Tile]) -> None:
    """Render statistics as a row of tiles.

    A stat tile is the right form for a single number that needs no comparison;
    putting these in a chart would add axes to four unrelated scalars.
    """

    if not tiles:
        return
    columns = st.columns(len(tiles), gap="small")
    for column, tile in zip(columns, tiles):
        note = f'<div class="amr-tile-note">{tile.note}</div>' if tile.note else ""
        column.markdown(
            f'<div class="amr-tile">'
            f'<div class="amr-tile-value">{tile.value}</div>'
            f'<div class="amr-tile-label">{tile.label}</div>'
            f"{note}</div>",
            unsafe_allow_html=True,
        )


def pill(text: str, done: bool) -> str:
    """HTML for a status pill."""
    kind = "amr-pill-done" if done else "amr-pill-todo"
    mark = "✓" if done else "•"
    return f'<span class="amr-pill {kind}">{mark} {text}</span>'


def header(title: str, tagline: str = "") -> None:
    """Page title with an optional one-line subtitle."""
    st.title(title)
    if tagline:
        st.markdown(f'<div class="amr-hero">{tagline}</div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """A labelled section divider."""
    st.markdown(f'<div class="amr-section">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="amr-caption">{caption}</div>', unsafe_allow_html=True)


def missing_stage(label: str, command: str, detail: str = "") -> None:
    """Explain an absent artefact and name the command that creates it.

    Used instead of an exception: a project that has not been built yet is an
    expected state, and the useful response is the command to run.
    """

    st.info(
        f"**{label} not built yet.**"
        + (f" {detail}" if detail else "")
        + f"\n\nRun:\n```bash\n{command}\n```"
    )


def chart(figure, *, key: str | None = None) -> None:
    """Render an Altair chart at container width."""
    st.altair_chart(figure, width='stretch', key=key)


def table_view(frame, label: str = "Show the numbers", **kwargs) -> None:
    """A collapsed table beneath a chart.

    The palette's lighter hues fall below 3:1 contrast against the surface, so
    the validator's contrast warning obliges a non-colour route to the values.
    Every chart in this app is paired with one of these.
    """

    with st.expander(label):
        st.dataframe(frame, width='stretch', hide_index=True, **kwargs)


def percent(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage, or an em dash when absent."""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def number(value: float | int | None, digits: int = 3) -> str:
    """Format a score, or an em dash when absent."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def thousands(value: int | float | None) -> str:
    """Format a count with thousands separators."""
    if value is None:
        return "—"
    return f"{int(value):,}"


def caption_list(items: Iterable[str]) -> None:
    """Render notes as a muted bullet list."""
    items = [item for item in items if item]
    if not items:
        return
    body = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f'<ul class="amr-caption">{body}</ul>', unsafe_allow_html=True)
