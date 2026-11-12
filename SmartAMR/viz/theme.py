"""Chart tokens for the SmartAMR web app and figure export.

The palette is the validated reference instance: eight categorical hues in a
fixed order, one blue ramp for sequential magnitude. The dark column is its own
set of steps chosen for the dark surface, not an automatic flip of the light
one, and both were checked with the palette validator (lightness band, chroma
floor, colour-vision separation, normal-vision floor, contrast).

Scatter plots carry a three-series cap, which is why the embedding projection is
faceted rather than coloured by host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

#: Categorical hues, in the fixed order they must be assigned in.
CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
CATEGORICAL_DARK: List[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

#: Blue ramp, light to dark, for continuous magnitude (heatmaps).
SEQUENTIAL_BLUE: List[str] = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


@dataclass(frozen=True)
class Theme:
    """Resolved colours for one mode."""

    mode: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]

    @property
    def series_1(self) -> str:
        return self.categorical[0]

    @property
    def series_2(self) -> str:
        return self.categorical[1]

    def series(self, index: int) -> str:
        """Hue of a categorical slot, assigned in fixed order."""
        if index >= len(self.categorical):
            raise IndexError(
                f"Categorical slot {index} does not exist: hues are never cycled. "
                "Fold the extra series into 'Other', or facet the chart."
            )
        return self.categorical[index]


LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#78766f",
    grid="#e6e5e1",
    categorical=tuple(CATEGORICAL_LIGHT),
    sequential=tuple(SEQUENTIAL_BLUE),
)

DARK = Theme(
    mode="dark",
    surface="#1a1a19",
    text_primary="#f0efec",
    text_secondary="#c3c2b7",
    text_muted="#9b998f",
    grid="#333331",
    categorical=tuple(CATEGORICAL_DARK),
    sequential=tuple(SEQUENTIAL_BLUE),
)

#: Neutral used for the background cloud of a faceted scatter.
CONTEXT_GREY_LIGHT = "#d6d5d0"
CONTEXT_GREY_DARK = "#4a4a47"


def active_theme() -> Theme:
    """The theme matching Streamlit's current appearance.

    Falls back to light when the running Streamlit cannot report its theme.
    """

    try:
        import streamlit as st

        mode = getattr(getattr(st, "context", None), "theme", None)
        if mode is not None and getattr(mode, "type", None) == "dark":
            return DARK

        if st.get_option("theme.base") == "dark":
            return DARK
    except Exception:  # noqa: BLE001 - a theme probe must never break the app
        pass

    return LIGHT


def context_grey(theme: Theme) -> str:
    """Grey for marks that provide context rather than carry data."""
    return CONTEXT_GREY_DARK if theme.mode == "dark" else CONTEXT_GREY_LIGHT
