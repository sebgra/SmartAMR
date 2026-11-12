"""Chart building blocks, shared by the web app and the figure export.

Three modules, separated by what they decide:

:mod:`SmartAMR.viz.theme`
    Colour and type tokens. The categorical palette is a fixed order of eight
    hues that are never cycled, plus a single-hue ramp for magnitude; the dark
    set is its own selection of steps rather than an inversion of the light
    one, and both were checked with a palette validator.

:mod:`SmartAMR.viz.charts`
    The chart functions themselves. Each takes a tidy DataFrame and a
    :class:`~SmartAMR.viz.theme.Theme` and returns an Altair chart, so the same
    builder serves the interactive app and the PNG/SVG export.

:mod:`SmartAMR.viz.style`
    The Matplotlib side of the same tokens, for the figure export.

``altair`` is imported by :mod:`~SmartAMR.viz.charts` and ``streamlit`` only by
:func:`~SmartAMR.viz.theme.active_theme`, so importing this package costs
nothing in a CLI run that draws no charts.

This package used to live at ``SmartAMR.dashboard``, alongside a Streamlit app that
has since been merged into :mod:`SmartAMR.webapp`. The charts moved here
because they are not specific to one front end.
"""

from __future__ import annotations

__all__ = ["charts", "style", "theme"]
