"""The SmartAMR web app: a Streamlit front end over the reproduced pipeline.

Run it with ``smartamr app`` or ``streamlit run SmartAMR/webapp/main.py``.

The app is read-only over the artefacts that
:mod:`SmartAMR.reproduce.runner` writes -- it never trains or embeds on a page
load. Anything missing is reported as a missing stage with the command that
produces it, so a half-built project renders as a checklist rather than a
traceback.
"""

from __future__ import annotations

APP_TITLE = "SmartAMR"
APP_TAGLINE = "Phage-host interaction from protein language model embeddings"

__all__ = ["APP_TITLE", "APP_TAGLINE"]
