"""SmartAMR: predicting phage-host interaction from protein language models.

The package is organised by what each part does:

=================================  =========================================
:mod:`SmartAMR.config`             where data lives; one ``Settings`` object
:mod:`SmartAMR.resources`          the share of the machine a run may use
:mod:`SmartAMR.data`               fetching and filtering sequence data
:mod:`SmartAMR.embedding`          ESM-C embeddings and their projections
:mod:`SmartAMR.models`             training, evaluation, prediction, screening
:mod:`SmartAMR.genomics`           AMR gene lookups and codon tables
:mod:`SmartAMR.reproduce`          rebuild the dataset and models end to end
:mod:`SmartAMR.webapp`             the Streamlit web app
:mod:`SmartAMR.viz`                chart builders and colour tokens
:mod:`SmartAMR.cli`                the ``smartamr`` command
=================================  =========================================

Nothing heavy is imported here: ``torch``, ``esm``, ``sklearn`` and ``streamlit``
are all imported inside the functions that need them, so ``import SmartAMR`` and
``smartamr --help`` stay fast and work in an environment that has only some of
them installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
