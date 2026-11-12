"""Reproduce the SmartAMR dataset and models from the raw inputs.

This subpackage is the runnable form of what ``refactored/`` documented as a
sequence of hand-edited scripts: select phages, fetch and filter the bacterial
proteomes, embed both sides with ESM-C, pair them into features, train the
classifiers. The scripts hardcoded ``/mnt/d/Sebastien/...`` paths, a Windows
mount that no longer exists; every path here comes from
:class:`SmartAMR.config.Settings` instead.

The stages are separate modules so each can be re-run on its own, and each
writes its output to disk before the next reads it:

===========================  ======================================
:mod:`~SmartAMR.reproduce.hosts`      the nine bacterial hosts and their ids
:mod:`~SmartAMR.reproduce.catalog`    choose phages that have a local proteome
:mod:`~SmartAMR.reproduce.proteomes`  fetch bacterial proteomes, filter them
:mod:`~SmartAMR.reproduce.embed`      ESM-C embeddings, resumable, capped
:mod:`~SmartAMR.reproduce.features`   pair embeddings into a labelled table
:mod:`~SmartAMR.reproduce.train`      the notebooks' two training tasks
:mod:`~SmartAMR.reproduce.runner`     run the stages in order
===========================  ======================================
"""

from __future__ import annotations

from SmartAMR.reproduce.hosts import HOSTS, Host, host_by_name

__all__ = ["HOSTS", "Host", "host_by_name"]
