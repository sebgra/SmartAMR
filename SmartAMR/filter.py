"""Deprecated: moved to :mod:`SmartAMR.data.filter`.

Kept as a re-export so existing imports keep working. The new module is the
one that is maintained; it type-annotates its paths and fixes a spelling:
``get_amr_measurment_data`` became ``get_amr_measurement_data``. The old name
is aliased below so nothing that used it breaks.
"""

from __future__ import annotations

import warnings

from SmartAMR.data.filter import (
    filter_dataset,
    genome_ids,
    get_amr_data,
    get_amr_measurement_data,
    get_taxon_ids,
)

#: Original, misspelled name of :func:`get_amr_measurement_data`.
get_amr_measurment_data = get_amr_measurement_data

warnings.warn(
    "SmartAMR.filter is deprecated: import from SmartAMR.data.filter instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "filter_dataset",
    "genome_ids",
    "get_amr_data",
    "get_amr_measurement_data",
    "get_amr_measurment_data",
    "get_taxon_ids",
]
