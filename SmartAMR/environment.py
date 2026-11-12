"""Report which scientific dependencies are installed and at which version.

Replaces the top-level ``module_checker.py`` script: it imports each package
individually so a missing one is reported instead of aborting the whole check.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

#: Core dependencies, installed with the package itself.
REQUIRED_MODULES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "Numpy"),
    ("scipy", "Scipy"),
    ("pandas", "pandas"),
    ("Bio", "Biopython"),
    ("sklearn", "Sci-kit learn"),
)

#: Packages only some commands need, with the extra that provides each one.
#: A miss is reported but never fatal.
OPTIONAL_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("xgboost", "XGBoost", "boost"),
    ("tables", "PyTables (HDF5)", "hdf5"),
    ("esm", "ESM", "embedding"),
    ("torch", "Pytorch", "embedding"),
    ("matplotlib", "Matplotlib", "viz"),
    ("seaborn", "Seaborn", "viz"),
    ("umap", "UMAP", "viz"),
    ("streamlit", "Streamlit", "app"),
    ("altair", "Altair", "app"),
    ("tensorflow", "TensorFlow", "prototypes only"),
    ("keras", "Keras", "prototypes only"),
)


def module_version(import_name: str) -> str | None:
    """Version of an installed module, or None when it cannot be imported."""
    try:
        module = importlib.import_module(import_name)
    except Exception:  # noqa: BLE001 - a broken install must not crash the check
        return None
    return str(getattr(module, "__version__", "unknown"))


def check_modules(include_optional: bool = True) -> Dict[str, Dict[str, str | None]]:
    """Collect the version of every required (and optional) dependency."""

    modules: List[Tuple[str, str, bool, str]] = [
        (import_name, display, True, "") for import_name, display in REQUIRED_MODULES
    ]
    if include_optional:
        modules += [
            (import_name, display, False, extra) for import_name, display, extra in OPTIONAL_MODULES
        ]

    report: Dict[str, Dict[str, str | None]] = {}
    for import_name, display, required, extra in modules:
        report[display] = {
            "module": import_name,
            "version": module_version(import_name),
            "required": "yes" if required else "no",
            "extra": extra or None,
        }

    return report


def format_report(report: Dict[str, Dict[str, str | None]]) -> str:
    """Render the dependency report as aligned text."""

    width = max(len(name) for name in report) if report else 0
    lines: List[str] = []
    missing_required: List[str] = []

    for display, info in report.items():
        version = info["version"]
        if version is None:
            if info["required"] == "yes":
                status = "MISSING (required)"
                missing_required.append(display)
            elif info["extra"]:
                status = f"not installed (extra: {info['extra']})"
            else:
                status = "not installed (optional)"
        else:
            status = version
        lines.append(f"{display:<{width}} : {status}")

    if missing_required:
        lines.append("")
        lines.append(f"Missing required package(s): {', '.join(missing_required)}")
        lines.append("Install them with: pip install -e . (or: uv sync)")

    return "\n".join(lines)
