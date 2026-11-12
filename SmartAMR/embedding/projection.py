"""Project embeddings down to two or three dimensions.

Logic lifted from ``notebooks/embedding_plot.ipynb``: t-SNE, UMAP and PCA over
the phage half of the feature table. Used by the dashboard to show how the
embeddings separate by host.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

METHODS = ("pca", "tsne", "umap")
DEFAULT_SEED = 42


def project(
    X: pd.DataFrame | np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    seed: int = DEFAULT_SEED,
    perplexity: float = 3.0,
) -> np.ndarray:
    """Reduce ``X`` to ``n_components`` dimensions.

    Parameters
    ----------
    X
        Samples by features.
    method
        ``"pca"``, ``"tsne"`` or ``"umap"``.
    n_components
        Target dimensionality.
    seed
        Random seed, for the stochastic methods.
    perplexity
        t-SNE perplexity; capped below the sample count, which t-SNE requires.

    Returns
    -------
    numpy.ndarray
        The projected coordinates.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {', '.join(METHODS)}")

    values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    if values.shape[0] < 2:
        raise ValueError("At least two samples are needed to project embeddings")

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed).fit_transform(values)

    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE requires perplexity < n_samples.
        safe_perplexity = max(2.0, min(perplexity, values.shape[0] - 1))
        return TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
            perplexity=safe_perplexity,
            random_state=seed,
        ).fit_transform(values)

    try:
        import umap
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("UMAP is required for method='umap'. Install it with: pip install umap-learn") from error

    return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(values)


def projection_frame(
    X: pd.DataFrame | np.ndarray,
    labels: Sequence[str],
    method: str = "pca",
    seed: int = DEFAULT_SEED,
    extra: dict[str, Sequence] | None = None,
) -> pd.DataFrame:
    """Project to 2D and return a tidy frame ready to plot.

    Parameters
    ----------
    X
        Samples by features.
    labels
        One label per sample, used as the ``bacteria_id`` column.
    method
        Projection method.
    seed
        Random seed.
    extra
        Further columns to carry through, for tooltips.

    Returns
    -------
    pandas.DataFrame
        Columns ``x``, ``y``, ``bacteria_id``, plus anything in ``extra``.
    """

    coordinates = project(X, method=method, n_components=2, seed=seed)

    frame = pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "bacteria_id": list(labels),
        }
    )

    for name, values in (extra or {}).items():
        frame[name] = list(values)

    return frame
