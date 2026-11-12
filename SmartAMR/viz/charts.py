"""Altair chart builders shared by the web app and the figure export.

Each builder picks its form from the job the data does: ranking a magnitude is a
horizontal bar, a confusion matrix is a one-hue heatmap, an embedding projection
is a faceted scatter (scatter plots cap at three colour series, and there are
more hosts than that). Every mark carries a tooltip, grids stay recessive, and
text never wears a series colour.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import altair as alt
import pandas as pd

from SmartAMR.viz.theme import Theme, context_grey

BAR_CORNER_RADIUS = 4
POINT_SIZE = 45
SURFACE_GAP = 2


def _base(chart: alt.Chart | alt.FacetChart | alt.LayerChart, theme: Theme) -> alt.Chart:
    """Apply the shared configuration: recessive axes, no view border.

    The background is transparent rather than a fixed colour, so the chart sits
    on whatever surface the page gives it. Vega-Lite otherwise defaults to
    white, which leaves dark-mode text unreadable on a white canvas.
    """
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None, fill=None)
        .configure_axis(
            grid=True,
            gridColor=theme.grid,
            gridWidth=1,
            domain=False,
            tickColor=theme.grid,
            labelColor=theme.text_secondary,
            titleColor=theme.text_secondary,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=theme.text_secondary,
            titleColor=theme.text_secondary,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_title(
            color=theme.text_primary,
            fontSize=13,
            fontWeight=600,
            anchor="start",
            subtitleColor=theme.text_secondary,
            subtitleFontSize=11,
        )
        .configure_header(
            labelColor=theme.text_secondary,
            titleColor=theme.text_secondary,
            labelFontSize=11,
            titleFontSize=11,
        )
    )


def bar_ranking(
    data: pd.DataFrame,
    label: str,
    value: str,
    theme: Theme,
    title: str = "",
    subtitle: str = "",
    top_n: int | None = 15,
    label_title: str | None = None,
    value_title: str | None = None,
) -> alt.Chart:
    """Rank categories by a magnitude: one series, horizontal bars.

    The category axis names each bar, so the chart needs no legend and no
    colour encoding.
    """

    frame = data.sort_values(value, ascending=False)
    if top_n:
        frame = frame.head(top_n)

    bars = (
        alt.Chart(frame)
        .mark_bar(color=theme.series_1, cornerRadiusEnd=BAR_CORNER_RADIUS, height=14)
        .encode(
            x=alt.X(f"{value}:Q", title=value_title or value, axis=alt.Axis(grid=True)),
            y=alt.Y(
                f"{label}:N",
                sort="-x",
                title=label_title or None,
                axis=alt.Axis(grid=False, labelLimit=220),
            ),
            tooltip=[
                alt.Tooltip(f"{label}:N", title=label_title or label),
                alt.Tooltip(f"{value}:Q", title=value_title or value, format=","),
            ],
        )
        .properties(
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
            # A fixed step per row keeps the gap between bars even, whether the
            # chart ranks 3 categories or 15.
            height=alt.Step(24),
        )
    )

    return _base(bars, theme)


def metric_bars(
    data: pd.DataFrame,
    label: str,
    value: str,
    theme: Theme,
    title: str = "",
    subtitle: str = "",
    domain: Sequence[float] = (0, 1),
    value_title: str | None = None,
    threshold: float | None = None,
) -> alt.Chart:
    """A bounded score per category, with the value written on each bar.

    With ``threshold`` set, a dashed rule marks the cut-off, so which bars
    clear it is visible without reading every number.
    """

    frame = data.sort_values(value, ascending=False)
    if threshold is not None:
        # Carried as a column so the rule shares the chart's single dataset.
        frame = frame.assign(_threshold=float(threshold))

    base = alt.Chart(frame).encode(
        x=alt.X(
            f"{value}:Q",
            title=value_title or value,
            scale=alt.Scale(domain=list(domain)),
        ),
        y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(grid=False, labelLimit=220)),
    )

    bars = base.mark_bar(color=theme.series_1, cornerRadiusEnd=BAR_CORNER_RADIUS, height=16).encode(
        tooltip=[
            alt.Tooltip(f"{label}:N", title=label),
            alt.Tooltip(f"{value}:Q", title=value_title or value, format=".3f"),
        ],
    )

    labels = base.mark_text(align="left", dx=6, color=theme.text_secondary, fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=".2f")
    )

    layers = [bars, labels]

    if threshold is not None:
        layers.append(
            alt.Chart(frame)
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.5, color=theme.series_2)
            .encode(
                x=alt.X("mean(_threshold):Q"),
                tooltip=[alt.Tooltip("mean(_threshold):Q", title="Threshold", format=".2f")],
            )
        )

    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        height=alt.Step(26),
    )

    return _base(chart, theme)


def confusion_heatmap(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    theme: Theme,
    title: str = "Confusion matrix",
    subtitle: str = "",
    normalise: bool = True,
) -> alt.Chart:
    """Confusion matrix as a one-hue heatmap with the value in every cell.

    Magnitude is a sequential encoding, so it uses the single blue ramp. Each
    cell is labelled, which keeps the matrix readable without relying on colour.
    """

    rows: List[dict] = []
    for i, row in enumerate(matrix):
        total = sum(row) or 1
        for j, count in enumerate(row):
            rows.append(
                {
                    "True label": str(labels[i]) if i < len(labels) else str(i),
                    "Predicted label": str(labels[j]) if j < len(labels) else str(j),
                    "count": float(count),
                    "share": float(count) / total,
                }
            )

    frame = pd.DataFrame(rows)
    field = "share" if normalise else "count"
    value_format = ".0%" if normalise else ","

    cells = (
        alt.Chart(frame)
        .mark_rect(stroke=theme.surface, strokeWidth=SURFACE_GAP)
        .encode(
            x=alt.X("Predicted label:N", title="Predicted label", axis=alt.Axis(grid=False, labelAngle=-30)),
            y=alt.Y("True label:N", title="True label", axis=alt.Axis(grid=False)),
            color=alt.Color(
                f"{field}:Q",
                title="Share" if normalise else "Pairs",
                scale=alt.Scale(range=list(theme.sequential)),
                legend=alt.Legend(format=value_format, gradientLength=120),
            ),
            tooltip=[
                alt.Tooltip("True label:N"),
                alt.Tooltip("Predicted label:N"),
                alt.Tooltip("count:Q", title="Pairs", format=","),
                alt.Tooltip("share:Q", title="Share of row", format=".1%"),
            ],
        )
    )

    # Direct labels: light ink on the dark end of the ramp, dark ink elsewhere.
    # float() matters: a numpy scalar is rendered into the Vega expression as
    # "np.float64(0.5)", which Vega cannot parse, and the chart silently fails.
    cutoff = float(frame[field].max()) * 0.6 if len(frame) else 0.0
    # The ink sits on the cell, not on the page, and the ramp is the same in
    # both modes - so these two are fixed rather than theme tokens.
    text = cells.mark_text(fontSize=11).encode(
        text=alt.Text(f"{field}:Q", format=value_format),
        color=alt.condition(
            alt.datum[field] > cutoff,
            alt.value("#fcfcfb"),
            alt.value("#0b0b0b"),
        ),
    )

    size = max(240, 64 * len(labels))
    chart = (cells + text).properties(
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        width=size,
        height=size,
    )

    return _base(chart, theme)


def projection_facets(
    data: pd.DataFrame,
    theme: Theme,
    x: str = "x",
    y: str = "y",
    facet: str = "bacteria_id",
    title: str = "",
    subtitle: str = "",
    columns: int = 3,
) -> alt.Chart:
    """Embedding projection as small multiples, one panel per host.

    A scatter cannot carry more than three colour series and there are more
    hosts than that, so identity moves to the panel title. The full cloud sits
    behind each panel in grey for context.
    """

    # A faceted layer is filtered down to its own panel, so the context cloud is
    # repeated once per panel. Both layers must come from one dataset, which a
    # faceted layered chart requires, so a "layer" column tells them apart.
    panels = data[facet].dropna().unique()
    background = pd.concat(
        [data[[x, y]].assign(**{facet: panel}) for panel in panels],
        ignore_index=True,
    ).assign(layer="context")

    combined = pd.concat([background, data.assign(layer="focus")], ignore_index=True)

    base = alt.Chart(combined).encode(
        x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
    )

    cloud = base.transform_filter(alt.datum.layer == "context").mark_circle(
        size=POINT_SIZE * 0.7, color=context_grey(theme), opacity=0.45
    )

    tooltip_columns = [
        column for column in data.columns if column not in {x, y, "layer"}
    ]
    points = (
        base.transform_filter(alt.datum.layer == "focus")
        .mark_circle(size=POINT_SIZE, color=theme.series_1, opacity=0.85, stroke=theme.surface, strokeWidth=1)
        .encode(tooltip=[alt.Tooltip(column) for column in tooltip_columns])
    )

    chart = (
        alt.layer(cloud, points)
        .properties(width=190, height=170)
        .facet(
            facet=alt.Facet(f"{facet}:N", title=None, header=alt.Header(labelFontSize=11)),
            columns=columns,
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        )
        .resolve_scale(x="shared", y="shared")
    )

    return _base(chart, theme)


def histogram(
    data: pd.DataFrame,
    value: str,
    theme: Theme,
    title: str = "",
    subtitle: str = "",
    bins: int = 40,
    value_title: str | None = None,
    count_title: str = "Phages",
) -> alt.Chart:
    """Distribution of one measure: one series, binned bars with a surface gap."""

    chart = (
        alt.Chart(data)
        .mark_bar(color=theme.series_1, cornerRadiusEnd=BAR_CORNER_RADIUS, binSpacing=SURFACE_GAP)
        .encode(
            x=alt.X(
                f"{value}:Q",
                bin=alt.Bin(maxbins=bins),
                title=value_title or value,
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y("count()", title=count_title),
            tooltip=[
                alt.Tooltip(f"{value}:Q", bin=alt.Bin(maxbins=bins), title=value_title or value),
                alt.Tooltip("count()", title=count_title, format=","),
            ],
        )
        .properties(
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
            height=220,
        )
    )

    return _base(chart, theme)


def model_comparison(
    data: pd.DataFrame,
    theme: Theme,
    title: str = "Model accuracy",
    subtitle: str = "",
) -> alt.Chart:
    """Accuracy per estimator, faceted by task.

    The two tasks answer different questions on different test sets, so their
    accuracies are not comparable. Faceting keeps them on separate scales
    instead of implying one bar beats another across tasks.
    """

    frame = data.copy()
    frame["Task"] = frame["Task"].map({"binary": "Binary", "multilabel": "Multilabel"}).fillna(frame["Task"])

    bars = (
        alt.Chart(frame)
        .mark_bar(color=theme.series_1, cornerRadiusEnd=BAR_CORNER_RADIUS, height=16)
        .encode(
            x=alt.X("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(grid=False)),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Task:N"),
                alt.Tooltip("Accuracy:Q", format=".3f"),
                alt.Tooltip("Test pairs:Q", format=","),
            ],
        )
    )

    labels = bars.mark_text(align="left", dx=6, color=theme.text_secondary, fontSize=11).encode(
        text=alt.Text("Accuracy:Q", format=".2f")
    )

    chart = (
        alt.layer(bars, labels)
        .properties(width=260, height=alt.Step(26))
        .facet(
            column=alt.Column("Task:N", title=None, header=alt.Header(labelFontSize=12)),
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        )
        .resolve_scale(y="independent")
    )

    return _base(chart, theme)


def _series_scale(models: Sequence[str], theme: Theme) -> alt.Scale:
    """Assign categorical hues in the fixed slot order, never cycled."""
    names = list(dict.fromkeys(models))
    if len(names) > len(theme.categorical):
        raise ValueError(
            f"{len(names)} series exceeds the {len(theme.categorical)} categorical slots. "
            "Facet the chart or fold the extras into 'Other'."
        )
    return alt.Scale(domain=names, range=[theme.series(i) for i in range(len(names))])


def _curve_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    theme: Theme,
    x_title: str,
    y_title: str,
    title: str,
    subtitle: str,
    diagonal: bool,
) -> alt.Chart:
    """Shared body of the ROC and DET charts: one 2px line per model."""

    scale = _series_scale(data["Model"].tolist(), theme)

    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2, interpolate="step-after")
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", title="Model", scale=scale, legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(f"{x}:Q", title=x_title, format=".3f"),
                alt.Tooltip(f"{y}:Q", title=y_title, format=".3f"),
            ],
        )
    )

    layers = [lines]

    if diagonal:
        reference = (
            alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]}))
            .mark_line(strokeDash=[2, 3], strokeWidth=1, color=context_grey(theme))
            .encode(x=f"{x}:Q", y=f"{y}:Q")
        )
        layers.insert(0, reference)

    # Direct labels at the end of each curve: three light-mode hues sit below
    # 3:1 on the light surface, so the labels are what carries identity there.
    # ROC curves all finish at (1, 1) and DET curves at (0, 0), so the labels
    # are staggered down the y axis instead of printing on top of each other.
    ends = data.sort_values(x).groupby("Model", as_index=False).last()
    ends = ends.sort_values(y, ascending=False).reset_index(drop=True)

    # Push each label at least one line below the previous one, then lift the
    # whole column if that ran off the bottom. A plain clamp would stack every
    # label that shares an endpoint back on the same spot.
    gap = 0.07
    positions: List[float] = []
    for value in ends[y]:
        position = float(value) if not positions else min(float(value), positions[-1] - gap)
        positions.append(position)

    deficit = 0.02 - min(positions)
    if deficit > 0:
        positions = [position + deficit for position in positions]

    ends["label_y"] = positions

    labels = (
        alt.Chart(ends)
        .mark_text(align="left", dx=6, fontSize=11, color=theme.text_secondary)
        .encode(x=alt.X(f"{x}:Q"), y=alt.Y("label_y:Q"), text=alt.Text("Model:N"))
    )
    layers.append(labels)

    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        width=320,
        height=300,
    )

    return _base(chart, theme)


def roc_curves(
    data: pd.DataFrame,
    theme: Theme,
    title: str = "ROC curves",
    subtitle: str = "True positives against false positives, every threshold",
) -> alt.Chart:
    """Receiver operating characteristic, one line per estimator."""
    return _curve_chart(
        data,
        x="fpr",
        y="tpr",
        theme=theme,
        x_title="False positive rate",
        y_title="True positive rate",
        title=title,
        subtitle=subtitle,
        diagonal=True,
    )


def det_curves(
    data: pd.DataFrame,
    theme: Theme,
    title: str = "DET curves",
    subtitle: str = "Missed infections against false alarms; lower left is better",
) -> alt.Chart:
    """Detection error tradeoff, one line per estimator."""
    return _curve_chart(
        data,
        x="fpr",
        y="fnr",
        theme=theme,
        x_title="False positive rate",
        y_title="False negative rate",
        title=title,
        subtitle=subtitle,
        diagonal=False,
    )


def score_separation(
    data: pd.DataFrame,
    theme: Theme,
    title: str = "Score separation",
    subtitle: str = "How far apart the model puts the two classes",
    model: str | None = None,
) -> alt.Chart:
    """Distribution of the model's score, split by the true class.

    A single accuracy number hides whether the classes are cleanly apart or
    only just on the right side of the threshold.
    """

    frame = data if model is None else data[data["Model"] == model]

    chart = (
        alt.Chart(frame)
        .mark_bar(binSpacing=0, opacity=0.75)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Model score", axis=alt.Axis(grid=False)),
            y=alt.Y("count()", title="Pairs", stack=None),
            color=alt.Color(
                "True class:N",
                title="True class",
                scale=alt.Scale(
                    domain=["Infects", "Does not infect"],
                    range=[theme.series_1, theme.series_2],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("True class:N"),
                alt.Tooltip("count()", title="Pairs", format=","),
            ],
        )
        .properties(
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
            height=220,
        )
    )

    return _base(chart, theme)


def silhouette_plot(
    data: pd.DataFrame,
    averages: Dict[str, float],
    theme: Theme,
    title: str = "Silhouette analysis",
    subtitle: str = "Per-sample silhouette, sorted within each group",
) -> alt.Chart:
    """The notebooks' silhouette knife plot, faceted by source.

    Each panel is one series, so the three sources stay comparable without
    putting nine cluster colours on screen. The dashed rule is the average.
    """

    from SmartAMR.models.evaluation import rank_within_group

    ordered = rank_within_group(data, value="silhouette", group="Source")
    # The average rides along as a column: a faceted layered chart must come
    # from one dataset, so the rule cannot have a frame of its own.
    ordered["average"] = ordered["Source"].map(averages)

    marks = (
        alt.Chart(ordered)
        .mark_area(color=theme.series_1, opacity=0.85, interpolate="step")
        .encode(
            x=alt.X("silhouette:Q", title="Silhouette", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("rank:Q", title="Samples", axis=alt.Axis(labels=False, ticks=False)),
            tooltip=[
                alt.Tooltip("Label:N", title="Group"),
                alt.Tooltip("silhouette:Q", format=".3f"),
            ],
        )
    )

    rules = (
        alt.Chart(ordered)
        .mark_rule(strokeDash=[3, 3], strokeWidth=1.5, color=theme.series_2)
        .encode(
            x=alt.X("mean(average):Q"),
            tooltip=[alt.Tooltip("mean(average):Q", title="Mean silhouette", format=".3f")],
        )
    )

    chart = (
        alt.layer(marks, rules)
        .properties(width=220, height=200)
        .facet(
            column=alt.Column(
                "Source:N",
                title=None,
                header=alt.Header(labelFontSize=12),
                # Reading order: what the labels achieve, what clustering
                # finds on its own, and the floor to beat.
                sort=["True labels", "KMeans clusters", "Random baseline"],
            ),
            title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        )
    )

    return _base(chart, theme)


def prediction_summary(
    data: pd.DataFrame,
    theme: Theme,
    title: str = "",
    subtitle: str = "",
    field: str = "interaction",
) -> alt.Chart:
    """How many candidates the model calls a match, and how many it does not.

    Two categories, so two categorical slots plus a legend; the bars are also
    directly labelled, so identity never rests on colour alone.
    """

    counts = (
        data[field]
        .map({1: "Infects", 0: "Does not infect"})
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Candidates")
    )

    bars = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=BAR_CORNER_RADIUS, height=26)
        .encode(
            x=alt.X("Candidates:Q", title="Candidates"),
            y=alt.Y("Outcome:N", title=None, axis=alt.Axis(grid=False)),
            color=alt.Color(
                "Outcome:N",
                title="Outcome",
                scale=alt.Scale(
                    domain=["Infects", "Does not infect"],
                    range=[theme.series_1, theme.series_2],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Candidates:Q", format=",")],
        )
    )

    labels = bars.mark_text(align="left", dx=6, color=theme.text_secondary, fontSize=11).encode(
        text=alt.Text("Candidates:Q", format=","), color=alt.value(theme.text_secondary)
    )

    chart = (bars + labels).properties(
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
        height=110,
    )

    return _base(chart, theme)


def projection_with_highlight(
    data: pd.DataFrame,
    theme: Theme,
    highlight: pd.DataFrame | None = None,
    x: str = "x",
    y: str = "y",
    label: str = "phage_id",
    title: str = "",
    subtitle: str = "",
    highlight_label: str = "Your upload",
) -> alt.LayerChart:
    """An embedding projection with one point called out.

    The known organisms form a grey context cloud -- they are the backdrop, not
    the subject -- and the uploaded one is drawn on top in the first categorical
    hue, larger, with a surface-coloured ring so it stays legible even where the
    cloud is dense. Identity is carried by the direct label beside the point as
    well as by colour, so it survives a colour-vision difference.
    """

    cloud = (
        alt.Chart(data)
        .mark_circle(size=POINT_SIZE * 0.7, color=context_grey(theme), opacity=0.5)
        .encode(
            x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            tooltip=[alt.Tooltip(f"{column}:N") for column in data.columns if column in {label, "bacteria_id"}],
        )
    )

    layers: List[alt.Chart] = [cloud]

    if highlight is not None and not highlight.empty:
        point = (
            alt.Chart(highlight)
            .mark_point(
                size=POINT_SIZE * 5,
                color=theme.series_1,
                filled=True,
                stroke=theme.surface,
                strokeWidth=SURFACE_GAP,
            )
            .encode(
                x=alt.X(f"{x}:Q"),
                y=alt.Y(f"{y}:Q"),
                tooltip=[alt.Tooltip(f"{label}:N", title=highlight_label)],
            )
        )
        text = (
            alt.Chart(highlight)
            .mark_text(
                align="left",
                dx=12,
                dy=-10,
                fontSize=11,
                fontWeight=600,
                color=theme.text_primary,
            )
            .encode(x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q"), text=alt.Text(f"{label}:N"))
        )
        layers += [point, text]

    chart = alt.layer(*layers).properties(
        height=360,
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
    )
    return _base(chart, theme)


def score_distribution(
    data: pd.DataFrame,
    theme: Theme,
    value: str = "score",
    threshold: float | None = None,
    title: str = "",
    subtitle: str = "",
    value_title: str = "Score",
    bins: int = 40,
) -> alt.LayerChart:
    """How a model's scores are spread, with the decision threshold marked.

    One series, so no legend: the title names what is counted. The threshold is
    a dashed rule rather than a second colour, which keeps the fill meaning one
    thing.
    """

    bars = (
        alt.Chart(data)
        .mark_bar(color=theme.series_1, cornerRadiusEnd=BAR_CORNER_RADIUS, stroke=theme.surface, strokeWidth=SURFACE_GAP)
        .encode(
            x=alt.X(f"{value}:Q", bin=alt.Bin(maxbins=bins), title=value_title),
            y=alt.Y("count():Q", title="Candidates"),
            tooltip=[
                alt.Tooltip(f"{value}:Q", bin=alt.Bin(maxbins=bins), title=value_title, format=".3f"),
                alt.Tooltip("count():Q", title="Candidates"),
            ],
        )
    )

    layers: List[alt.Chart] = [bars]

    if threshold is not None:
        rule = (
            alt.Chart(pd.DataFrame({value: [threshold]}))
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.5, color=theme.series_2)
            .encode(
                x=alt.X(f"{value}:Q"),
                tooltip=[alt.Tooltip(f"{value}:Q", title="Threshold", format=".2f")],
            )
        )
        layers.append(rule)

    chart = alt.layer(*layers).properties(
        height=240,
        title=alt.TitleParams(text=title, subtitle=subtitle) if title else alt.Undefined,
    )
    return _base(chart, theme)
