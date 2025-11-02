"""This module contains utility functions for Plotly figures."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind


def standard_layout(fig: go.Figure, border: bool) -> go.Figure:
    """
    Standardize the layout of a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        figure
    border : bool
        whether to bound the plot on all sides

    Returns
    -------
    go.Figure
        standardized figure
    """
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(255, 255, 255, 1)",
        plot_bgcolor="rgba(255, 255, 255, 1)",
        font=dict(color="black", size=18, family="Arial"),
        title=dict(text=None, font=dict(color="black", size=22, family="Arial"), x=0.5, xref="paper"),
        xaxis=dict(title=None, showline=True, showticklabels=True, ticks="outside", mirror=border, tickfont=dict(color="black", size=18, family="Arial")),
        yaxis=dict(title=None, showline=True, showticklabels=True, ticks="outside", mirror=border, tickfont=dict(color="black", size=18, family="Arial")),
    )
    return fig


def annotation_t_test(x1: pd.Series, x2: pd.Series, symbolic_mode: bool) -> str:
    """
    Compute the p-value of a two-sample t-test.

    Parameters
    ----------
    x1 : pd.Series
        sample 1
    x2 : pd.Series
        sample 2
    symbolic_mode : bool
        use symbols for p-values

    Returns
    -------
    str
        p-value
    """
    p_value: float = ttest_ind(x1, x2, equal_var=False, alternative="two-sided")[1]
    if symbolic_mode:
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
    else:
        return f"p = {p_value:.6f}"


def annotation_cohens_d(x1: pd.Series, x2: pd.Series, symbolic_mode: bool) -> str:
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    x1 : pd.Series
        sample 1
    x2 : pd.Series
        sample 2
    symbolic_mode : bool
        use symbols for effect sizes

    Returns
    -------
    str
        effect size
    """
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(), x2.std()
    s_pooled: float = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohen_d: float = (x1.mean() - x2.mean()) / s_pooled
    if symbolic_mode:
        if cohen_d < 0.2:
            return "very small"
        elif cohen_d < 0.5:
            return "small"
        elif cohen_d < 0.8:
            return "medium"
        else:
            return "large"
    else:
        return f"d = {cohen_d:.2f}"


def annotation_tukey(df_tukey: pd.DataFrame, dv: str, comparison: tuple[str, str], symbolic_mode: bool) -> str:
    """
    Get the Tukey adjusted p-value for a pair of stages.

    Parameters
    ----------
    df_tukey : pd.DataFrame
        Tukey results, tukey_multiple_dvs() output
    dv : str
        dependent variable
    comparison : tuple[str, str]
        pair of stages
    symbolic_mode : bool
        use symbols for p-values

    Returns
    -------
    str
        p-value
    """
    p_value: float = df_tukey.at[dv, comparison]
    if symbolic_mode:
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
    else:
        return f"p = {p_value:.6f}"


def add_pairwise_comparison(
    fig: go.Figure,
    data: pd.DataFrame,
    stage_list: list[str],
    dv: str,
    comparison: str,
    method: str,
    height: int,
    width: int,
    interline: float,
    symbolic_mode: bool,
    tukey_results: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Add p-value annotations to a Plotly Figure.

    Parameters
    ----------
    fig : go.Figure
        figure
    data : pd.DataFrame
        data
    stage_list : list[str]
        list of stages
    dv : str
        dependent variable
    comparison : str
        condition for comparison
    method : str
        method for group comparison
    height : int
        plot height
    width : int
        plot width
    interline : float
        interline
    symbolic_mode : bool
        use symbols for p-values
    tukey_results : pd.DataFrame, optional
        Tukey results, by default None

    Returns
    -------
    go.Figure
        figure with annotations
    """
    # Define the pairwise comparisons
    # Each element of the outer list corresponds to a layer of comparisons
    # Each element of the inner list corresponds to a pair of columns to compare

    # The following code block generates all possible pairwise comparisons
    pairwise_comparisons: list[list[list[int]]] = []
    n: int = len(stage_list)
    for i in range(1, n):
        if i == 1:
            layer: list[list[int]] = []
            for j in range(n - i):
                layer.append([j, j + i])
            pairwise_comparisons.append(layer)
        else:
            for j in range(n - i):
                pairwise_comparisons.append([[j, j + i]])

    # Alternatively, customize pairwise_comparisons yourself
    # pairwise_comparisons: list[list[list[int]]] = [
    #     [[0, 1], [1, 2], [2, 3]],
    #     [[0, 2]],
    #     [[1, 3]],
    #     [[0, 3]],
    # ]

    if comparison == "none":
        for layer_index, layer in enumerate(pairwise_comparisons):
            for column_pair in layer:
                # Compute the left, right, bottom, and top coordinates
                xl: float = column_pair[0] + 30 / width
                xr: float = column_pair[1] - 30 / width
                yb: float = 1 + interline * layer_index / height
                yt: float = 1.05 + interline * layer_index / height

                # Subset the two groups for comparison
                group1: pd.Series = data.loc[data["stage"].isin([stage_list[column_pair[0]]]), dv].dropna()
                group2: pd.Series = data.loc[data["stage"].isin([stage_list[column_pair[1]]]), dv].dropna()

                # Compute the p-value or effect size based on the method
                if method == "t-test p-value":
                    symbol: str = annotation_t_test(group1, group2, symbolic_mode)
                elif method == "Cohen's D effsize":
                    symbol: str = annotation_cohens_d(group1, group2, symbolic_mode)
                elif method == "Tukey p-value" and tukey_results is not None:
                    symbol: str = annotation_tukey(tukey_results, dv, (stage_list[column_pair[0]], stage_list[column_pair[1]]), symbolic_mode)
                else:
                    raise ValueError("Invalid method")

                # Left vertical whisker
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y domain",
                    x0=xl,
                    y0=yb,
                    x1=xl,
                    y1=yt,
                    line=dict(color="black", width=1),
                    opacity=1,
                )

                # Right vertical whisker
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y domain",
                    x0=xr,
                    y0=yb,
                    x1=xr,
                    y1=yt,
                    line=dict(color="black", width=1),
                    opacity=1,
                )

                # Horizontal bar
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y domain",
                    x0=xl,
                    y0=yt,
                    x1=xr,
                    y1=yt,
                    line=dict(color="black", width=1),
                    opacity=1,
                )

                # Annotation
                fig.add_annotation(
                    xref="x",
                    yref="y domain",
                    x=(xl + xr) / 2,
                    y=yt,
                    text=symbol,
                    textangle=0,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    xanchor="center",
                    yanchor="bottom",
                    opacity=1,
                )
        return fig

    for ii, group in enumerate(stage_list):
        # Compute the left, right, bottom, and top coordinates
        xl: float = ii - 120 / width
        xr: float = ii + 120 / width
        yb: float = 1 + interline / height
        yt: float = 1.05 + interline / height

        # Subset the two groups for comparison
        group1: pd.Series = data.loc[~data[comparison] & data["stage"].isin([group]), dv].dropna()
        group2: pd.Series = data.loc[data[comparison] & data["stage"].isin([group]), dv].dropna()

        # Compute the statistic given input method
        if method == "t-test p-value":
            symbol: str = annotation_t_test(group1, group2, symbolic_mode)
        elif method == "Cohen's effsize":
            symbol: str = annotation_cohens_d(group1, group2, symbolic_mode)

        # Left vertical whisker
        fig.add_shape(
            type="line",
            xref="x",
            yref="y domain",
            x0=xl,
            y0=yb,
            x1=xl,
            y1=yt,
            line=dict(color="black", width=1),
            opacity=1,
        )

        # Right vertical whisker
        fig.add_shape(
            type="line",
            xref="x",
            yref="y domain",
            x0=xr,
            y0=yb,
            x1=xr,
            y1=yt,
            line=dict(color="black", width=1),
            opacity=1,
        )

        # Horizontal bar
        fig.add_shape(
            type="line",
            xref="x",
            yref="y domain",
            x0=xl,
            y0=yt,
            x1=xr,
            y1=yt,
            line=dict(color="black", width=1),
            opacity=1,
        )

        # Annotation
        fig.add_annotation(
            xref="x",
            yref="y domain",
            x=ii,
            y=yt,
            text=symbol,
            textangle=0,
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom",
            opacity=1,
        )
    return fig


def add_box(
    fig: go.Figure,
    data: pd.DataFrame,
    stage_list: list,
    dv: str,
    comparison: str,
    width: int,
) -> go.Figure:
    """
    Draw box plot on a strip plot.

    Parameters
    ----------
    fig : go.Figure
        figure
    data : pd.DataFrame
        data
    stage_list : list[str]
        stage list
    dv : str
        dependent variable
    comparison : str
        comparison condition
    width : int
        plot width

    Returns
    -------
    go.Figure
        strip plot with box
    """
    # For each stage
    for x_ind, stage in enumerate(stage_list):

        # Subset the current stage
        subset: pd.DataFrame = data.loc[data["stage"] == stage]

        if comparison == "none":
            # If no comparison is made, draw a single box for each stage
            _format_c: dict[str, str | float] = {"color": "black", "width": 1.5}
            xwidth: float = 125 / width

            # Compute the quantiles
            ymin_c, y25_c, y50_c, y75_c, ymax_c = subset[dv].quantile([0, 1 / 4, 1 / 2, 3 / 4, 1])

            # Pack the box parameters into a single-tuple-element list
            tup: list[tuple[dict, float, float, float, float, float, float]] = [(_format_c, x_ind, ymin_c, y25_c, y50_c, y75_c, ymax_c)]
        else:
            # If a comparison is made, draw two boxes for each stage
            _format_l: dict[str, str | float] = {"color": "#190FCA", "width": 2}
            _format_r: dict[str, str | float] = {"color": "#D21414", "width": 2}
            xwidth: float = 100 / width
            xcenter_l: float = x_ind - 1.5 * xwidth
            xcenter_r: float = x_ind + 1.5 * xwidth

            # Compute the quantiles
            ymin_l, y25_l, y50_l, y75_l, ymax_l = subset.loc[~subset[comparison].astype(bool), dv].quantile([0, 1 / 4, 1 / 2, 3 / 4, 1])
            ymin_r, y25_r, y50_r, y75_r, ymax_r = subset.loc[subset[comparison].astype(bool), dv].quantile([0, 1 / 4, 1 / 2, 3 / 4, 1])

            # Pack the box parameters into a two-tuple-element list
            tup: list[tuple[dict, float, float, float, float, float, float]] = [
                (_format_l, xcenter_l, ymin_l, y25_l, y50_l, y75_l, ymax_l),
                (_format_r, xcenter_r, ymin_r, y25_r, y50_r, y75_r, ymax_r),
            ]

        # Iterate over the box parameters
        for _format, xcenter, ymin, y25, y50, y75, ymax in tup:

            # Compute the interquartile range and bounding whiskers
            iqr: float = y75 - y25
            ylower: float = max(ymin, y25 - 1.5 * iqr)
            yupper: float = min(ymax, y75 + 1.5 * iqr)

            # Verticle line from box to upper whisker
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xcenter,
                y0=y75,
                x1=xcenter,
                y1=yupper,
                line=_format,
                opacity=1,
            )

            # Upper horizontal whisker
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xcenter - 0.8 * xwidth,
                y0=yupper,
                x1=xcenter + 0.8 * xwidth,
                y1=yupper,
                line=_format,
                opacity=1,
            )

            #  Verticle line from box to lower whisker
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xcenter,
                y0=ylower,
                x1=xcenter,
                y1=y25,
                line=_format,
                opacity=1,
            )

            # Lower horizontal whisker
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xcenter - 0.8 * xwidth,
                y0=ylower,
                x1=xcenter + 0.8 * xwidth,
                y1=ylower,
                line=_format,
                opacity=1,
            )

            # Median horizontal line
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xcenter - xwidth,
                y0=y50,
                x1=xcenter + xwidth,
                y1=y50,
                line=_format,
                opacity=1,
            )

            # Rectangular box
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=xcenter - xwidth,
                y0=y25,
                x1=xcenter + xwidth,
                y1=y75,
                line=_format,
                opacity=1,
                fillcolor="rgba(0, 0, 0, 0)",
            )
    return fig
