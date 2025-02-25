# analysis/plots/base.py

"""Base plotting utilities and common functions."""

import matplotlib.pyplot as plt
from typing import Tuple
from constants import STYLE_CONFIG, GRID_STYLE


def set_plot_style() -> None:
    """Set consistent plot style across all visualizations."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(STYLE_CONFIG)


def create_figure(plot_type: str) -> Tuple[plt.Figure, plt.Axes]:
    """Create figure with consistent styling."""
    from constants import DIMENSIONS

    fig, ax = plt.subplots(figsize=DIMENSIONS[plot_type])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def add_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add grid lines with consistent styling."""
    if axis == "y":
        ax.yaxis.grid(True, **GRID_STYLE)
    else:
        ax.xaxis.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)


def add_best_point_annotation(
    ax: plt.Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    spacing: float = 0.15,
    relative: bool = False,
) -> None:
    """Add annotation with star and arrow for best point."""
    from constants import ANNOTATION

    if relative:
        # Get the y-axis limits and calculate equivalent spacing
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        # Scale the spacing to match the energy plot's proportions
        # Energy plot: 0.15 spacing in 0.6 range = 0.25 ratio
        scale = y_range * 0.25  # This will give same visual proportion
        spacing = scale

    # Fixed proportions that work well for both plots
    text_height = y_pos + spacing * 1.5
    star_height = y_pos + spacing * 1.0
    arrow_start = star_height - spacing * 0.3
    arrow_end = y_pos + spacing * 0.05

    # Add text
    ax.text(
        x_pos,
        text_height,
        text,
        ha="center",
        va="bottom",
        fontsize=ANNOTATION["text_size"],
    )

    # Add star
    ax.plot(
        x_pos,
        star_height,
        marker="*",
        markersize=ANNOTATION["star_size"],
        color=ANNOTATION["star_color"],
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
    )

    # Add arrow
    ax.annotate(
        "",
        xy=(x_pos, arrow_end),
        xytext=(x_pos, arrow_start),
        arrowprops=dict(
            arrowstyle=ANNOTATION["arrow_style"],
            color="black",
            linewidth=1.0,
            shrinkA=0,
            shrinkB=0,
        ),
    )


def format_axis_ticks(
    ax: plt.Axes, max_val: float, increment: float, format_str: str = "{:.1f}"
) -> None:
    """Format axis ticks with consistent increments."""
    import numpy as np

    max_rounded = np.ceil(max_val / increment) * increment
    ticks = np.arange(0, max_rounded + increment, increment)
    ax.set_ylim(0, max_rounded)
    ax.set_yticks(ticks)
    ax.set_yticklabels([format_str.format(x) for x in ticks])
