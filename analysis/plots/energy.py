# analysis/plots/energy.py

"""Energy analysis plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple

from .base import create_figure, add_grid, add_best_point_annotation, format_axis_ticks
from constants import (
    COLORS,
    ENERGY_INCREMENT,
    POWER_INCREMENT,
    LEGEND_SPACING,
    ROTATION,
    TICK_PADDING,
    DIMENSIONS,
    PLOT_PADDING,
)


def plot_energy_analysis(
    layer_metrics_df: pd.DataFrame, energy_analysis_df: pd.DataFrame, output_path: str
) -> None:
    """Plot energy analysis metrics showing mobile processing and data communication energy."""
    # Create figure with twin axes
    fig, ax = plt.subplots(figsize=DIMENSIONS["energy_analysis"])
    ax2 = ax.twinx()

    # Extract data
    split_layers = sorted(energy_analysis_df["Split Layer"].unique())
    mobile_energy = []
    comm_energy = []
    power_readings = []

    for split in split_layers:
        split_df = layer_metrics_df[layer_metrics_df["Split Layer"] == split]
        # Sum processing energy up to split point
        mobile_energy.append(
            split_df[split_df["Layer ID"] <= split]["Processing Energy (J)"].sum()
        )
        # Get communication energy at split point
        comm_energy.append(
            split_df[split_df["Layer ID"] == split]["Communication Energy (J)"].iloc[0]
        )
        # Get power reading and convert to mW
        power_readings.append(
            energy_analysis_df[energy_analysis_df["Split Layer"] == split][
                "Power Reading (W)"
            ].iloc[0]
            * 1000
        )

    # Plot stacked bars
    x = np.arange(len(split_layers))

    # Data communication (bottom)
    bars1 = ax.bar(
        x,
        comm_energy,
        color=COLORS["data_comm"],
        edgecolor="black",
        linewidth=0.5,
        label="Data communication",
        width=0.65,
    )

    # Mobile processing (top)
    bars2 = ax.bar(
        x,
        mobile_energy,
        bottom=comm_energy,
        color=COLORS["mobile_proc"],
        edgecolor="black",
        linewidth=0.5,
        label="Mobile processing",
        width=0.65,
    )

    # Power reading line
    line_power = ax2.plot(
        x,
        power_readings,
        color=COLORS["power_line"],
        linestyle="-",
        linewidth=1.0,
        label="Power (mW)",
        zorder=3,
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgecolor=COLORS["power_line"],
        markeredgewidth=1,
    )

    # Get layer names
    layer_names = []
    for split in split_layers:
        layer_type = layer_metrics_df[layer_metrics_df["Layer ID"] == split][
            "Layer Type"
        ].iloc[0]
        layer_names.append(layer_type)

    # Customize axes
    ax.set_ylabel("Energy (J)")
    ax2.set_ylabel("Power (mW)", color=COLORS["power_line"])
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=ROTATION, ha="right", va="top", fontsize=7)
    ax.tick_params(axis="x", pad=TICK_PADDING)

    # Calculate and annotate best energy point
    total_energy = [m + c for m, c in zip(mobile_energy, comm_energy)]
    best_idx = np.argmin(total_energy)
    best_energy = total_energy[best_idx]

    # Add best energy annotation
    add_best_point_annotation(
        ax=ax,
        x_pos=best_idx,
        y_pos=best_energy,
        text="Best energy",
        spacing=0.15,
        relative=False,
    )

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.08, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", right=False)
    ax2.tick_params(axis="y", colors=COLORS["power_line"])

    # Add legend
    handles = [bars1, bars2] + line_power
    labels = ["Data communication", "Mobile processing", "Power (mW)"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        **LEGEND_SPACING,
    )

    # Set axis limits and ticks
    format_energy_ticks(ax, max(total_energy))
    format_power_ticks(ax2, max(power_readings))

    # Save plot
    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()


def format_energy_ticks(ax: plt.Axes, max_energy: float) -> None:
    """Format energy axis ticks with 0.3J increments."""
    max_energy_rounded = np.ceil(max_energy / ENERGY_INCREMENT) * ENERGY_INCREMENT
    energy_ticks = np.arange(0, max_energy_rounded + ENERGY_INCREMENT, ENERGY_INCREMENT)
    ax.set_ylim(0, max_energy_rounded)
    ax.set_yticks(energy_ticks)
    ax.set_yticklabels([f"{x:.1f}" for x in energy_ticks])


def format_power_ticks(ax: plt.Axes, max_power: float) -> None:
    """Format power axis ticks with 300mW increments."""
    power_limit = np.ceil(max_power / POWER_INCREMENT) * POWER_INCREMENT
    ax.set_ylim(0, power_limit)
    ax.yaxis.set_major_locator(plt.MultipleLocator(POWER_INCREMENT))
