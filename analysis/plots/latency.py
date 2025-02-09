# analysis/plots/latency.py

"""Latency plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

from .base import create_figure, add_grid, add_best_point_annotation
from constants import (
    COLORS,
    BAR_WIDTH,
    LEGEND_SPACING,
    TICK_PADDING,
    ROTATION,
    DIMENSIONS,
    PLOT_PADDING,
)


def plot_layer_metrics(
    df: pd.DataFrame, split_df: pd.DataFrame, output_path: str
) -> None:
    """Create visualization for layer latency and output size."""
    # Create figure
    fig, ax1 = plt.subplots(figsize=DIMENSIONS["layer_metrics"])
    ax2 = ax1.twinx()

    # Get valid layer IDs
    valid_layer_ids = split_df["Split Layer Index"].unique()

    # Process layer metrics
    grouped = (
        df[df["Layer ID"].isin(valid_layer_ids)]
        .groupby("Layer ID")
        .agg(
            {
                "Layer Type": "first",
                "Layer Latency (ms)": "mean",
                "Output Size (MB)": "mean",
            }
        )
        .reset_index()
    )
    grouped = grouped.sort_values("Layer ID")

    # Set positions
    x = np.arange(len(grouped))

    # Plot metrics
    latency_bars = ax1.bar(
        x - BAR_WIDTH / 2,
        grouped["Layer Latency (ms)"],
        BAR_WIDTH,
        label="Layer latency",
        color=COLORS["gpu_energy"],
        edgecolor="black",
        linewidth=0.5,
    )

    size_bars = ax2.bar(
        x + BAR_WIDTH / 2,
        grouped["Output Size (MB)"],
        BAR_WIDTH,
        label="Size of output data",
        color=COLORS["battery"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add grid
    ax1.yaxis.grid(True, linestyle=":", alpha=0.3, color="gray")
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)

    # Customize axes
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Data size (MB)")

    # Set axis limits and ticks
    max_latency = max(grouped["Layer Latency (ms)"])
    max_size = max(grouped["Output Size (MB)"])

    # Format ticks
    format_latency_ticks(ax1, max_latency)
    format_size_ticks(ax2, max_size)

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        grouped["Layer Type"],
        rotation=ROTATION,
        ha="right",
        va="top",
        fontsize=7,
    )
    ax1.tick_params(axis="x", pad=TICK_PADDING)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=2,
        **LEGEND_SPACING,
    )

    # Clean up spines
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Save plot
    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()


def format_latency_ticks(ax: plt.Axes, max_latency: float) -> None:
    """Format latency axis ticks."""
    max_latency_rounded = np.ceil(max_latency / 5) * 5
    latency_ticks = np.arange(0, max_latency_rounded + 5, 5)
    ax.set_ylim(0, max_latency_rounded)
    ax.set_yticks(latency_ticks)
    ax.set_yticklabels([f"{x:.0f}" for x in latency_ticks])


def format_size_ticks(ax: plt.Axes, max_size: float) -> None:
    """Format size axis ticks."""
    max_size_rounded = np.ceil(max_size / 0.5) * 0.5
    size_ticks = np.arange(0, max_size_rounded + 0.5, 0.5)
    ax.set_ylim(0, max_size_rounded)
    ax.set_yticks(size_ticks)
    ax.set_yticklabels([f"{x:.1f}" for x in size_ticks])


def plot_overall_performance(
    df: pd.DataFrame, layer_metrics_df: pd.DataFrame, output_path: str
) -> None:
    """Create visualization for overall performance with stacked latency bars."""
    # Create figure with same height as energy plot
    fig, ax = plt.subplots(figsize=DIMENSIONS["overall_performance"])

    # Colors for stacked bars
    metrics = ["Server Time", "Travel Time", "Host Time"]
    labels = ["Server processing", "Data communication", "Mobile processing"]
    colors = [COLORS["server"], COLORS["communication"], COLORS["mobile"]]

    # Plot stacked bars
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for metric, color, label in zip(metrics, colors, labels):
        ax.bar(
            x,
            df[metric],
            bottom=bottom,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            width=0.65,
        )
        bottom += df[metric]

    # Get layer names
    layer_names = []
    for idx in df["Split Layer Index"]:
        layer_name = layer_metrics_df[layer_metrics_df["Layer ID"] == idx][
            "Layer Type"
        ].iloc[0]
        layer_names.append(layer_name)

    # Customize axes
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=ROTATION, ha="right", va="top", fontsize=7)
    ax.tick_params(axis="x", pad=TICK_PADDING)

    # Set y-axis limits and ticks
    max_total = df["Total Processing Time"].max()
    y_max = np.ceil(max_total / 50) * 50
    major_ticks = np.arange(0, y_max + 50, 50)
    ax.set_ylim(0, y_max)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f"{x:.0f}" for x in major_ticks])

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.08, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", right=False)

    # Add legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        fontsize=7,
        **LEGEND_SPACING,
    )

    # Find and annotate best latency
    total_latencies = df["Server Time"] + df["Travel Time"] + df["Host Time"]
    best_idx = total_latencies.idxmin()
    best_latency = total_latencies[best_idx]

    # Add best latency annotation
    add_best_point_annotation(
        ax=ax,
        x_pos=best_idx,
        y_pos=best_latency,
        text="Best latency",
        spacing=2.0,  # Adjusted to match energy plot proportions
        relative=True,
    )

    # Use consistent padding
    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
