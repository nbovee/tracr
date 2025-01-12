#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

logger = logging.getLogger(__name__)


def read_excel_data(excel_path: str) -> Dict[str, pd.DataFrame]:
    """Read all required sheets from the Excel file."""
    data = {}

    # Read named sheets
    sheet_names = {
        "overall_performance": "Overall Performance",
        "layer_metrics": "Layer Metrics",
        "energy_analysis": "Energy Analysis",
    }

    for key, sheet_name in sheet_names.items():
        try:
            data[key] = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.debug(f"Successfully read sheet '{sheet_name}'")
        except Exception as e:
            print(f"Warning: Could not read sheet '{sheet_name}': {e}")
            data[key] = None

    return data


def validate_dataframe(df: pd.DataFrame, required_cols: list, sheet_name: str) -> None:
    """Validate that DataFrame contains required columns."""
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Excel sheet '{sheet_name}' must contain columns: {required_cols}"
        )


def plot_layer_metrics_tab(
    df: pd.DataFrame, split_df: pd.DataFrame, output_path: str
) -> None:
    """Create visualization for 'Layer Metrics' tab with per-layer latency and output size."""
    # Validate required columns
    required_cols = [
        "Split Layer",
        "Layer ID",
        "Layer Type",
        "Layer Latency (ms)",
        "Output Size (MB)",
    ]
    validate_dataframe(df, required_cols, "Layer Metrics")

    # Set style
    _set_plot_style()

    # Create figure with reduced height
    fig, ax1 = plt.subplots(figsize=(8, 2))  # Reduced height from 3 to 2
    ax2 = ax1.twinx()

    # Get valid layer IDs from overall performance tab
    valid_layer_ids = split_df["Split Layer Index"].unique()

    # Process layer metrics - use Layer ID directly for ordering
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

    # Set bar width and positions
    bar_width = 0.35
    x = np.arange(len(grouped))

    # Color scheme (matching JMLR style)
    color_latency = "#a1c9f4"  # Light blue
    color_size = "#2c3e50"  # Dark blue

    # Plot layer metrics
    latency_bars = ax1.bar(
        x - bar_width / 2,
        grouped["Layer Latency (ms)"],
        bar_width,
        label="Layer latency",
        color=color_latency,
        edgecolor="black",
        linewidth=0.5,
    )

    size_bars = ax2.bar(
        x + bar_width / 2,
        grouped["Output Size (MB)"],
        bar_width,
        label="Size of output data",
        color=color_size,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add only horizontal grid lines
    ax1.yaxis.grid(True, linestyle=":", alpha=0.3, color="gray")
    ax1.xaxis.grid(False)  # Disable vertical grid lines
    ax1.set_axisbelow(True)

    # Customize axes
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Data size (MB)")

    # Set axis limits and ticks with specific increments
    max_latency = max(grouped["Layer Latency (ms)"])
    max_size = max(grouped["Output Size (MB)"])

    # Left y-axis (Latency): increments of 10
    max_latency_rounded = np.ceil(max_latency / 10) * 10
    latency_ticks = np.arange(0, max_latency_rounded + 10, 10)
    ax1.set_ylim(0, max_latency_rounded)
    ax1.set_yticks(latency_ticks)

    # Right y-axis (Output size): increments of 0.3
    max_size_rounded = np.ceil(max_size / 0.3) * 0.3
    size_ticks = np.arange(0, max_size_rounded + 0.3, 0.3)
    ax2.set_ylim(0, max_size_rounded)
    ax2.set_yticks(size_ticks)

    # Set x-axis labels with reduced padding
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        grouped["Layer Type"], rotation=90, ha="center", va="top", fontsize=7
    )
    ax1.tick_params(axis="x", pad=2)  # Reduce padding between ticks and labels

    # Add legend with adjusted position
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
        columnspacing=1,
        handletextpad=0.5,
        borderaxespad=0.2,
    )  # Reduced borderaxespad

    # Clean up spines
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Adjust layout and save with reduced padding
    plt.tight_layout(pad=0.2)  # Reduced padding in tight_layout
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", pad_inches=0.01
    )  # Reduced pad_inches
    plt.close()


def plot_overall_performance_tab(
    df: pd.DataFrame, layer_metrics_df: pd.DataFrame, output_path: str
) -> None:
    """Create visualization for 'Overall Performance' tab with stacked latency bars."""
    # Validate required columns
    required_cols = [
        "Split Layer Index",
        "Host Time",
        "Travel Time",
        "Server Time",
        "Total Processing Time",
    ]
    validate_dataframe(df, required_cols, "Overall Performance")

    # Set style
    _set_plot_style()

    # Create figure with extra space at top for legend
    fig, ax = plt.subplots(figsize=(8, 3))

    # Colors matching the reference plot
    colors = [
        "#4a6fa5",  # Server processing (dark blue)
        "#93b7be",  # Data communication (medium blue)
        "#c7dbe6",  # Mobile processing (light blue)
    ]

    # Plot stacked bars
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    metrics = ["Server Time", "Travel Time", "Host Time"]
    labels = ["Server processing", "Data communication", "Mobile processing"]

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

    # Get layer names from layer_metrics_df
    layer_names = []
    for idx in df["Split Layer Index"]:
        layer_name = layer_metrics_df[layer_metrics_df["Layer ID"] == idx][
            "Layer Type"
        ].iloc[0]
        layer_names.append(layer_name)

    # Customize axes
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=90, ha="center", va="top", fontsize=7)

    # Set y-axis limits and ticks
    y_max = 35
    ax.set_ylim(0, y_max)
    major_ticks = np.arange(0, y_max + 5, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f"{x:.0f}" for x in major_ticks])

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.08, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", right=False)

    # Add legend at the top of the plot
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        fontsize=7,
    )

    # Find the bar with minimum total latency
    total_latencies = df["Server Time"] + df["Travel Time"] + df["Host Time"]
    best_idx = total_latencies.idxmin()
    best_latency = total_latencies[best_idx]

    # Calculate positions for annotation with consistent spacing
    spacing = 1.5  # Consistent spacing between elements
    text_height = best_latency + 8  # Text at top
    star_height = text_height - spacing  # Star below text
    arrow_end = best_latency  # Arrow end at bar top

    # Add "Best latency" text at top
    ax.text(best_idx, text_height, "Best latency", ha="center", va="bottom", fontsize=7)

    # Add star below text with black edge
    ax.plot(
        best_idx,
        star_height,
        marker="*",
        markersize=10,
        color="#ffd700",
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
    )

    # Add simple vertical arrow pointing down to the bar
    ax.annotate(
        "",
        xy=(best_idx, arrow_end),  # arrow tip at bar top
        xytext=(best_idx, star_height - spacing),  # arrow starts below star
        ha="center",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            linewidth=0.8,
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
        ),
    )

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_raw_power_metrics_tab(df: pd.DataFrame, output_path: str) -> None:
    """Create visualization for 'Raw Power Metrics' tab showing CPU, memory, and battery usage."""
    # Set style
    _set_plot_style()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[1, 1])

    # Plot CPU metrics
    cpu_cols = [col for col in df.columns if "cpu" in col.lower()]
    for col in cpu_cols:
        ax1.plot(df.index, df[col], label=col, linewidth=1)

    ax1.set_ylabel("CPU Usage (%)")
    ax1.grid(True, alpha=0.08, color="gray")
    ax1.legend(loc="upper right", ncol=2, fontsize=7)

    # Plot memory and battery metrics
    mem_cols = [col for col in df.columns if "memory" in col.lower()]
    bat_cols = [col for col in df.columns if "battery" in col.lower()]

    for col in mem_cols + bat_cols:
        ax2.plot(df.index, df[col], label=col, linewidth=1)

    ax2.set_ylabel("Usage")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.08, color="gray")
    ax2.legend(loc="upper right", ncol=2, fontsize=7)

    # Clean up spines
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", labelsize=7)

    # Save plot with consistent padding
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_energy_analysis_tab(
    df: pd.DataFrame, layer_metrics_df: pd.DataFrame, output_path: str
) -> None:
    """Plot energy analysis metrics showing mobile processing and data communication energy."""
    # Set style
    _set_plot_style()

    # Create figure with extra space at top for legend
    fig, ax = plt.subplots(figsize=(8, 3))
    ax2 = ax.twinx()  # Create second y-axis for power metrics

    # Get valid layer IDs from overall performance tab
    valid_layer_ids = df[df["Layer Type"] != "Detect"]["Layer ID"].unique()

    # Extract data for each split layer
    split_layers = sorted(
        df[df["Layer ID"].isin(valid_layer_ids)]["Split Layer"].unique()
    )
    mobile_energy = []
    comm_energy = []
    power_readings = []

    for split in split_layers:
        split_df = df[df["Split Layer"] == split]
        # Sum processing energy up to split point
        mobile_energy.append(
            split_df[split_df["Layer ID"] <= split]["Processing Energy (J)"].sum()
        )
        # Get communication energy at split point
        comm_energy.append(
            split_df[split_df["Layer ID"] == split]["Communication Energy (J)"].iloc[0]
        )
        # Get power reading at split point
        power_readings.append(
            split_df[split_df["Layer ID"] == split]["Power Reading (W)"].iloc[0] * 1000
        )  # Convert W to mW

    # Colors matching the reference image
    color_comm = "#e67e22"  # Darker orange for data communication
    color_mobile = "#ffd4b2"  # Light orange for mobile processing
    color_power = "#2c3e50"  # Dark blue for power line (JMLR style)

    # Plot stacked bars
    x = np.arange(len(split_layers))

    # First plot data communication (darker orange, bottom)
    bars1 = ax.bar(
        x,
        comm_energy,
        color=color_comm,
        edgecolor="black",
        linewidth=0.5,
        label="Data communication",
        width=0.65,
    )

    # Then plot mobile processing on top (lighter orange)
    bars2 = ax.bar(
        x,
        mobile_energy,
        bottom=comm_energy,
        color=color_mobile,
        edgecolor="black",
        linewidth=0.5,
        label="Mobile processing",
        width=0.65,
    )

    # Plot power reading as a line with improved JMLR styling
    line_power = ax2.plot(
        x,
        power_readings,
        color=color_power,
        linestyle="-",
        linewidth=1.0,
        label="Power (mW)",
        zorder=3,
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgecolor=color_power,
        markeredgewidth=1,
    )

    # Get layer names from layer_metrics_df
    layer_names = []
    for split in split_layers:
        layer_type = layer_metrics_df[layer_metrics_df["Layer ID"] == split][
            "Layer Type"
        ].iloc[0]
        layer_names.append(layer_type)

    # Customize axes
    ax.set_ylabel("Energy (J)")
    ax2.set_ylabel("Power (mW)", color=color_power)  # Match power line color
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=90, ha="center", va="top", fontsize=7)

    # Calculate total energy and find the best index
    total_energy = [m + c for m, c in zip(mobile_energy, comm_energy)]
    best_idx = np.argmin(total_energy)
    best_latency = total_energy[best_idx]

    # Calculate positions for annotation with consistent spacing
    spacing = 0.05  # Consistent spacing between elements
    text_height = best_latency + 0.2  # Text at top
    star_height = text_height - spacing  # Star below text
    arrow_end = best_latency  # Arrow end at bar top

    # Add "Best energy" text at top
    ax.text(best_idx, text_height, "Best energy", ha="center", va="bottom", fontsize=7)

    # Add star below text with black edge
    ax.plot(
        best_idx,
        star_height,
        marker="*",
        markersize=10,
        color="#ffd700",
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
    )

    # Add simple vertical arrow pointing down to the bar
    ax.annotate(
        "",
        xy=(best_idx, arrow_end),  # arrow tip at bar top
        xytext=(best_idx, star_height - spacing),  # arrow starts below star
        ha="center",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            linewidth=0.8,
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
        ),
    )

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.08, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", right=False)
    ax2.tick_params(axis="y", colors=color_power)  # Match power line color

    # Add legend at the top of the plot
    # Get handles and labels in the correct order
    handles = [bars1, bars2] + line_power
    labels = ["Data communication", "Mobile processing", "Power (mW)"]

    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        fontsize=7,
    )

    # Set y-axis limits and ticks for energy
    ymax = max(total_energy)
    # Round up to nearest 0.1
    ymax_rounded = np.ceil(ymax * 10) / 10
    ax.set_ylim(0, ymax_rounded)
    # Set ticks at 0.1 intervals
    yticks = np.arange(0, ymax_rounded + 0.1, 0.1)
    ax.set_yticks(yticks)
    # Format tick labels to 1 decimal place
    ax.set_yticklabels([f"{x:.1f}" for x in yticks])

    # Set y-axis limits for power
    max_power = max(power_readings)
    power_limit = np.ceil(max_power / 100) * 100  # Round up to nearest 100
    ax2.set_ylim(0, power_limit)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(200))  # 100-unit increments

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def _set_plot_style() -> None:
    """Set consistent plot style across all visualizations."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,  # Base font size
            "axes.labelsize": 8,  # Axis label size
            "axes.titlesize": 8,  # Title size
            "xtick.labelsize": 7,  # X-tick label size
            "ytick.labelsize": 7,  # Y-tick label size
            "legend.fontsize": 7,  # Legend font size
            "figure.dpi": 300,
            "axes.grid": False,  # No grid by default
            "grid.alpha": 0.08,  # Consistent grid transparency
            "grid.color": "gray",  # Consistent grid color
            "grid.linestyle": "-",  # Consistent grid style
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from Excel metrics"
    )
    parser.add_argument("excel_path", help="Path to the Excel file containing metrics")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory for plots (default: current directory)",
    )
    parser.add_argument(
        "--plot-type",
        "-t",
        choices=["layer_metrics", "overall_performance", "energy_analysis", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Read all data at once
        data = read_excel_data(args.excel_path)

        if args.plot_type in ["layer_metrics", "all"]:
            if data["layer_metrics"] is not None:
                output_path = os.path.join(args.output_dir, "layer_metrics.png")
                plot_layer_metrics_tab(
                    data["layer_metrics"], data["overall_performance"], output_path
                )
                print(f"Layer metrics plot saved to: {output_path}")
            else:
                print(
                    "Warning: Could not generate layer metrics plot - required sheet not found"
                )

        if args.plot_type in ["overall_performance", "all"]:
            if data["overall_performance"] is not None:
                output_path = os.path.join(args.output_dir, "overall_performance.png")
                plot_overall_performance_tab(
                    data["overall_performance"], data["layer_metrics"], output_path
                )
                print(f"Overall performance plot saved to: {output_path}")
            else:
                print(
                    "Warning: Could not generate overall performance plot - required sheet not found"
                )

        if args.plot_type in ["energy_analysis", "all"]:
            if data["layer_metrics"] is not None:
                output_path = os.path.join(args.output_dir, "energy_analysis.png")
                plot_energy_analysis_tab(
                    data["layer_metrics"], data["layer_metrics"], output_path
                )
                print(f"Energy analysis plot saved to: {output_path}")
            else:
                print(
                    "Warning: Could not generate energy analysis plot - required sheet not found"
                )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
