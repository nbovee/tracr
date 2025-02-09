# analysis/plots/comparative.py

"""Comparative plotting functions for multiple models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .base import add_grid, add_best_point_annotation
from constants import (
    COLORS,
    DIMENSIONS,
    PLOT_PADDING,
    LEGEND_SPACING,
    ROTATION,
    TICK_PADDING,
    BAR_WIDTH,
    ANNOTATION,
)


def plot_comparative_latency(
    model_data: Dict[str, Dict[str, pd.DataFrame]], output_path: str
) -> None:
    """Create comparative visualization of overall latency for different models."""
    fig, ax = plt.subplots(figsize=DIMENSIONS["overall_performance"])

    # Track best points for each model
    best_points = {}
    max_total = 0

    # Plot bars for each model side by side
    models = list(model_data.keys())
    x = np.arange(len(model_data[models[0]]["overall_performance"]))

    # Define professional color schemes for each model
    model_colors = {
        models[0]: {  # First model (YOLOv5s)
            "server": "#1f77b4",  # Strong blue
            "communication": "#7393b3",  # Steel blue
            "mobile": "#bdd7e7",  # Light blue
        },
        models[1]: {  # Second model (YOLOv8s)
            "server": "#2ca02c",  # Strong green
            "communication": "#69b3a2",  # Sage green
            "mobile": "#b5e5bf",  # Light green
        },
    }

    for i, model in enumerate(models):
        df = model_data[model]["overall_performance"]
        metrics = ["Server Time", "Travel Time", "Host Time"]
        labels = [f"{model} {m.replace(' Time', '')}" for m in metrics]
        colors = [
            model_colors[model]["server"],
            model_colors[model]["communication"],
            model_colors[model]["mobile"],
        ]

        # Offset bars for each model
        offset = (i - 0.5) * BAR_WIDTH
        bottom = np.zeros(len(df))

        for metric, color, label in zip(metrics, colors, labels):
            ax.bar(
                x + offset,
                df[metric],
                BAR_WIDTH,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=label,
                alpha=1.0,  # Full opacity for professional look
            )
            bottom += df[metric]

        # Track best point and maximum total
        total_latencies = df["Total Processing Time"]
        max_total = max(max_total, total_latencies.max())
        best_idx = total_latencies.idxmin()
        best_latency = total_latencies[best_idx]
        best_points[model] = (best_idx + offset, best_latency)

    # Add best point annotations with improved positioning
    for i, (model, (x_pos, y_pos)) in enumerate(best_points.items()):
        # Determine curve direction based on which model's best point is leftmost
        leftmost_x = min(p[0] for p in best_points.values())
        curve_direction = (
            -1 if x_pos == leftmost_x else 1
        )  # Left curve for leftmost point

        # Calculate positions with more space above bars
        text_x = x_pos + (
            0.8 * curve_direction
        )  # Reduced horizontal offset for more vertical arrows
        text_y = y_pos + 8.0  # Higher vertical position for more vertical appearance

        # Add star with more space above bar
        star_y = y_pos + 2.0  # Increased vertical spacing from bar
        ax.plot(
            x_pos,
            star_y,
            marker="*",
            markersize=ANNOTATION["star_size"],
            color=ANNOTATION["star_color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )

        # Add text with matching color from model's scheme
        text_color = model_colors[model]["server"]

        # Add annotation with arrow
        ax.annotate(
            f"Best {model}",
            xy=(x_pos, star_y + 0.8),  # Increased gap between arrow and star
            xytext=(text_x, text_y),  # Text position
            color=text_color,
            fontsize=ANNOTATION["text_size"],
            ha=(
                "right" if curve_direction < 0 else "left"
            ),  # Align text based on curve direction
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad={0.05 * curve_direction}",  # Much reduced curve for near-vertical path
                color=text_color,
                linewidth=1.0,
                shrinkA=0,
                shrinkB=2,  # Add small gap at arrow end
            ),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
            zorder=6,
        )

    # Customize axes
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x)

    # Get layer names from first model (assuming same layers)
    layer_names = []
    df = model_data[models[0]]["layer_metrics"]
    for idx in model_data[models[0]]["overall_performance"]["Split Layer Index"]:
        layer_name = df[df["Layer ID"] == idx]["Layer Type"].iloc[0]
        layer_names.append(layer_name)

    ax.set_xticklabels(layer_names, rotation=ROTATION, ha="right", va="top", fontsize=7)
    ax.tick_params(axis="x", pad=TICK_PADDING)

    # Set y-axis limits and ticks
    y_max = np.ceil(max_total / 5) * 5
    major_ticks = np.arange(0, y_max + 5, 5)
    ax.set_ylim(0, y_max)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f"{x:.0f}" for x in major_ticks])

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    add_grid(ax)

    # Reorganize legend items to group by component type
    handles, labels = ax.get_legend_handles_labels()

    # Reorder handles and labels to group by component
    ordered_labels = []
    ordered_handles = []

    # Add Server components
    for model in models:
        ordered_labels.append(f"{model} Server")
        ordered_handles.append(handles[labels.index(f"{model} Server")])

    # Add Travel/Communication components
    for model in models:
        ordered_labels.append(f"{model} Travel")
        ordered_handles.append(handles[labels.index(f"{model} Travel")])

    # Add Host/Mobile components
    for model in models:
        ordered_labels.append(f"{model} Host")
        ordered_handles.append(handles[labels.index(f"{model} Host")])

    # Add legend with reordered items
    legend_params = LEGEND_SPACING.copy()  # Create a copy of the spacing parameters
    legend_params["columnspacing"] = 1.0  # Override columnspacing
    legend_params["handletextpad"] = 0.5  # Add handletextpad

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,  # Three columns for three metric types
        frameon=False,
        fontsize=7,
        **legend_params,
    )

    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()


def plot_comparative_energy(
    model_data: Dict[str, Dict[str, pd.DataFrame]], output_path: str
) -> None:
    """Create comparative visualization of energy consumption for different models."""
    fig, ax = plt.subplots(figsize=DIMENSIONS["energy_analysis"])
    ax2 = ax.twinx()  # For power readings

    # Plot bars for each model side by side
    models = list(model_data.keys())  # Define models first
    x = np.arange(len(model_data[models[0]]["energy_analysis"]))

    # Define professional color schemes for each model - Energy-themed colors
    model_colors = {
        models[0]: {  # First model (YOLOv5s)
            "processing": "#e63946",  # Deep red/coral
            "communication": "#f4a261",  # Light orange
            "power": "#9d0208",  # Dark red for power line
        },
        models[1]: {  # Second model (YOLOv8s)
            "processing": "#fb8b24",  # Bright orange
            "communication": "#ffd6a5",  # Light peach
            "power": "#bb3e03",  # Dark orange for power line
        },
    }

    # Track best points and maximum values
    best_points = {}
    max_energy = 0
    max_power = 0

    for i, model in enumerate(models):
        layer_metrics_df = model_data[model]["layer_metrics"]
        energy_df = model_data[model]["energy_analysis"]

        # Extract data
        split_layers = sorted(energy_df["Split Layer"].unique())
        mobile_energy = []
        comm_energy = []
        power_readings = []

        for split in split_layers:
            split_df = layer_metrics_df[layer_metrics_df["Split Layer"] == split]
            mobile_energy.append(
                split_df[split_df["Layer ID"] <= split]["Processing Energy (J)"].sum()
            )
            comm_energy.append(
                split_df[split_df["Layer ID"] == split][
                    "Communication Energy (J)"
                ].iloc[0]
            )
            power_readings.append(
                energy_df[energy_df["Split Layer"] == split]["Power Reading (W)"].iloc[
                    0
                ]
                * 1000
            )

        # Offset bars for each model
        offset = (i - 0.5) * BAR_WIDTH

        # Plot stacked bars with new colors
        bars1 = ax.bar(
            x + offset,
            comm_energy,
            BAR_WIDTH,
            color=model_colors[model]["communication"],
            edgecolor="black",
            linewidth=0.5,
            label=f"{model} Communication",
            alpha=1.0,
        )

        bars2 = ax.bar(
            x + offset,
            mobile_energy,
            BAR_WIDTH,
            bottom=comm_energy,
            color=model_colors[model]["processing"],
            edgecolor="black",
            linewidth=0.5,
            label=f"{model} Processing",
            alpha=1.0,
        )

        # Plot power line with new style
        line = ax2.plot(
            x + offset,
            power_readings,
            color=model_colors[model]["power"],
            linestyle="-" if i == 0 else "--",
            linewidth=1.5,
            label=f"{model} Power",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=model_colors[model]["power"],
            markeredgewidth=1.5,
            alpha=1.0,
        )

        # Track best points and maximum values
        total_energy = [m + c for m, c in zip(mobile_energy, comm_energy)]
        max_energy = max(max_energy, max(total_energy))
        max_power = max(max_power, max(power_readings))
        best_idx = np.argmin(total_energy)
        best_energy = total_energy[best_idx]
        best_points[model] = (best_idx + offset, best_energy)

    # Add best point annotations with improved positioning
    for i, (model, (x_pos, y_pos)) in enumerate(best_points.items()):
        # YOLOv8s (i=1) vertical, YOLOv5s (i=0) curved right
        if i == 0:  # YOLOv5s
            text_x = x_pos + 1.0  # Move text right
            text_y = y_pos + 0.15  # Lower text position
            curve_amount = 0.15  # More curve
        else:  # YOLOv8s
            text_x = x_pos  # Keep text directly above
            text_y = y_pos + 0.3  # Higher text position
            curve_amount = 0.0  # Perfectly vertical

        # Add star closer to bar
        star_y = y_pos + 0.03  # Reduced gap between bar and star
        ax.plot(
            x_pos,
            star_y,
            marker="*",
            markersize=ANNOTATION["star_size"],
            color=ANNOTATION["star_color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )

        # Add text with matching color
        text_color = model_colors[model]["processing"]

        # Add annotation with arrow
        ax.annotate(
            f"Best {model}",
            xy=(x_pos, star_y + 0.02),  # Reduced gap between arrow and star
            xytext=(text_x, text_y),  # Different positions for each model
            color=text_color,
            fontsize=ANNOTATION["text_size"],
            ha="left" if i == 0 else "center",  # Center align YOLOv8s text
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad={curve_amount}",
                color=text_color,
                linewidth=1.0,
                shrinkA=0,
                shrinkB=1,
            ),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
            zorder=6,
        )

    # Customize axes
    ax.set_ylabel("Energy (J)")
    ax2.set_ylabel("Power (mW)", color=COLORS["power_line"])
    ax.set_xticks(x)

    # Get layer names from first model
    layer_names = []
    df = model_data[models[0]]["layer_metrics"]
    for split in model_data[models[0]]["energy_analysis"]["Split Layer"]:
        layer_name = df[df["Layer ID"] == split]["Layer Type"].iloc[0]
        layer_names.append(layer_name)

    ax.set_xticklabels(layer_names, rotation=ROTATION, ha="right", va="top", fontsize=7)

    # Clean up plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    add_grid(ax)
    ax2.tick_params(axis="y", colors=COLORS["power_line"])

    # Reorganize legend items to group by component type
    handles, labels = [], []
    handles.extend(ax.get_legend_handles_labels()[0])
    labels.extend(ax.get_legend_handles_labels()[1])
    handles.extend(ax2.get_legend_handles_labels()[0])
    labels.extend(ax2.get_legend_handles_labels()[1])

    # Reorder handles and labels to group by metric type
    ordered_labels = []
    ordered_handles = []

    # Group by metric type with models distinguished by color
    # Mobile Processing
    ordered_labels.extend(
        ["Mobile Processing:", "YOLOv5s", "YOLOv8s"]  # Category label
    )
    ordered_handles.extend(
        [
            plt.Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="none"
            ),  # Empty handle for category
            handles[labels.index(f"{models[0]} Processing")],
            handles[labels.index(f"{models[1]} Processing")],
        ]
    )

    # Data Communication
    ordered_labels.extend(
        ["Data Communication:", "YOLOv5s", "YOLOv8s"]  # Category label
    )
    ordered_handles.extend(
        [
            plt.Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="none"
            ),  # Empty handle for category
            handles[labels.index(f"{models[0]} Communication")],
            handles[labels.index(f"{models[1]} Communication")],
        ]
    )

    # Power
    ordered_labels.extend(["Power (mW):", "YOLOv5s", "YOLOv8s"])  # Category label
    ordered_handles.extend(
        [
            plt.Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="none"
            ),  # Empty handle for category
            handles[labels.index(f"{models[0]} Power")],
            handles[labels.index(f"{models[1]} Power")],
        ]
    )

    # Add legend with reordered items
    legend_params = LEGEND_SPACING.copy()  # Create a copy of the spacing parameters
    legend_params["columnspacing"] = 1.0  # Override columnspacing
    legend_params["handletextpad"] = 0.5  # Add handletextpad

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,  # Three columns for three metric types
        frameon=False,
        fontsize=7,
        **legend_params,
    )

    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()


def plot_comparative_layer_metrics(
    model_data: Dict[str, Dict[str, pd.DataFrame]], output_path: str
) -> None:
    """Create comparative visualization of layer metrics for different models."""
    fig, ax1 = plt.subplots(figsize=DIMENSIONS["layer_metrics"])
    ax2 = ax1.twinx()

    models = list(model_data.keys())
    max_latency = 0
    max_size = 0

    # First, get common layer types across all models
    common_layer_types = set()
    for model in models:
        df = model_data[model]["layer_metrics"]
        split_df = model_data[model]["overall_performance"]
        valid_layer_ids = split_df["Split Layer Index"].unique()
        layer_types = df[df["Layer ID"].isin(valid_layer_ids)]["Layer Type"].unique()
        common_layer_types.update(layer_types)

    common_layer_types = sorted(list(common_layer_types))
    x = np.arange(len(common_layer_types))

    for i, model in enumerate(models):
        df = model_data[model]["layer_metrics"]
        split_df = model_data[model]["overall_performance"]

        # Get valid layer IDs
        valid_layer_ids = split_df["Split Layer Index"].unique()

        # Process layer metrics
        grouped = (
            df[df["Layer ID"].isin(valid_layer_ids)]
            .groupby("Layer Type")  # Group by Layer Type instead of Layer ID
            .agg(
                {
                    "Layer Latency (ms)": "mean",
                    "Output Size (MB)": "mean",
                }
            )
            .reindex(
                common_layer_types, fill_value=0
            )  # Ensure all layer types are included
            .reset_index()
        )

        # Set positions
        offset = (i - 0.5) * BAR_WIDTH

        # Plot metrics
        ax1.bar(
            x + offset,
            grouped["Layer Latency (ms)"],
            BAR_WIDTH,
            label=f"{model} Latency",
            color=COLORS["gpu_energy"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8 if i == 0 else 0.6,
        )

        ax2.bar(
            x + offset,
            grouped["Output Size (MB)"],
            BAR_WIDTH,
            label=f"{model} Size",
            color=COLORS["battery"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8 if i == 0 else 0.6,
        )

        max_latency = max(max_latency, grouped["Layer Latency (ms)"].max())
        max_size = max(max_size, grouped["Output Size (MB)"].max())

    # Customize axes
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Data size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        common_layer_types, rotation=ROTATION, ha="right", va="top", fontsize=7
    )

    # Format ticks
    latency_max = np.ceil(max_latency / 5) * 5
    size_max = np.ceil(max_size / 0.5) * 0.5

    ax1.set_ylim(0, latency_max)
    ax2.set_ylim(0, size_max)

    # Add grid
    add_grid(ax1)

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

    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()
