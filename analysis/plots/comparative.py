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
    # Increase both width and height
    dimensions = list(DIMENSIONS["energy_analysis"])
    dimensions[0] += 2  # Add 2 inches to width
    dimensions[1] += 1  # Add 1 inch to height
    fig, ax = plt.subplots(figsize=tuple(dimensions))

    ax2 = ax.twinx()  # For power readings

    # Create third y-axis for battery energy with more spacing
    ax3 = ax.twinx()
    # Offset the third axis closer to the power axis
    ax3.spines["right"].set_position(("axes", 1.15))  # Changed from 1.25 to 1.15

    # Plot bars for each model side by side
    models = list(model_data.keys())
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
    max_battery = 0

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

        # Plot stacked bars with adjusted alpha
        bars1 = ax.bar(
            x + offset,
            comm_energy,
            BAR_WIDTH,
            color=model_colors[model]["communication"],
            edgecolor="black",
            linewidth=0.5,
            label=f"{model} Communication",
            alpha=0.8,  # Slightly reduced alpha for better visibility
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
            alpha=0.8,  # Slightly reduced alpha
        )

        # Plot power line with adjusted style
        line = ax2.plot(
            x + offset,
            power_readings,
            color=model_colors[model]["power"],
            linestyle="-" if i == 0 else "--",  # Solid vs dashed for distinction
            linewidth=1.2,  # Slightly thinner for clarity
            label=f"{model} Power",
            marker="D",  # Diamond marker (more professional than circle)
            markersize=4,  # Slightly larger marker
            markerfacecolor="white",
            markeredgecolor=model_colors[model]["power"],
            markeredgewidth=1.0,
            alpha=0.9,
        )

        # Track best points and maximum values
        total_energy = [m + c for m, c in zip(mobile_energy, comm_energy)]
        max_energy = max(max_energy, max(total_energy))
        max_power = max(max_power, max(power_readings))
        best_idx = np.argmin(total_energy)
        best_energy = total_energy[best_idx]
        best_points[model] = (best_idx + offset, best_energy)

        # Add battery energy line
        battery_energy = energy_df["Host Battery Energy (mWh)"].values
        max_battery = max(max_battery, max(battery_energy))

        # Plot battery energy line
        line_battery = ax3.plot(
            x + offset,
            battery_energy,
            color=model_colors[model]["power"],
            linestyle=(
                "-." if i == 0 else ":"
            ),  # Dash-dot vs dotted (standard in publications)
            linewidth=1.2,
            label=f"{model} Battery",
            marker="^",  # Triangle marker (distinct and professional)
            markersize=5,  # Slightly larger marker
            markerfacecolor="white",
            markeredgecolor=model_colors[model]["power"],
            markeredgewidth=1.0,
            alpha=0.8,
        )

    # Add best point annotations with improved positioning
    for i, (model, (x_pos, y_pos)) in enumerate(best_points.items()):
        # YOLOv8s (i=1) vertical, YOLOv5s (i=0) curved right
        if i == 0:  # YOLOv5s
            text_x = x_pos + 2.0  # Move text further right (was 1.0)
            text_y = y_pos + 0.25  # Higher text position (was 0.15)
            curve_amount = 0.3  # More curve (was 0.15)
        else:  # YOLOv8s
            text_x = x_pos  # Keep text directly above
            text_y = y_pos + 0.3  # Higher text position
            curve_amount = 0.0  # Perfectly vertical

        # Add star closer to bar
        star_y = y_pos + 0.03  # Keep same gap between bar and star
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
            xy=(x_pos, star_y + 0.02),  # Keep same gap between arrow and star
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

    # Customize axes with better spacing
    ax.set_ylabel("Energy (J)", labelpad=15)
    ax2.set_ylabel("Power (mW)", color=COLORS["power_line"], labelpad=15)
    ax3.set_ylabel("Battery Energy (mWh)", color=COLORS["power_line"], labelpad=20)

    # Adjust tick parameters for better readability
    ax.tick_params(axis="both", labelsize=8, pad=5)
    ax2.tick_params(axis="y", colors=COLORS["power_line"], labelsize=8, pad=5)
    ax3.tick_params(axis="y", colors=COLORS["power_line"], labelsize=8, pad=5)

    # Add grid with light alpha
    ax.grid(True, axis="y", alpha=0.1, linestyle="-")

    # Adjust legend
    legend_params = LEGEND_SPACING.copy()
    legend_params["columnspacing"] = 2.0  # Increased spacing between columns
    legend_params["handletextpad"] = 0.7  # More space between handle and text

    handles1, labels1 = ax.get_legend_handles_labels()  # Energy bars
    handles2, labels2 = ax2.get_legend_handles_labels()  # Power lines
    handles3, labels3 = ax3.get_legend_handles_labels()  # Battery lines

    handles = handles1 + handles2 + handles3
    labels = labels1 + labels2 + labels3

    # Reorder handles and labels to group by metric type
    ordered_labels = []
    ordered_handles = []

    # Group by metric type with models distinguished by color
    # Mobile Processing
    ordered_labels.extend(["Mobile Processing:", "YOLOv5s", "YOLOv8s"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Processing")],
            handles[labels.index(f"{models[1]} Processing")],
        ]
    )

    # Data Communication
    ordered_labels.extend(["Data Communication:", "YOLOv5s", "YOLOv8s"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Communication")],
            handles[labels.index(f"{models[1]} Communication")],
        ]
    )

    # Power
    ordered_labels.extend(["Power (mW):", "YOLOv5s", "YOLOv8s"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Power")],
            handles[labels.index(f"{models[1]} Power")],
        ]
    )

    # Battery
    ordered_labels.extend(["Battery (mWh):", "YOLOv5s", "YOLOv8s"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Battery")],
            handles[labels.index(f"{models[1]} Battery")],
        ]
    )

    # Add legend with reordered items
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),  # Move legend even higher
        ncol=4,
        frameon=False,
        fontsize=8,  # Slightly larger font
        **legend_params,
    )

    # Adjust layout with more space
    plt.tight_layout(pad=PLOT_PADDING["tight_layout"])
    plt.subplots_adjust(
        right=0.82,  # Changed from 0.85 to 0.82 to reduce right margin
        top=0.80,  # More space for legend
        bottom=0.15,  # Keep space for x-axis labels
        left=0.10,  # Add some space on the left
    )

    # Set battery energy axis limits with nice round numbers
    battery_limit = np.ceil(max_battery / 100) * 100
    ax3.set_ylim(0, battery_limit)
    ax3.yaxis.set_major_locator(plt.MultipleLocator(200))  # Tick every 200 mWh

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()
