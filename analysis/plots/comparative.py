# analysis/plots/comparative.py

"""Comparative plotting functions for multiple models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

from .base import add_grid
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

    # Add spacing between groups by stretching x-axis
    group_spacing = 1.0  # Increased from 0.5 to 1.0 for more separation between groups
    x = x * (1 + group_spacing)  # Stretch x positions to add gaps between groups

    # Define professional color schemes for each model with CPU/GPU grouping
    model_colors = {
        "YOLOv5s CPU": {  # CPU Group - Blues
            "server": "#1f77b4",  # Strong blue
            "communication": "#6baed6",  # Medium blue
            "mobile": "#bdd7e7",  # Light blue
        },
        "YOLOv8s CPU": {  # CPU Group - Blue-purples
            "server": "#4a4090",  # Strong blue-purple
            "communication": "#807dba",  # Medium blue-purple
            "mobile": "#bcbddc",  # Light blue-purple
        },
        "YOLOv5s GPU": {  # GPU Group - Oranges
            "server": "#d95f02",  # Strong orange
            "communication": "#fc8d62",  # Medium orange
            "mobile": "#fdd0a2",  # Light orange
        },
        "YOLOv8s GPU": {  # GPU Group - Red-oranges
            "server": "#e6550d",  # Strong red-orange
            "communication": "#fdae6b",  # Medium red-orange
            "mobile": "#fee6ce",  # Light red-orange
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

        # Keep bars in group close together but add spacing between groups
        offset = (i - 1.5) * BAR_WIDTH
        bottom = np.zeros(len(df))

        for metric, color, label in zip(metrics, colors, labels):
            ax.bar(
                x + offset,  # x is now stretched to create group spacing
                df[metric],
                BAR_WIDTH,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=label,
                alpha=1.0,
            )
            bottom += df[metric]

        # Update best point tracking with new x positions
        total_latencies = df["Total Processing Time"]
        max_total = max(max_total, total_latencies.max())
        best_idx = total_latencies.idxmin()
        best_latency = total_latencies[best_idx]
        best_points[model] = (x[best_idx] + offset, best_latency)

    # Add best point annotations with improved positioning
    for i, (model, (x_pos, y_pos)) in enumerate(best_points.items()):
        # Customize text position and arrow length for each model to prevent overlap
        if model == "YOLOv8s CPU":  # Changed order - YOLOv8 on left
            text_x = x_pos - 1.0  # Move left
            text_y = y_pos + 6.0  # Keep CPU annotations lower
            curve_direction = -1
        elif model == "YOLOv8s GPU":  # YOLOv8 GPU also on left
            text_x = x_pos - 0.8  # Move slightly left
            text_y = y_pos + 12.0  # Move GPU annotation much higher
            curve_direction = -1
        elif model == "YOLOv5s CPU":  # YOLOv5 on right
            text_x = x_pos + 0.8  # Move slightly right
            text_y = y_pos + 7.0  # Keep CPU annotations lower
            curve_direction = 1
        else:  # YOLOv5s GPU - on right
            text_x = x_pos + 1.0  # Move right
            text_y = y_pos + 13.0  # Move GPU annotation highest
            curve_direction = 1

        # Add star with matching color as the arrow
        star_y = y_pos + 2.0
        ax.plot(
            x_pos,
            star_y,
            marker="*",
            markersize=ANNOTATION["star_size"],
            color=model_colors[model]["server"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )

        # Add text with matching color
        text_color = model_colors[model]["server"]

        # Add annotation with arrow
        ax.annotate(
            f"Best {model}",
            xy=(x_pos, star_y + 0.8),
            xytext=(text_x, text_y),
            color=text_color,
            fontsize=ANNOTATION["text_size"],
            ha="right" if curve_direction < 0 else "left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad={0.05 * curve_direction}",
                color=text_color,
                linewidth=1.0,
                shrinkA=0,
                shrinkB=2,
            ),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
            zorder=6,
        )

    # Customize axes
    ax.set_ylabel("Time (s)")
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

    # Reorganize legend items to group by CPU/GPU and components
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = []
    ordered_handles = []

    # Add CPU components
    ordered_labels.extend(["CPU Server", "CPU Communication", "CPU Host"])
    ordered_handles.extend(
        [
            handles[
                labels.index("YOLOv5s CPU Server")
            ],  # Use YOLOv5s CPU as representative
            handles[labels.index("YOLOv5s CPU Travel")],
            handles[labels.index("YOLOv5s CPU Host")],
        ]
    )

    # Add GPU components
    ordered_labels.extend(["GPU Server", "GPU Communication", "GPU Host"])
    ordered_handles.extend(
        [
            handles[
                labels.index("YOLOv5s GPU Server")
            ],  # Use YOLOv5s GPU as representative
            handles[labels.index("YOLOv5s GPU Travel")],
            handles[labels.index("YOLOv5s GPU Host")],
        ]
    )

    # Add legend with reordered items
    legend_params = LEGEND_SPACING.copy()
    legend_params["columnspacing"] = 1.0
    legend_params["handletextpad"] = 0.5

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,  # Changed to 3 columns for better layout
        frameon=False,
        fontsize=8,  # Slightly larger font for better readability
        **legend_params,
    )

    # Add dotted bounding boxes around the last layer (Detect) to group YOLOv5 and YOLOv8
    last_x = x[-1]  # X position of the last layer (Detect)

    # YOLOv5 box (around the two rightmost bars in group)
    v5_left = last_x + 0.5 * BAR_WIDTH  # Start after YOLOv8 bars
    v5_right = last_x + 2.5 * BAR_WIDTH
    v5_bottom = 0
    v5_height = max_total  # Use full height
    v5_rect = plt.Rectangle(
        (v5_left, v5_bottom),
        v5_right - v5_left,
        v5_height,
        fill=False,
        linestyle=":",
        edgecolor=model_colors["YOLOv5s CPU"]["server"],  # Use YOLOv5 blue color
        linewidth=1,
        zorder=1,
    )
    ax.add_patch(v5_rect)

    # YOLOv8 box (around the two leftmost bars in group)
    v8_left = last_x - 2.5 * BAR_WIDTH  # Start before first bar
    v8_right = last_x - 0.5 * BAR_WIDTH
    v8_rect = plt.Rectangle(
        (v8_left, v5_bottom),
        v8_right - v8_left,
        v5_height,
        fill=False,
        linestyle=":",
        edgecolor=model_colors["YOLOv8s CPU"]["server"],  # Use YOLOv8 purple color
        linewidth=1,
        zorder=1,
    )
    ax.add_patch(v8_rect)

    # Add small labels above the boxes with more spacing
    # YOLOv8 text - moved further left
    ax.text(
        (v8_left + v8_right) / 2 - BAR_WIDTH,  # Move left by one bar width
        v5_height * 1.05,
        "v8",
        ha="center",
        va="bottom",
        fontsize=8,  # Slightly larger for bold text
        color=model_colors["YOLOv8s CPU"]["server"],
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
        weight="bold",  # Make text bold
    )

    # Add "v" text in center
    ax.text(
        last_x,  # Center at the x position of the last layer
        v5_height * 1.05,
        "v",
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
    )

    # YOLOv5 text - moved further right
    ax.text(
        (v5_left + v5_right) / 2 + BAR_WIDTH,  # Move right by one bar width
        v5_height * 1.05,
        "v5",
        ha="center",
        va="bottom",
        fontsize=8,  # Slightly larger for bold text
        color=model_colors["YOLOv5s CPU"]["server"],
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
        weight="bold",  # Make text bold
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
        "YOLOv5s CPU": {  # CPU Group - Blues
            "processing": "#e63946",  # Deep red/coral
            "communication": "#f4a261",  # Light orange
            "power": "#9d0208",  # Dark red for power line
        },
        "YOLOv8s CPU": {  # CPU Group - Blue-purples
            "processing": "#fb8b24",  # Bright orange
            "communication": "#ffd6a5",  # Light peach
            "power": "#bb3e03",  # Dark orange for power line
        },
        "YOLOv5s GPU": {  # GPU Group - Oranges
            "processing": "#ff7f0e",  # Strong orange
            "communication": "#ffa852",  # Medium orange
            "power": "#ffd0a8",  # Light orange
        },
        "YOLOv8s GPU": {  # GPU Group - Red-oranges
            "processing": "#9467bd",  # Strong purple
            "communication": "#b39bc8",  # Medium purple
            "power": "#d1c9e6",  # Light purple
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
        # Customize text position and arrow length for each model to prevent overlap
        if model == "YOLOv8s CPU":  # Changed order - YOLOv8 on left
            text_x = x_pos - 1.0  # Move left
            text_y = y_pos + 6.0  # Keep CPU annotations lower
            curve_direction = -1
        elif model == "YOLOv8s GPU":  # YOLOv8 GPU also on left
            text_x = x_pos - 0.8  # Move slightly left
            text_y = y_pos + 12.0  # Move GPU annotation much higher
            curve_direction = -1
        elif model == "YOLOv5s CPU":  # YOLOv5 on right
            text_x = x_pos + 0.8  # Move slightly right
            text_y = y_pos + 7.0  # Keep CPU annotations lower
            curve_direction = 1
        else:  # YOLOv5s GPU - on right
            text_x = x_pos + 1.0  # Move right
            text_y = y_pos + 13.0  # Move GPU annotation highest
            curve_direction = 1

        # Add star with matching color as the arrow
        star_y = y_pos + 2.0
        ax.plot(
            x_pos,
            star_y,
            marker="*",
            markersize=ANNOTATION["star_size"],
            color=model_colors[model]["server"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )

        # Add text with matching color
        text_color = model_colors[model]["server"]

        # Add annotation with arrow
        ax.annotate(
            f"Best {model}",
            xy=(x_pos, star_y + 0.8),
            xytext=(text_x, text_y),
            color=text_color,
            fontsize=ANNOTATION["text_size"],
            ha="right" if curve_direction < 0 else "left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad={0.05 * curve_direction}",
                color=text_color,
                linewidth=1.0,
                shrinkA=0,
                shrinkB=2,
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
    ordered_labels.extend(["Mobile Processing:", "YOLOv5s CPU", "YOLOv8s CPU"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Processing")],
            handles[labels.index(f"{models[1]} Processing")],
        ]
    )

    # Data Communication
    ordered_labels.extend(["Data Communication:", "YOLOv5s CPU", "YOLOv8s CPU"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Communication")],
            handles[labels.index(f"{models[1]} Communication")],
        ]
    )

    # Power
    ordered_labels.extend(["Power (mW):", "YOLOv5s CPU", "YOLOv8s CPU"])
    ordered_handles.extend(
        [
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none"),
            handles[labels.index(f"{models[0]} Power")],
            handles[labels.index(f"{models[1]} Power")],
        ]
    )

    # Battery
    ordered_labels.extend(["Battery (mWh):", "YOLOv5s CPU", "YOLOv8s CPU"])
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

    # Add x-axis labels using layer names (same as comparative_latency)
    ax.set_xticks(x)

    # Get layer names from first model (assuming same layers)
    layer_names = []
    df = model_data[models[0]]["layer_metrics"]
    for idx in model_data[models[0]]["energy_analysis"]["Split Layer"]:
        layer_name = df[df["Layer ID"] == idx]["Layer Type"].iloc[0]
        layer_names.append(layer_name)

    ax.set_xticklabels(layer_names, rotation=ROTATION, ha="right", va="top", fontsize=7)
    ax.tick_params(axis="x", pad=TICK_PADDING)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches=PLOT_PADDING["bbox_inches"],
        pad_inches=PLOT_PADDING["pad_inches"],
    )
    plt.close()
