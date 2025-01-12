#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def process_layer_metrics(excel_path: str) -> pd.DataFrame:
    """Read and process layer metrics from Excel file."""
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name="Layer Metrics")

    # Ensure we have the required columns
    required_cols = ["Layer ID", "Layer Type", "Layer Latency (ms)", "Output Size (MB)"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    # Extract layer number from Layer ID, handling both string and numeric IDs
    def extract_layer_num(layer_id):
        if isinstance(layer_id, str):
            # Try to extract number from string
            import re

            match = re.search(r"\d+", layer_id)
            return int(match.group()) if match else 0
        elif isinstance(layer_id, (int, float)):
            # If it's already a number, just return it
            return int(layer_id)
        else:
            return 0

    # Apply the extraction function
    df["Layer Num"] = df["Layer ID"].apply(extract_layer_num)

    # Group by layer number and calculate means
    grouped = (
        df.groupby("Layer Num")
        .agg(
            {
                "Layer Type": "first",  # Take first layer type
                "Layer Latency (ms)": "mean",
                "Output Size (MB)": "mean",
            }
        )
        .reset_index()
    )

    # Sort by layer number
    grouped = grouped.sort_values("Layer Num")

    return grouped


def plot_layer_metrics(df: pd.DataFrame, output_path: str) -> None:
    """Create a visualization of layer latencies and output sizes."""
    # Create figure and axis with two y-axes - more compact figure
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    # Set bar width and positions - slightly thinner bars
    bar_width = 0.3
    x = np.arange(len(df))

    # Plot latency bars (light blue)
    latency_bars = ax1.bar(
        x - bar_width / 2,
        df["Layer Latency (ms)"],
        bar_width,
        label="Layer latency",
        color="lightblue",
        alpha=0.9,
    )

    # Plot output size bars (dark blue)
    size_bars = ax2.bar(
        x + bar_width / 2,
        df["Output Size (MB)"],
        bar_width,
        label="Size of output data",
        color="navy",
        alpha=0.9,
    )

    # Customize axes
    ax1.set_xlabel("")
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Data size (MB)")

    # Set axis limits and ticks with specific increments
    max_latency = max(df["Layer Latency (ms)"])
    max_size = max(df["Output Size (MB)"])

    # Left y-axis (Latency): increments of 10
    yticks_left = np.arange(0, max_latency * 1.2, 10)
    ax1.set_ylim(0, max_latency * 1.2)
    ax1.set_yticks(yticks_left)

    # Right y-axis (Data size): increments of 0.3
    yticks_right = np.arange(0, max_size * 1.2, 0.3)
    ax2.set_ylim(0, max_size * 1.2)
    ax2.set_yticks(yticks_right)

    # Set x-axis ticks with layer types - properly vertical
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Layer Type"], rotation="vertical", fontsize=6)

    # Adjust spacing for vertical labels
    plt.subplots_adjust(bottom=0.15)

    # Add horizontal dotted lines for better readability
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)

    # Remove vertical grid
    ax1.xaxis.grid(False)

    # Add legends - more compact
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=8,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate layer metrics visualization")
    parser.add_argument(
        "excel_path", help="Path to the Excel file containing layer metrics"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="layer_metrics_plot.png",
        help="Output path for the generated plot (default: layer_metrics_plot.png)",
    )

    args = parser.parse_args()

    try:
        # Process data
        df = process_layer_metrics(args.excel_path)

        # Create visualization
        plot_layer_metrics(df, args.output)
        print(f"Plot saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
