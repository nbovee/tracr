#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_layer_metrics(excel_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and process layer metrics from Excel file."""
    # Read the Excel file - both sheets
    layer_df = pd.read_excel(excel_path, sheet_name="Layer Metrics")
    split_df = pd.read_excel(excel_path)  # Default sheet with split times
    
    # Ensure we have the required columns
    required_layer_cols = ["Layer ID", "Layer Type", "Layer Latency (ms)", "Output Size (MB)"]
    if not all(col in layer_df.columns for col in required_layer_cols):
        raise ValueError(f"Excel file must contain columns: {required_layer_cols}")
        
    required_split_cols = ["Split Layer Index", "Host Time", "Travel Time", "Server Time", "Total Processing Time"]
    if not all(col in split_df.columns for col in required_split_cols):
        raise ValueError(f"Excel file must contain columns: {required_split_cols}")

    # Extract layer number from Layer ID, handling both string and numeric IDs
    def extract_layer_num(layer_id):
        if isinstance(layer_id, str):
            import re
            match = re.search(r"\d+", layer_id)
            return int(match.group()) if match else 0
        elif isinstance(layer_id, (int, float)):
            return int(layer_id)
        else:
            return 0

    # Process layer metrics
    layer_df["Layer Num"] = layer_df["Layer ID"].apply(extract_layer_num)
    grouped = layer_df.groupby("Layer Num").agg({
        "Layer Type": "first",
        "Layer Latency (ms)": "mean",
        "Output Size (MB)": "mean"
    }).reset_index()
    grouped = grouped.sort_values("Layer Num")
    
    return grouped, split_df

def plot_layer_metrics(layer_df: pd.DataFrame, split_df: pd.DataFrame, output_path: str) -> None:
    """Create a publication-quality visualization of layer metrics and split analysis."""
    # Set professional plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 7,           # Reduced base font size
        'axes.labelsize': 7,      # Smaller axis labels
        'axes.titlesize': 8,      # Smaller title
        'xtick.labelsize': 6,     # Smaller tick labels
        'ytick.labelsize': 6,     # Smaller tick labels
        'legend.fontsize': 6,     # Smaller legend
        'figure.dpi': 300,
    })

    # Create figure - more compact size for JMLR
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()  # Create twin axis for output size
    ax3 = ax1.twinx()  # Create another twin axis for total time
    
    # Offset the right spine for ax3
    ax3.spines['right'].set_position(('outward', 50))  # Slightly reduced offset

    # Set bar width and positions
    bar_width = 0.25  # Even thinner bars for better proportion
    x = np.arange(len(layer_df))

    # Color scheme (colorblind-friendly)
    color_latency = "#a1c9f4"     # Light blue
    color_size = "#2c3e50"        # Dark blue
    color_time = "#8b0000"        # Dark red for total time

    # Plot layer metrics
    latency_bars = ax1.bar(x - bar_width/2, layer_df["Layer Latency (ms)"], 
                          bar_width, label="Layer latency", 
                          color=color_latency, edgecolor='black', linewidth=0.5)
    
    size_bars = ax2.bar(x + bar_width/2, layer_df["Output Size (MB)"], 
                       bar_width, label="Output size", 
                       color=color_size, edgecolor='black', linewidth=0.5)

    # Plot total processing time curve
    times = split_df.set_index("Split Layer Index")
    total_time_line = ax3.plot(times.index, times["Total Processing Time"], 
                              color=color_time, linestyle='-', linewidth=1,
                              label='Total processing time')

    # Find optimal split
    optimal_idx = times["Total Processing Time"].idxmin()
    optimal_time = times["Total Processing Time"].min()
    
    # Add vertical line for optimal split
    ax1.axvline(x=optimal_idx, color=color_time, linestyle='--', 
                linewidth=0.8, alpha=0.4)

    # Add grid
    ax1.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax1.set_axisbelow(True)  # Ensure grid is behind the bars
    
    # Customize axes
    ax1.set_ylabel("Layer latency (ms)")
    ax2.set_ylabel("Output size (MB)")
    ax3.set_ylabel("Total processing time (s)", color=color_time)
    ax3.tick_params(axis='y', labelcolor=color_time)
    
    # Set axis limits and ticks with specific increments
    max_latency = max(layer_df["Layer Latency (ms)"])
    max_size = max(layer_df["Output Size (MB)"])
    max_total_time = max(times["Total Processing Time"])
    
    # Left y-axis (Latency): increments of 10
    max_latency_rounded = np.ceil(max_latency / 10) * 10
    latency_ticks = np.arange(0, max_latency_rounded + 10, 10)
    ax1.set_ylim(0, max_latency_rounded)
    ax1.set_yticks(latency_ticks)
    
    # Middle y-axis (Output size): increments of 0.3
    max_size_rounded = np.ceil(max_size / 0.3) * 0.3
    size_ticks = np.arange(0, max_size_rounded + 0.3, 0.3)
    ax2.set_ylim(0, max_size_rounded)
    ax2.set_yticks(size_ticks)
    
    # Right y-axis (Total time): adaptive increments based on max time
    if max_total_time < 50:  # YOLOv8 case
        increment = 5
    else:  # ResNet case
        increment = 30
    
    max_time_rounded = np.ceil(max_total_time / increment) * increment
    time_ticks = np.arange(0, max_time_rounded + increment, increment)
    ax3.set_ylim(0, max_time_rounded)
    ax3.set_yticks(time_ticks)

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_df["Layer Type"], rotation=90, ha='center', va='top')

    # Add subtle annotation for optimal split
    min_time_text = f"(min: {optimal_time:.2f}s)"
    ax3.annotate(min_time_text, xy=(optimal_idx, optimal_time),
                xytext=(5, 5), textcoords='offset points',
                fontsize=6, color=color_time, alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Add legend with smaller font and tighter spacing
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
              loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='none', ncol=3, columnspacing=1,
              handletextpad=0.5, borderaxespad=0.5)  # Reduced padding

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate layer metrics visualization")
    parser.add_argument("excel_path", help="Path to the Excel file containing metrics")
    parser.add_argument("--output", "-o", default="layer_metrics_plot.png",
                       help="Output path for the generated plot (default: layer_metrics_plot.png)")
    
    args = parser.parse_args()
    
    try:
        # Process data
        layer_df, split_df = process_layer_metrics(args.excel_path)
        
        # Create visualization
        plot_layer_metrics(layer_df, split_df, args.output)
        print(f"Plot saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
