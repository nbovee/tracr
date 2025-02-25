# analysis/constants.py

"""Constants for plotting and analysis."""

# Plot Style Constants
STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "axes.grid": False,
}

# Colors
COLORS = {
    "server": "#4a6fa5",  # Dark blue
    "communication": "#93b7be",  # Medium blue
    "mobile": "#c7dbe6",  # Light blue
    "gpu_energy": "#a1c9f4",  # Light blue for GPU
    "battery": "#2c3e50",  # Dark blue for battery
    "power_line": "#2c3e50",  # Dark blue for power readings
    "data_comm": "#e67e22",  # Dark orange
    "mobile_proc": "#ffd4b2",  # Light orange
}

# Plot Dimensions - Make all plots same height
DIMENSIONS = {
    "layer_metrics": (8, 2.5),  # Changed from 2.0 to 2.5
    "overall_performance": (8, 2.5),  # Already correct
    "energy_analysis": (8, 2.5),  # Already correct
    "raw_metrics": (8, 2.5),  # Updated for consistency
}

# Grid Settings
GRID_STYLE = {
    "alpha": 0.08,
    "color": "gray",
    "linestyle": "-",
}

# Annotation Settings
ANNOTATION = {
    "star_color": "#ffd700",
    "star_size": 10,
    "arrow_style": "->",
    "text_size": 7,
}

# Required Columns for Each Sheet
REQUIRED_COLUMNS = {
    "layer_metrics": [
        "Split Layer",
        "Layer ID",
        "Layer Type",
        "Layer Latency (ms)",
        "Output Size (MB)",
    ],
    "overall_performance": [
        "Split Layer Index",
        "Host Time",
        "Travel Time",
        "Server Time",
        "Total Processing Time",
    ],
    "energy_analysis": [
        "Split Layer",
        "Processing Energy (J)",
        "Communication Energy (J)",
        "Power Reading (W)",
    ],
}

# Plot Settings - Add consistent padding
PLOT_PADDING = {"tight_layout": 0.2, "bbox_inches": "tight", "pad_inches": 0.02}

# Bar Settings
BAR_WIDTH = 0.35
LEGEND_SPACING = {
    "columnspacing": 1.0,
    "handletextpad": 0.3,
}

# Axis Settings
TICK_PADDING = 5
ROTATION = 45

# Energy Analysis Settings
ENERGY_INCREMENT = 0.3
POWER_INCREMENT = 300
