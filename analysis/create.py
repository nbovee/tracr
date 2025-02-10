# analysis/create.py

"""Main script for creating visualizations from metrics data."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict
import pandas as pd

from data_loader import read_excel_data, validate_dataframe
from plots.latency import plot_layer_metrics, plot_overall_performance
from plots.energy import plot_energy_analysis
from constants import REQUIRED_COLUMNS
from plots.comparative import (
    plot_comparative_latency,
    plot_comparative_energy,
)

logger = logging.getLogger(__name__)


def create_plots(excel_path: str, output_dir: str = ".", plot_type: str = "all") -> int:
    """Create visualizations from Excel metrics data.

    Args:
        excel_path: Path to Excel file containing metrics
        output_dir: Output directory for plots
        plot_type: Type of plot to generate ("layer_metrics", "overall_performance",
                  "energy_analysis", or "all")

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Read all data
        data = read_excel_data(excel_path)

        if plot_type in ["layer_metrics", "all"]:
            if data["layer_metrics"] is not None:
                # Validate required columns
                validate_dataframe(
                    data["layer_metrics"],
                    REQUIRED_COLUMNS["layer_metrics"],
                    "Layer Metrics",
                )

                output_path = os.path.join(output_dir, "layer_metrics.png")
                plot_layer_metrics(
                    data["layer_metrics"], data["overall_performance"], output_path
                )
                logger.info(f"Layer metrics plot saved to: {output_path}")
            else:
                logger.warning(
                    "Could not generate layer metrics plot - required sheet not found"
                )

        if plot_type in ["overall_performance", "all"]:
            if data["overall_performance"] is not None:
                # Validate required columns
                validate_dataframe(
                    data["overall_performance"],
                    REQUIRED_COLUMNS["overall_performance"],
                    "Overall Performance",
                )

                output_path = os.path.join(output_dir, "overall_performance.png")
                plot_overall_performance(
                    data["overall_performance"], data["layer_metrics"], output_path
                )
                logger.info(f"Overall performance plot saved to: {output_path}")
            else:
                logger.warning(
                    "Could not generate overall performance plot - required sheet not found"
                )

        if plot_type in ["energy_analysis", "all"]:
            if (
                data["layer_metrics"] is not None
                and data["energy_analysis"] is not None
            ):
                # Validate required columns
                validate_dataframe(
                    data["energy_analysis"],
                    REQUIRED_COLUMNS["energy_analysis"],
                    "Energy Analysis",
                )

                output_path = os.path.join(output_dir, "energy_analysis.png")
                plot_energy_analysis(
                    data["layer_metrics"], data["energy_analysis"], output_path
                )
                logger.info(f"Energy analysis plot saved to: {output_path}")
            else:
                logger.warning(
                    "Could not generate energy analysis plot - required sheets not found"
                )

        return 0

    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        return 1


def create_comparative_plots(
    model_data: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str = ".",
    plot_type: str = "all",
) -> int:
    """Create comparative visualizations for multiple models.

    Args:
        model_data: Dictionary with model names as keys, containing dataframes
                   for each model's metrics
        output_dir: Output directory for plots
        plot_type: Type of plot to generate
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if plot_type in ["overall_performance", "all"]:
            output_path = os.path.join(output_dir, "comparative_latency.png")
            plot_comparative_latency(model_data, output_path)

        if plot_type in ["energy_analysis", "all"]:
            output_path = os.path.join(output_dir, "comparative_energy.png")
            plot_comparative_energy(model_data, output_path)

        return 0

    except Exception as e:
        logger.error(f"Error creating comparative plots: {e}")
        return 1


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from Excel metrics"
    )
    parser.add_argument(
        "excel_paths",
        nargs="+",
        help="Path(s) to Excel file(s) containing metrics. One file for individual plots, multiple files for comparison.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Names of models when comparing multiple models (required for multiple files)",
    )
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

    # Determine if we're doing individual or comparative plots
    if len(args.excel_paths) == 1:
        # Single model case - create individual plots
        return create_plots(args.excel_paths[0], args.output_dir, args.plot_type)
    else:
        # Multiple model case - create comparative plots
        if not args.model_names or len(args.model_names) != len(args.excel_paths):
            logger.error(
                "Must provide model names (--model-names) matching number of Excel files for comparison"
            )
            return 1

        # Load data for all models
        model_data = {}
        for path, name in zip(args.excel_paths, args.model_names):
            model_data[name] = read_excel_data(path)

        return create_comparative_plots(model_data, args.output_dir, args.plot_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())
