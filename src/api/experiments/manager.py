"""Experiment management utilities"""

import logging
import os
from typing import Any, Dict, Union

import pandas as pd

from ..devices import DeviceManager
from .local import LocalExperiment
from .networked import NetworkedExperiment

logger = logging.getLogger("split_computing_logger")


class ExperimentManager:
    """Factory class for creating and managing experiments.

    This class provides methods to create the appropriate experiment
    type based on configuration and network availability.
    """

    def __init__(self, config: Dict[str, Any], force_local: bool = False):
        """Initialize the experiment manager.

        Args:
            config: Dictionary containing experiment configuration.
            force_local: Whether to force using local experiments even if a server is available.
        """
        self.config = config
        self.device_manager = DeviceManager()
        self.server_device = self.device_manager.get_device_by_type("SERVER")
        self.collect_metrics = config.get("default", {}).get("collect_metrics", False)

        # Log if metrics collection is disabled
        if not self.collect_metrics:
            logger.info("Metrics collection is disabled for this experiment")

        # Decide whether to use networked or local experiment based on server availability and force_local flag.
        self.is_networked = (
            bool(self.server_device and self.server_device.is_reachable())
            and not force_local
        )

        if self.is_networked:
            self.host = self.server_device.get_host()
            self.port = self.server_device.get_port()
            logger.info("Using networked experiment")
        else:
            self.host = None
            self.port = None
            logger.info("Using local experiment")

        # These attributes will be initialized when setup_experiment is called
        self.experiment = None
        self.model = None
        self.results = pd.DataFrame()
        self.output_file = None

    def setup_experiment(self) -> Union[NetworkedExperiment, LocalExperiment]:
        """Create and return an experiment instance based on network availability.

        Returns:
            A NetworkedExperiment or LocalExperiment instance.
        """
        if self.is_networked:
            self.experiment = NetworkedExperiment(self.config, self.host, self.port)
        else:
            self.experiment = LocalExperiment(self.config, self.host, self.port)

        # Store reference to the model and setup data for saving results
        self.model = self.experiment.model if self.experiment else None

        return self.experiment

    def save_results(self, output_file=None, include_columns=None):
        """Save experiment results to an Excel file.

        Args:
            output_file: Path to the output file. If None, uses self.output_file.
            include_columns: List of columns to include. If None, includes all columns.
        """
        if not self.collect_metrics:
            logger.info("Metrics collection is disabled, skipping results saving")
            return

        if output_file is None:
            output_file = self.output_file

        if output_file is None:
            logger.warning("No output file specified, skipping save_results")
            return

        # Create a results directory if it doesn't exist
        results_dir = os.path.dirname(output_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Get layer metrics if available
        layer_metrics_df = None
        if self.model is not None and hasattr(self.model, "get_layer_metrics"):
            layer_metrics = self.model.get_layer_metrics()
            if layer_metrics:
                layer_metrics_df = pd.DataFrame(layer_metrics).T.reset_index()
                layer_metrics_df = layer_metrics_df.rename(
                    columns={"index": "layer_idx"}
                )

                # For Windows CPU, check if we need to fill in missing values
                is_windows_cpu = (
                    hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
                )
                if is_windows_cpu and layer_metrics_df is not None:
                    # Check if we have zero values in crucial fields
                    has_zero_processing_energy = (
                        layer_metrics_df["processing_energy"] == 0
                    ).any()
                    has_zero_power_reading = (
                        layer_metrics_df["power_reading"] == 0
                    ).any()

                    if has_zero_processing_energy or has_zero_power_reading:
                        logger.info(
                            "Found zero values in Windows CPU metrics, attempting to fix..."
                        )
                        try:
                            # Get updated metrics directly from the model
                            updated_metrics = self.model.get_layer_metrics()

                            # Update the DataFrame with non-zero values
                            for idx, row in layer_metrics_df.iterrows():
                                layer_idx = row["layer_idx"]
                                if layer_idx in updated_metrics:
                                    if (
                                        row["processing_energy"] == 0
                                        and updated_metrics[layer_idx][
                                            "processing_energy"
                                        ]
                                        > 0
                                    ):
                                        layer_metrics_df.at[
                                            idx, "processing_energy"
                                        ] = updated_metrics[layer_idx][
                                            "processing_energy"
                                        ]
                                        logger.info(
                                            f"Updated processing_energy for layer {layer_idx}"
                                        )

                                    if (
                                        row["power_reading"] == 0
                                        and updated_metrics[layer_idx]["power_reading"]
                                        > 0
                                    ):
                                        layer_metrics_df.at[idx, "power_reading"] = (
                                            updated_metrics[layer_idx]["power_reading"]
                                        )
                                        logger.info(
                                            f"Updated power_reading for layer {layer_idx}"
                                        )

                                    # Update total energy
                                    layer_metrics_df.at[idx, "total_energy"] = (
                                        layer_metrics_df.at[idx, "processing_energy"]
                                        + layer_metrics_df.at[
                                            idx, "communication_energy"
                                        ]
                                    )

                                    # Update memory utilization if available
                                    if (
                                        "memory_utilization"
                                        in updated_metrics[layer_idx]
                                    ):
                                        layer_metrics_df.at[
                                            idx, "memory_utilization"
                                        ] = updated_metrics[layer_idx][
                                            "memory_utilization"
                                        ]

                            logger.info(
                                "Successfully updated Windows CPU metrics in DataFrame"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update Windows CPU metrics: {e}")

        # Check if output file exists
        if os.path.exists(output_file):
            logger.info(f"Appending to existing file: {output_file}")
            with pd.ExcelWriter(
                output_file, mode="a", engine="openpyxl", if_sheet_exists="replace"
            ) as writer:
                self._write_excel_data(writer, include_columns, layer_metrics_df)
        else:
            logger.info(f"Creating new file: {output_file}")
            with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
                self._write_excel_data(writer, include_columns, layer_metrics_df)

        logger.info(f"Results saved to {output_file}")

    def _write_excel_data(self, writer, include_columns, layer_metrics_df):
        """Write data to Excel sheets.

        Args:
            writer: pandas ExcelWriter object.
            include_columns: List of columns to include.
            layer_metrics_df: DataFrame containing layer metrics.
        """
        # Save the results dataframe
        if not self.results.empty:
            # Filter columns if specified
            if include_columns:
                df_filtered = self.results[
                    [c for c in include_columns if c in self.results.columns]
                ]
            else:
                df_filtered = self.results

            df_filtered.to_excel(writer, sheet_name="Results", index=False)

        # Save layer metrics if available
        if layer_metrics_df is not None and not layer_metrics_df.empty:
            # Add explicit GPU utilization logging before writing to Excel
            for idx, row in layer_metrics_df.iterrows():
                gpu_util = row.get("GPU Utilization (%)", 0.0)
                logger.debug(
                    f"Excel row {idx}: Layer {row.get('Layer ID', -1)} GPU utilization = {gpu_util}%"
                )

            layer_metrics_df.to_excel(writer, sheet_name="Layer Metrics", index=False)

            # Save energy summary
            energy_summary = self._create_energy_summary(layer_metrics_df)

            # For Windows CPU, update the energy summary with non-zero metrics
            is_windows_cpu = (
                hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
            )

            if is_windows_cpu and not energy_summary.empty:
                self._update_windows_cpu_metrics(energy_summary, layer_metrics_df)

            if not energy_summary.empty:
                energy_summary.to_excel(
                    writer, sheet_name="Energy Analysis", index=False
                )

    def _create_energy_summary(self, layer_metrics_df=None) -> pd.DataFrame:
        """Create a summary of energy metrics.

        Args:
            layer_metrics_df: DataFrame containing layer metrics.

        Returns:
            DataFrame with energy summary by split layer.
        """
        if layer_metrics_df is None or layer_metrics_df.empty:
            return pd.DataFrame()

        # Define aggregation dictionary
        energy_agg_dict = {
            "Processing Energy (J)": "sum",
            "Communication Energy (J)": "sum",
            "Total Energy (J)": "sum",
            "Power Reading (W)": "mean",
            "GPU Utilization (%)": "mean",
            "Host Battery Energy (mWh)": "first",
        }

        # Add Memory Utilization to aggregation if available
        if "Memory Utilization (%)" in layer_metrics_df.columns:
            energy_agg_dict["Memory Utilization (%)"] = "mean"

        # Group by Split Layer and create summary
        energy_summary = (
            layer_metrics_df.groupby("Split Layer").agg(energy_agg_dict).reset_index()
        )

        return energy_summary

    def _update_windows_cpu_metrics(self, energy_summary, layer_metrics_df):
        """Update energy metrics for Windows CPU measurements.

        Args:
            energy_summary: DataFrame with energy summary.
            layer_metrics_df: DataFrame with layer metrics.
        """
        logger.info("Checking Windows CPU energy summary for zero values...")

        try:
            for idx, row in energy_summary.iterrows():
                split_layer = row.get("Split Layer", -1)

                # Skip if split layer is not valid
                if split_layer < 0:
                    continue

                # Get all metrics for layers up to this split point
                split_df = layer_metrics_df[
                    layer_metrics_df["layer_idx"] <= split_layer
                ]

                # Only update if we have valid metrics
                if not split_df.empty:
                    # Fix power reading - use max value
                    valid_power = split_df["power_reading"].max()
                    if valid_power > 0 and row.get("Power Reading (W)", 0) == 0:
                        energy_summary.at[idx, "Power Reading (W)"] = valid_power
                        logger.info(
                            f"Updated power reading for split {split_layer} to {valid_power:.2f}W"
                        )

                    # Fix processing energy - use sum
                    valid_energy = split_df["processing_energy"].sum()
                    if valid_energy > 0 and row.get("Processing Energy (J)", 0) == 0:
                        energy_summary.at[idx, "Processing Energy (J)"] = valid_energy
                        logger.info(
                            f"Updated processing energy for split {split_layer} to {valid_energy:.6f}J"
                        )

                    # Update total energy
                    comm_energy = row.get("Communication Energy (J)", 0)
                    energy_summary.at[idx, "Total Energy (J)"] = (
                        valid_energy + comm_energy
                    )

                    # Update memory utilization if available
                    if "memory_utilization" in split_df.columns:
                        valid_mem = split_df["memory_utilization"].max()
                        if valid_mem > 0:
                            energy_summary.at[idx, "Memory Utilization (%)"] = valid_mem

            logger.info("Successfully updated Windows CPU energy summary")
        except Exception as e:
            logger.warning(f"Failed to update energy summary: {e}")
