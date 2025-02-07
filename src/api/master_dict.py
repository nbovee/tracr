# src/api/master_dict.py

import logging
import pickle  # For serializing the master dictionary to bytes.
import threading  # To provide thread-safety.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, ClassVar

import pandas as pd  # For DataFrame conversion.
from rpyc.utils.classic import obtain  # For deep-copying objects over RPyC.

logger = logging.getLogger("split_computing_logger")


@dataclass
class InferenceMetrics:
    """Container for inference-related metrics.
    These metrics summarize key timings and resource usage for an inference run."""

    split_layer: int
    transmission_latency: int
    inf_time_client: int
    inf_time_edge: int
    total_time: int
    watts_used: float


@dataclass
class LayerData:
    """Container for layer-specific data.
    Each layer in the model can record its own inference time, which node completed it,
    the size of its output, and the energy (watts) used."""

    layer_id: int
    inference_time: Optional[int] = None
    completed_by_node: Optional[str] = None
    output_bytes: Optional[int] = None
    watts_used: float = 0.0


@dataclass
class InferenceData:
    """Container for complete inference data.
    Combines an inference identifier with detailed layer information."""

    inference_id: str
    layer_information: Dict[str, Dict[str, Any]]


class MasterDict:
    """Thread-safe dictionary to store and manage inference data.
    This class supports setting, updating, and retrieving inference results
    and computing various performance and resource metrics."""

    # Default constants for transmission speed, node names, and split layer.
    DEFAULT_BANDWIDTH: ClassVar[float] = 4.0  # MB/s
    DEFAULT_NODES: ClassVar[List[str]] = ["SERVER", "PARTICIPANT"]
    DEFAULT_SPLIT_LAYER: ClassVar[int] = 20

    def __init__(self) -> None:
        """Initialize the master dictionary with thread-safe storage."""
        self._lock = threading.RLock()  # Recursive lock for thread safety.
        # Internal storage for inference data.
        self._data: Dict[str, Dict[str, Any]] = {}

    def _validate_inference_id(self, inference_id: str) -> None:
        """Raise an error if the given inference ID is not found in the store."""
        if inference_id not in self._data:
            raise KeyError(f"Inference ID '{inference_id}' not found.")

    def _get_layer_info(self, inference_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve the layer information for the given inference ID."""
        self._validate_inference_id(inference_id)
        return self._data[inference_id].get("layer_information", {})

    def set_item(self, inference_id: str, value: Dict[str, Any]) -> None:
        """Set or update the value for a given inference ID."""
        with self._lock:
            if inference_id in self._data:
                # If the inference already exists, merge new layer information.
                self._update_existing_inference(inference_id, value)
            else:
                # Otherwise, add the new inference data.
                self._data[inference_id] = value

    def _update_existing_inference(
        self, inference_id: str, value: Dict[str, Any]
    ) -> None:
        """Update an existing inference's data with new layer information.
        Expects that the incoming value contains a 'layer_information' key.
        Only updates layers that have a non-None 'inference_time'."""
        layer_info = value.get("layer_information")
        if not layer_info:
            raise ValueError(
                "Cannot integrate inference_dict without 'layer_information' field."
            )

        # Filter out layers without an inference time.
        filtered_info = {
            k: v for k, v in layer_info.items() if v.get("inference_time") is not None
        }
        # Update the stored layer information with the filtered new data.
        self._data[inference_id]["layer_information"].update(filtered_info)

    def get_item(self, inference_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the data for a given inference ID."""
        with self._lock:
            return self._data.get(inference_id)

    def update_data(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """Update the master dictionary with new inference data.
        If by_value is True, the new data is deep-copied using rpyc's obtain."""
        info_to_update = obtain(new_info) if by_value else new_info
        with self._lock:
            for inference_id, layer_data in info_to_update.items():
                self.set_item(inference_id, layer_data)

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = DEFAULT_BANDWIDTH
    ) -> int:
        """Calculate the transmission latency (in nanoseconds) for the given split layer.
        If the split layer is the default value, returns zero.
        Otherwise, computes latency based on output size and bandwidth."""
        if split_layer == self.DEFAULT_SPLIT_LAYER:
            return 0

        layer_info = self._get_layer_info(inference_id)
        send_layer = split_layer - 1
        # Use a fixed output size if split_layer is 0; otherwise, look up the output bytes.
        output_size = (
            602112
            if split_layer == 0
            else layer_info.get(send_layer, {}).get("output_bytes", 0)
        )

        # Convert the transfer time from seconds to nanoseconds.
        return int((output_size / (mb_per_s * 1e6)) * 1e9)

    def get_total_inference_time(
        self, inference_id: str, nodes: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """Calculate the total inference time (in nanoseconds) for client and edge nodes.
        Returns a tuple: (client inference time, edge inference time).
        For now, both values are computed as the sum of layer times."""
        nodes = nodes or self.DEFAULT_NODES
        layer_info = self._get_layer_info(inference_id)

        def sum_inference_time(layer_data: Dict[str, Any]) -> int:
            # Only add times for layers completed by one of the designated nodes.
            return (
                int(layer_data["inference_time"])
                if (
                    layer_data.get("inference_time")
                    and layer_data.get("completed_by_node") in nodes
                )
                else 0
            )

        times = [sum_inference_time(layer) for layer in layer_info.values()]
        return sum(times), sum(times)

    def get_split_layer(
        self, inference_id: str, nodes: Optional[List[str]] = None
    ) -> int:
        """Determine the split layer where the model transitions between nodes.
        The split is defined as the first layer where the completing node differs from the first layer's node.
        If no change is detected, returns 0 or a default split layer."""
        nodes = nodes or self.DEFAULT_NODES
        layer_info = self._get_layer_info(inference_id)

        # Sort layer IDs numerically (they are stored as strings).
        sorted_layers = sorted(layer_info.keys(), key=int)
        if not sorted_layers:
            raise ValueError(f"No layer information for inference ID '{inference_id}'.")

        # Determine the node that processed the first layer.
        start_node = layer_info[sorted_layers[0]].get("completed_by_node")
        for layer_id in sorted_layers:
            if layer_info[layer_id].get("completed_by_node") != start_node:
                return int(layer_id)

        # If no split is found, return 0 if the start node is among the known nodes; otherwise, use default.
        return 0 if start_node in nodes else self.DEFAULT_SPLIT_LAYER

    def calculate_supermetrics(self, inference_id: str) -> InferenceMetrics:
        """Calculate comprehensive metrics for an inference, including:
          - The split layer,
          - Transmission latency,
          - Total inference times for client and edge,
          - Total energy consumption (watts used).
        """
        split_layer = self.get_split_layer(inference_id)
        transmission_latency = self.get_transmission_latency(inference_id, split_layer)
        inf_time_client, inf_time_edge = self.get_total_inference_time(inference_id)
        total_time = inf_time_client + inf_time_edge + transmission_latency
        watts_used = self.calculate_total_watts_used(inference_id)

        return InferenceMetrics(
            split_layer=split_layer,
            transmission_latency=transmission_latency,
            inf_time_client=inf_time_client,
            inf_time_edge=inf_time_edge,
            total_time=total_time,
            watts_used=watts_used,
        )

    def calculate_total_watts_used(self, inference_id: str) -> float:
        """Calculate the total energy (in watts) used for an inference by summing over all layers."""
        layer_info = self._get_layer_info(inference_id)
        return sum(float(layer.get("watts_used", 0)) for layer in layer_info.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the master dictionary into a pandas DataFrame.
        Each row in the DataFrame corresponds to a single layer's information,
        combined with overall inference metrics."""
        records = []
        layer_attributes = set()

        with self._lock:
            for superfields in self._data.values():
                inference_id = superfields.get("inference_id")
                if not inference_id:
                    continue

                try:
                    # Calculate supermetrics for the inference.
                    metrics = self.calculate_supermetrics(inference_id)
                    # Build individual records for each layer.
                    self._build_dataframe_records(
                        inference_id, metrics, superfields, layer_attributes, records
                    )
                except (KeyError, ValueError):
                    continue

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by=["inference_id", "layer_id"], inplace=True)
        return df

    def _build_dataframe_records(
        self,
        inference_id: str,
        metrics: InferenceMetrics,
        superfields: Dict[str, Any],
        layer_attributes: set,
        records: List[Dict[str, Any]],
    ) -> None:
        """Build records for each layer of an inference to facilitate DataFrame conversion.
        Records include both overall metrics and per-layer details."""
        for layer in superfields.get("layer_information", {}).values():
            layer_id = layer.get("layer_id")
            if layer_id is None:
                continue

            # Collect attribute names (excluding 'layer_id') once.
            if not layer_attributes:
                layer_attributes.update(k for k in layer.keys() if k != "layer_id")

            record = {
                "inference_id": inference_id,
                "split_layer": metrics.split_layer,
                "total_time_ns": metrics.total_time,
                "inf_time_client": metrics.inf_time_client,
                "inf_time_edge": metrics.inf_time_edge,
                "transmission_latency_ns": metrics.transmission_latency,
                "watts_used": metrics.watts_used,
                "layer_id": layer_id,
                **{attr: layer.get(attr) for attr in layer_attributes},
            }
            records.append(record)

    def to_pickle(self) -> bytes:
        """Serialize the master dictionary to a pickle byte stream."""
        with self._lock:
            return pickle.dumps(self._data)

    # Dunder methods for dictionary-like access.
    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        """Allow dictionary-like access for getting items."""
        return self.get_item(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Allow dictionary-like access for setting items."""
        self.set_item(key, value)

    def __iter__(self):
        """Enable iteration over the master dictionary's keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of inference items stored."""
        return len(self._data)
