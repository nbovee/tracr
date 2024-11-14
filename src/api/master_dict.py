# src/api/master_dict.py

import logging
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, ClassVar

import pandas as pd
from rpyc.utils.classic import obtain

logger = logging.getLogger("split_computing_logger")


@dataclass
class InferenceMetrics:
    """Container for inference-related metrics."""

    split_layer: int
    transmission_latency: int
    inf_time_client: int
    inf_time_edge: int
    total_time: int
    watts_used: float


@dataclass
class LayerData:
    """Container for layer-specific data."""

    layer_id: int
    inference_time: Optional[int] = None
    completed_by_node: Optional[str] = None
    output_bytes: Optional[int] = None
    watts_used: float = 0.0


@dataclass
class InferenceData:
    """Container for complete inference data."""

    inference_id: str
    layer_information: Dict[str, Dict[str, Any]]


class MasterDict:
    """Thread-safe dictionary to store and manage inference data."""

    DEFAULT_BANDWIDTH: ClassVar[float] = 4.0  # MB/s
    DEFAULT_NODES: ClassVar[List[str]] = ["SERVER", "PARTICIPANT"]
    DEFAULT_SPLIT_LAYER: ClassVar[int] = 20

    def __init__(self) -> None:
        """Initialize the master dictionary with thread-safe storage."""
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def _validate_inference_id(self, inference_id: str) -> None:
        """Validate inference ID exists in data store."""
        if inference_id not in self._data:
            raise KeyError(f"Inference ID '{inference_id}' not found.")

    def _get_layer_info(self, inference_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve layer information for given inference ID."""
        self._validate_inference_id(inference_id)
        return self._data[inference_id].get("layer_information", {})

    def set_item(self, inference_id: str, value: Dict[str, Any]) -> None:
        """Set or update value for given inference ID."""
        with self._lock:
            if inference_id in self._data:
                self._update_existing_inference(inference_id, value)
            else:
                self._data[inference_id] = value

    def _update_existing_inference(
        self, inference_id: str, value: Dict[str, Any]
    ) -> None:
        """Update existing inference data with new layer information."""
        layer_info = value.get("layer_information")
        if not layer_info:
            raise ValueError(
                "Cannot integrate inference_dict without 'layer_information' field."
            )

        filtered_info = {
            k: v for k, v in layer_info.items() if v.get("inference_time") is not None
        }
        self._data[inference_id]["layer_information"].update(filtered_info)

    def get_item(self, inference_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve value for given inference ID."""
        with self._lock:
            return self._data.get(inference_id)

    def update_data(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """Update master dictionary with new inference data."""
        info_to_update = obtain(new_info) if by_value else new_info
        with self._lock:
            for inference_id, layer_data in info_to_update.items():
                self.set_item(inference_id, layer_data)

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = DEFAULT_BANDWIDTH
    ) -> int:
        """Calculate transmission latency in nanoseconds for split layer."""
        if split_layer == self.DEFAULT_SPLIT_LAYER:
            return 0

        layer_info = self._get_layer_info(inference_id)
        send_layer = split_layer - 1
        output_size = (
            602112
            if split_layer == 0
            else layer_info.get(send_layer, {}).get("output_bytes", 0)
        )

        return int((output_size / (mb_per_s * 1e6)) * 1e9)

    def get_total_inference_time(
        self, inference_id: str, nodes: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """Calculate total inference time for client and edge nodes."""
        nodes = nodes or self.DEFAULT_NODES
        layer_info = self._get_layer_info(inference_id)

        def sum_inference_time(layer_data: Dict[str, Any]) -> int:
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
        """Determine split layer where model transitions nodes."""
        nodes = nodes or self.DEFAULT_NODES
        layer_info = self._get_layer_info(inference_id)

        sorted_layers = sorted(layer_info.keys(), key=int)
        if not sorted_layers:
            raise ValueError(f"No layer information for inference ID '{inference_id}'.")

        start_node = layer_info[sorted_layers[0]].get("completed_by_node")
        for layer_id in sorted_layers:
            if layer_info[layer_id].get("completed_by_node") != start_node:
                return int(layer_id)

        return 0 if start_node in nodes else self.DEFAULT_SPLIT_LAYER

    def calculate_supermetrics(self, inference_id: str) -> InferenceMetrics:
        """Calculate comprehensive metrics for an inference."""
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
        """Calculate total watts used for an inference."""
        layer_info = self._get_layer_info(inference_id)
        return sum(float(layer.get("watts_used", 0)) for layer in layer_info.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert master dictionary to pandas DataFrame."""
        records = []
        layer_attributes = set()

        with self._lock:
            for superfields in self._data.values():
                inference_id = superfields.get("inference_id")
                if not inference_id:
                    continue

                try:
                    metrics = self.calculate_supermetrics(inference_id)
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
        """Build records for DataFrame conversion."""
        for layer in superfields.get("layer_information", {}).values():
            layer_id = layer.get("layer_id")
            if layer_id is None:
                continue

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
        """Serialize master dictionary to pickle byte stream."""
        with self._lock:
            return pickle.dumps(self._data)

    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        """Enable dictionary-like access for getting items."""
        return self.get_item(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Enable dictionary-like access for setting items."""
        self.set_item(key, value)

    def __iter__(self):
        """Enable iteration over master dictionary."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of items in master dictionary."""
        return len(self._data)
