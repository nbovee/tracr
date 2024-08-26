import threading
import pickle
from typing import Dict, Tuple, Any, Optional
import pandas as pd
from rpyc.utils.classic import obtain

BYTES_PER_MB = 1e6
MAX_SPLIT_LAYER = 20
DEFAULT_OUTPUT_BYTES = 602112


class MasterDict:
    """A thread-safe dictionary for managing inference data in a distributed environment."""

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set a value in the dictionary, ensuring thread-safety."""
        with self._lock:
            if key in self._data:
                if value.get("layer_information"):
                    layer_info = value["layer_information"]
                    layer_info = {
                        k: v
                        for k, v in layer_info.items()
                        if v["inference_time"] is not None
                    }
                    self._data[key]["layer_information"].update(layer_info)
                else:
                    raise ValueError(
                        f"Cannot integrate inference_dict without 'layer_information' field"
                    )
                return
            self._data[key] = value

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the dictionary."""
        with self._lock:
            return self._data.get(key)

    def update(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """Update the dictionary with new information."""
        if by_value:
            try:
                new_info = obtain(new_info)
            except Exception as e:
                print(f"Error obtaining new_info: {e}")
                new_info = pickle.loads(
                    pickle.dumps(new_info))  # Fallback method
        with self._lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = 4.0
    ) -> int:
        """Calculate the transmission latency for a given inference."""
        with self._lock:
            try:
                inf_data = self._data[inference_id]
                if split_layer == MAX_SPLIT_LAYER:
                    return 0
                send_layer = split_layer - 1
                sent_output_size_bytes = (
                    DEFAULT_OUTPUT_BYTES
                    if split_layer == 0
                    else inf_data["layer_information"][send_layer]["output_bytes"]
                )
                bytes_per_second = mb_per_s * BYTES_PER_MB
                latency_s = sent_output_size_bytes / bytes_per_second
                return int(latency_s * 1e9)
            except KeyError as e:
                raise KeyError(
                    f"Inference ID {inference_id} not found in data") from e

    def get_total_inference_time(self, inference_id: str) -> Tuple[int, int]:
        """Get the total inference time for initiator and receiver."""
        with self._lock:
            inf_data = self._data[inference_id]
            initiator_node = inf_data["layer_information"]["0"]["completed_by_node"]
            initiator_time = sum(
                int(layer["inference_time"])
                for layer in inf_data["layer_information"].values()
                if layer["inference_time"] and layer["completed_by_node"] == initiator_node
            )
            receiver_time = sum(
                int(layer["inference_time"])
                for layer in inf_data["layer_information"].values()
                if layer["inference_time"] and layer["completed_by_node"] != initiator_node
            )
            return initiator_time, receiver_time

    def get_split_layer(self, inference_id: str) -> int:
        """Get the split layer for a given inference."""
        with self._lock:
            inf_data = self._data[inference_id]
            layer_ids = sorted(list(inf_data["layer_information"].keys()))
            start_node = inf_data["layer_information"]["0"]["completed_by_node"]
            for layer_id in layer_ids:
                if inf_data["layer_information"][layer_id]["completed_by_node"] != start_node:
                    return int(layer_id)
            return 0 if start_node == inf_data["layer_information"]["0"]["completed_by_node"] else MAX_SPLIT_LAYER

    def calculate_supermetrics(
        self, inference_id: str
    ) -> Tuple[int, int, int, int, int]:
        """Calculate supermetrics for a given inference."""
        with self._lock:
            split_layer = self.get_split_layer(inference_id)
            transmission_latency = self.get_transmission_latency(
                inference_id, split_layer
            )
            inf_time_initiator, inf_time_receiver = self.get_total_inference_time(
                inference_id)
            time_to_result = inf_time_initiator + inf_time_receiver + transmission_latency
            return (
                split_layer,
                transmission_latency,
                inf_time_initiator,
                inf_time_receiver,
                time_to_result,
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the MasterDict to a pandas DataFrame."""
        with self._lock:
            flattened_data = []
            layer_attrs = []

            for superfields in self._data.values():
                inf_id = superfields["inference_id"]
                supermetrics = self.calculate_supermetrics(inf_id)

                for subdict in superfields["layer_information"].values():
                    layer_id = subdict.pop("layer_id")
                    if not layer_attrs:
                        layer_attrs = list(subdict.keys())
                    row = (
                        inf_id,
                        *supermetrics,
                        layer_id,
                        *(subdict[k] for k in layer_attrs),
                    )
                    flattened_data.append(row)

            flattened_data.sort(key=lambda tup: (tup[0], tup[1]))

            columns = [
                "inference_id",
                "split_layer",
                "transmission_latency_ns",
                "inf_time_initiator",
                "inf_time_receiver",
                "total_time_ns",
                "layer_id",
                *layer_attrs,
            ]
            return pd.DataFrame(flattened_data, columns=columns)

    def to_pickle(self) -> bytes:
        """Serialize the MasterDict to a pickle byte string."""
        with self._lock:
            return pickle.dumps(self._data)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert the MasterDict to a regular dictionary."""
        with self._lock:
            return dict(self._data)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.get(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self.set(key, value)
