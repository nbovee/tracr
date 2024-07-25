import threading
import pickle
from typing import Dict, Tuple, Any, Optional
import pandas as pd
from rpyc.utils.classic import obtain


class MasterDict:
    """
    A thread-safe dictionary for managing inference data in a distributed environment.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Set a value in the dictionary, ensuring thread-safety.

        Args:
            key (str): The inference ID.
            value (Dict[str, Any]): The inference data.

        Raises:
            ValueError: If the value doesn't contain required 'layer_information'.
        """
        with self._lock:
            if key in self._data:
                return  # Avoid overwriting existing data
            if "layer_information" not in value:
                raise ValueError(
                    "Cannot integrate inference_dict without 'layer_information' field"
                )
            self._data[key] = value

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from the dictionary.

        Args:
            key (str): The inference ID.

        Returns:
            Optional[Dict[str, Any]]: The inference data if found, None otherwise.
        """
        with self._lock:
            return self._data.get(key)

    def update(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """
        Update the dictionary with new information.

        Args:
            new_info (Dict[str, Dict[str, Any]]): New inference data to add.
            by_value (bool): If True, obtain a local copy of the data.
        """
        if by_value:
            new_info = obtain(new_info)
        with self._lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = 4.0
    ) -> int:
        """
        Calculate the transmission latency for a given inference.

        Args:
            inference_id (str): The inference ID.
            split_layer (int): The layer at which the model is split.
            mb_per_s (float): The network speed in MB/s.

        Returns:
            int: The calculated latency in nanoseconds.
        """
        with self._lock:
            inf_data = self._data[inference_id]
            if split_layer == 20:
                return 0
            send_layer = split_layer - 1
            sent_output_size_bytes = (
                602112
                if split_layer == 0
                else inf_data["layer_information"][send_layer]["output_bytes"]
            )
            bytes_per_second = mb_per_s * 1e6
            latency_s = sent_output_size_bytes / bytes_per_second
            return int(latency_s * 1e9)

    def get_total_inference_time(self, inference_id: str) -> Tuple[int, int]:
        """
        Get the total inference time for client and edge.

        Args:
            inference_id (str): The inference ID.

        Returns:
            Tuple[int, int]: Client and edge inference times in nanoseconds.
        """
        with self._lock:
            inf_data = self._data[inference_id]
            client_time = sum(
                int(layer["inference_time"])
                for layer in inf_data["layer_information"].values()
                if layer["inference_time"] and layer["completed_by_node"] == "CLIENT1"
            )
            edge_time = sum(
                int(layer["inference_time"])
                for layer in inf_data["layer_information"].values()
                if layer["inference_time"] and layer["completed_by_node"] == "EDGE1"
            )
            return client_time, edge_time

    def get_split_layer(self, inference_id: str) -> int:
        """
        Get the split layer for a given inference.

        Args:
            inference_id (str): The inference ID.

        Returns:
            int: The split layer.
        """
        with self._lock:
            inf_data = self._data[inference_id]
            start_node = inf_data["layer_information"][0]["completed_by_node"]
            for layer_id, layer_info in inf_data["layer_information"].items():
                if layer_info["completed_by_node"] != start_node:
                    return int(layer_id)
            return 0 if start_node == "CLIENT1" else 20

    def calculate_supermetrics(
        self, inference_id: str
    ) -> Tuple[int, int, int, int, int]:
        """
        Calculate supermetrics for a given inference.

        Args:
            inference_id (str): The inference ID.

        Returns:
            Tuple[int, int, int, int, int]: Split layer, transmission latency,
                                            client inference time, edge inference time,
                                            and total time to result.
        """
        with self._lock:
            split_layer = self.get_split_layer(inference_id)
            transmission_latency = self.get_transmission_latency(
                inference_id, split_layer
            )
            inf_time_client, inf_time_edge = self.get_total_inference_time(inference_id)
            time_to_result = inf_time_client + inf_time_edge + transmission_latency
            return (
                split_layer,
                transmission_latency,
                inf_time_client,
                inf_time_edge,
                time_to_result,
            )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the MasterDict to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame representation of the MasterDict.
        """
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
                "inf_time_client",
                "inf_time_edge",
                "total_time_ns",
                "layer_id",
                *layer_attrs,
            ]
            return pd.DataFrame(flattened_data, columns=columns)

    def to_pickle(self) -> bytes:
        """
        Serialize the MasterDict to a pickle byte string.

        Returns:
            bytes: Pickled representation of the MasterDict.
        """
        with self._lock:
            return pickle.dumps(self._data)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.get(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self.set(key, value)
