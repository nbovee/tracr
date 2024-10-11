# src/api/master_dict.py

import pickle
import threading
import logging
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from rpyc.utils.classic import obtain

logger = logging.getLogger(__name__)


class MasterDict:
    """Thread-safe dictionary to store and manage inference data."""

    def __init__(self):
        """Initializes the MasterDict with a reentrant lock and an empty dictionary."""
        self.lock = threading.RLock()
        self.inner_dict: Dict[str, Dict[str, Any]] = {}
        logger.info("MasterDict initialized")

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Sets the value for a given key in the master dictionary.

        If the key already exists, it updates the 'layer_information' by filtering out
        layers where 'inference_time' is None.

        Args:
            key (str): The inference ID.
            value (Dict[str, Any]): The inference data containing 'layer_information'.

        Raises:
            ValueError: If 'layer_information' is missing in the provided value.
        """
        with self.lock:
            if key in self.inner_dict:
                layer_info = value.get("layer_information")
                if layer_info is not None:
                    filtered_layer_info = {
                        k: v
                        for k, v in layer_info.items()
                        if v.get("inference_time") is not None
                    }
                    self.inner_dict[key]["layer_information"].update(
                        filtered_layer_info
                    )
                    logger.debug(
                        f"Updated existing key {key} with new layer information"
                    )
                else:
                    logger.error(
                        f"Cannot integrate inference_dict without 'layer_information' field for key {key}"
                    )
                    raise ValueError(
                        "Cannot integrate inference_dict without 'layer_information' field"
                    )
            else:
                self.inner_dict[key] = value
                logger.debug(f"Added new key {key} to MasterDict")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves the value associated with a given key."""
        with self.lock:
            value = self.inner_dict.get(key)
            if value:
                logger.debug(f"Retrieved value for key {key}")
            else:
                logger.debug(f"Key {key} not found in MasterDict")
            return value

    def update(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """Updates the master dictionary with new information.

        If `by_value` is True, it obtains the actual data from a possibly remote object.

        Args:
            new_info (Dict[str, Dict[str, Any]]): New inference data to be integrated.
            by_value (bool, optional): Whether to obtain the actual data. Defaults to True.
        """
        if by_value:
            new_info = obtain(new_info)
        with self.lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)
        logger.info(f"Updated MasterDict with {len(new_info)} new entries")

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = 4.0
    ) -> int:
        """Calculates the transmission latency in nanoseconds for a given split layer.

        Args:
            inference_id (str): The inference ID.
            split_layer (int): The layer at which the model is split.
            mb_per_s (float, optional): Transmission rate in megabytes per second. Defaults to 4.0.

        Returns:
            int: Transmission latency in nanoseconds.
        """
        inf_data = self.inner_dict.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        # TODO: Replace hardcoded split_layer value (20) with dynamic determination if possible
        if split_layer == 20:
            return 0

        send_layer = split_layer - 1
        sent_output_size_bytes = (
            602112
            if split_layer == 0
            else inf_data["layer_information"][send_layer].get("output_bytes", 0)
        )

        bytes_per_second = mb_per_s * 1e6
        latency_ns = int((sent_output_size_bytes / bytes_per_second) * 1e9)
        logger.debug(
            f"Calculated transmission latency for inference {inference_id}, split layer {split_layer}: {latency_ns} ns"
        )
        return latency_ns

    def get_total_inference_time(
        self, inference_id: str, nodes: List[str] = ["CLIENT1", "EDGE1"]
    ) -> Tuple[int, int]:
        """Calculates the total inference time for client and edge nodes."""
        inf_data = self.inner_dict.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        layer_info = inf_data.get("layer_information", {})
        inf_time_client = sum(
            int(layer["inference_time"])
            for layer in layer_info.values()
            if layer.get("inference_time") and layer.get("completed_by_node") in nodes
        )
        inf_time_edge = sum(
            int(layer["inference_time"])
            for layer in layer_info.values()
            if layer.get("inference_time") and layer.get("completed_by_node") in nodes
        )
        logger.debug(
            f"Calculated total inference time for {inference_id}: Client: {inf_time_client} ns, Edge: {inf_time_edge} ns"
        )
        return inf_time_client, inf_time_edge

    def get_split_layer(
        self, inference_id: str, nodes: List[str] = ["CLIENT1", "EDGE1"]
    ) -> int:
        """Determines the split layer where the model transitions from one node to another."""
        inf_data = self.inner_dict.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        layer_info = inf_data.get("layer_information", {})
        sorted_layers = sorted(layer_info.keys())
        if not sorted_layers:
            logger.error(
                f"No layer information found for inference ID '{inference_id}'"
            )
            raise ValueError(
                f"No layer information found for inference ID '{inference_id}'."
            )

        start_node = layer_info[sorted_layers[0]].get("completed_by_node")
        for layer_id in sorted_layers:
            current_node = layer_info[layer_id].get("completed_by_node")
            if current_node != start_node:
                logger.debug(
                    f"Split layer for inference {inference_id} determined: {layer_id}"
                )
                return int(layer_id)

        logger.debug(
            f"No split layer found for inference {inference_id}, returning default value"
        )
        # TODO: Replace hardcoded return values (0 or 20) with dynamic logic
        return 0 if start_node in nodes else 20

    def calculate_supermetrics(
        self, inference_id: str
    ) -> Tuple[int, int, int, int, int]:
        """Calculates various metrics for an inference."""
        logger.info(f"Calculating supermetrics for inference {inference_id}")
        split_layer = self.get_split_layer(inference_id)
        transmission_latency = self.get_transmission_latency(inference_id, split_layer)
        inf_time_client, inf_time_edge = self.get_total_inference_time(inference_id)
        total_time_to_result = inf_time_client + inf_time_edge + transmission_latency
        logger.debug(
            f"Supermetrics for {inference_id}: split_layer={split_layer}, transmission_latency={transmission_latency}, "
            f"inf_time_client={inf_time_client}, inf_time_edge={inf_time_edge}, total_time={total_time_to_result}"
        )
        return (
            split_layer,
            transmission_latency,
            inf_time_client,
            inf_time_edge,
            total_time_to_result,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the master dictionary into a pandas DataFrame."""
        logger.info("Converting MasterDict to DataFrame")
        records = []
        layer_attrs = set()

        with self.lock:
            for superfields in self.inner_dict.values():
                inf_id = superfields.get("inference_id")
                if not inf_id:
                    logger.warning(
                        f"Skipping entry without inference_id: {superfields}"
                    )
                    continue

                try:
                    (
                        split_layer,
                        trans_latency,
                        inf_time_client,
                        inf_time_edge,
                        total_time_to_result,
                    ) = self.calculate_supermetrics(inf_id)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating supermetrics for {inf_id}: {e}")
                    continue

                for layer_id, subdict in superfields.get(
                    "layer_information", {}
                ).items():
                    layer_id = subdict.get("layer_id")
                    if layer_id is None:
                        logger.warning(
                            f"Skipping layer without layer_id for inference {inf_id}"
                        )
                        continue

                    if not layer_attrs:
                        layer_attrs.update(subdict.keys())
                        layer_attrs.discard("layer_id")

                    row = {
                        "inference_id": inf_id,
                        "split_layer": split_layer,
                        "total_time_ns": total_time_to_result,
                        "inf_time_client": inf_time_client,
                        "inf_time_edge": inf_time_edge,
                        "transmission_latency_ns": trans_latency,
                        "layer_id": layer_id,
                    }
                    for attr in layer_attrs:
                        row[attr] = subdict.get(attr)

                    records.append(row)

        df = pd.DataFrame(records)
        df.sort_values(by=["inference_id", "layer_id"], inplace=True)
        logger.info(f"Created DataFrame with {len(df)} rows")
        return df

    def to_pickle(self) -> bytes:
        """Serializes the master dictionary to a pickle byte stream."""
        logger.info("Serializing MasterDict to pickle")
        with self.lock:
            return pickle.dumps(self.inner_dict)

    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        """Enables dictionary-like access for getting items."""
        return self.get(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Enables dictionary-like access for setting items."""
        self.set(key, value)
