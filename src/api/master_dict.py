# src/api/master_dict.py

import logging
import pickle
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rpyc.utils.classic import obtain

logger = logging.getLogger(__name__)


class MasterDict:
    """Thread-safe dictionary to store and manage inference data."""

    def __init__(self) -> None:
        """Initialize MasterDict with a reentrant lock and an empty dictionary."""
        self.lock = threading.RLock()
        self.data: Dict[str, Dict[str, Any]] = {}
        logger.info("MasterDict initialized")

    def set_item(self, inference_id: str, value: Dict[str, Any]) -> None:
        """Set or update the value for a given inference ID."""
        with self.lock:
            if inference_id in self.data:
                layer_info = value.get("layer_information")
                if layer_info is not None:
                    filtered_layer_info = {
                        k: v
                        for k, v in layer_info.items()
                        if v.get("inference_time") is not None
                    }
                    self.data[inference_id]["layer_information"].update(
                        filtered_layer_info
                    )
                    logger.debug(
                        f"Updated key {inference_id} with new layer information"
                    )
                else:
                    logger.error(
                        f"Cannot integrate inference_dict without 'layer_information' for key {inference_id}"
                    )
                    raise ValueError(
                        "Cannot integrate inference_dict without 'layer_information' field."
                    )
            else:
                self.data[inference_id] = value
                logger.debug(f"Added new key {inference_id} to MasterDict")

    def get_item(self, inference_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the value associated with a given inference ID."""
        with self.lock:
            value = self.data.get(inference_id)
            if value:
                logger.debug(f"Retrieved value for key {inference_id}")
            else:
                logger.debug(f"Key {inference_id} not found in MasterDict")
            return value

    def update_data(
        self, new_info: Dict[str, Dict[str, Any]], by_value: bool = True
    ) -> None:
        """Update the master dictionary with new inference data."""
        if by_value:
            new_info = obtain(new_info)
        with self.lock:
            for inference_id, layer_data in new_info.items():
                self.set_item(inference_id, layer_data)
        logger.info(f"Updated MasterDict with {len(new_info)} new entries")

    def get_transmission_latency(
        self, inference_id: str, split_layer: int, mb_per_s: float = 4.0
    ) -> int:
        """Calculate transmission latency in nanoseconds for a split layer."""
        inf_data = self.data.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        # TODO: Replace hardcoded split_layer value (20) with dynamic determination if possible
        if split_layer == 20:
            return 0

        send_layer = split_layer - 1
        sent_output_size = (
            602112
            if split_layer == 0
            else inf_data["layer_information"]
            .get(send_layer, {})
            .get("output_bytes", 0)
        )

        bytes_per_second = mb_per_s * 1e6
        latency_ns = int((sent_output_size / bytes_per_second) * 1e9)
        logger.debug(
            f"Calculated transmission latency for inference {inference_id}, split layer {split_layer}: {latency_ns} ns"
        )
        return latency_ns

    def get_total_inference_time(
        self, inference_id: str, nodes: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """Calculate total inference time for client and edge nodes."""
        if nodes is None:
            nodes = ["SERVER", "PARTICIPANT"]

        inf_data = self.data.get(inference_id)
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
            f"Total inference time for {inference_id}: Client: {inf_time_client} ns, Edge: {inf_time_edge} ns"
        )
        return inf_time_client, inf_time_edge

    def get_split_layer(
        self, inference_id: str, nodes: Optional[List[str]] = None
    ) -> int:
        """Determine the split layer where the model transitions nodes."""
        if nodes is None:
            nodes = ["SERVER", "PARTICIPANT"]

        inf_data = self.data.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        layer_info = inf_data.get("layer_information", {})
        sorted_layers = sorted(layer_info.keys(), key=lambda x: int(x))

        if not sorted_layers:
            logger.error(f"No layer information for inference ID '{inference_id}'")
            raise ValueError(f"No layer information for inference ID '{inference_id}'.")

        start_node = layer_info[sorted_layers[0]].get("completed_by_node")
        for layer_id in sorted_layers:
            current_node = layer_info[layer_id].get("completed_by_node")
            if current_node != start_node:
                split_layer = int(layer_id)
                logger.debug(f"Split layer for inference {inference_id}: {split_layer}")
                return split_layer

        logger.debug(
            f"No split layer found for inference {inference_id}, returning default."
        )
        # TODO: Replace hardcoded return values (0 or 20) with dynamic logic
        return 0 if start_node in nodes else 20

    def calculate_supermetrics(
        self, inference_id: str
    ) -> Tuple[int, int, int, int, int, float]:
        """Calculate various metrics for an inference."""
        logger.info(f"Calculating supermetrics for inference {inference_id}")
        split_layer = self.get_split_layer(inference_id)
        transmission_latency = self.get_transmission_latency(inference_id, split_layer)
        inf_time_client, inf_time_edge = self.get_total_inference_time(inference_id)
        total_time = inf_time_client + inf_time_edge + transmission_latency
        watts_used = self.calculate_total_watts_used(inference_id)
        logger.debug(
            f"Supermetrics for {inference_id}: split_layer={split_layer}, transmission_latency={transmission_latency}, "
            f"inf_time_client={inf_time_client}, inf_time_edge={inf_time_edge}, total_time={total_time}, "
            f"watts_used={watts_used}"
        )
        return (
            split_layer,
            transmission_latency,
            inf_time_client,
            inf_time_edge,
            total_time,
            watts_used,
        )

    def calculate_total_watts_used(self, inference_id: str) -> float:
        """Calculate the total watts used for an inference."""
        inf_data = self.data.get(inference_id)
        if not inf_data:
            logger.error(f"Inference ID '{inference_id}' not found in MasterDict")
            raise KeyError(f"Inference ID '{inference_id}' not found.")

        layer_info = inf_data.get("layer_information", {})
        total_watts = sum(
            float(layer.get("watts_used", 0)) for layer in layer_info.values()
        )
        logger.debug(f"Total watts used for inference {inference_id}: {total_watts}")
        return total_watts

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the master dictionary into a pandas DataFrame."""
        logger.info("Converting MasterDict to DataFrame")
        records = []
        layer_attributes = set()

        with self.lock:
            for superfields in self.data.values():
                inference_id = superfields.get("inference_id")
                if not inference_id:
                    logger.warning(
                        f"Skipping entry without inference_id: {superfields}"
                    )
                    continue

                try:
                    (
                        split_layer,
                        transmission_latency,
                        inf_time_client,
                        inf_time_edge,
                        total_time,
                        watts_used,
                    ) = self.calculate_supermetrics(inference_id)
                except (KeyError, ValueError) as e:
                    logger.error(
                        f"Error calculating supermetrics for {inference_id}: {e}"
                    )
                    continue

                for layer in superfields.get("layer_information", {}).values():
                    layer_id = layer.get("layer_id")
                    if layer_id is None:
                        logger.warning(
                            f"Skipping layer without layer_id for inference {inference_id}"
                        )
                        continue

                    if not layer_attributes:
                        layer_attributes.update(layer.keys())
                        layer_attributes.discard("layer_id")

                    record = {
                        "inference_id": inference_id,
                        "split_layer": split_layer,
                        "total_time_ns": total_time,
                        "inf_time_client": inf_time_client,
                        "inf_time_edge": inf_time_edge,
                        "transmission_latency_ns": transmission_latency,
                        "watts_used": watts_used,
                        "layer_id": layer_id,
                    }
                    for attr in layer_attributes:
                        record[attr] = layer.get(attr)

                    records.append(record)

        df = pd.DataFrame(records)
        df.sort_values(by=["inference_id", "layer_id"], inplace=True)
        logger.info(f"Created DataFrame with {len(df)} rows")
        return df

    def to_pickle(self) -> bytes:
        """Serialize the master dictionary to a pickle byte stream."""
        logger.info("Serializing MasterDict to pickle")
        with self.lock:
            return pickle.dumps(self.data)

    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        """Enable dictionary-like access for getting items."""
        return self.get_item(key)

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Enable dictionary-like access for setting items."""
        self.set_item(key, value)

    def __iter__(self):
        """Enable iteration over the master dictionary."""
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of items in the master dictionary."""
        return len(self.data)
