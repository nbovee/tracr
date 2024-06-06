import pickle
import threading
import pandas as pd
from rpyc.utils.classic import obtain


# TODO: fix all the hardcoding for edge1 and client1 - make it a generic initiator and receiver or something


class MasterDict:

    def __init__(self):
        self.lock = threading.RLock()
        self.inner_dict = {}

    def set(self, key: str, value: dict):
        with self.lock:
            if key in self.inner_dict:
                if value.get("layer_information"):
                    layer_info = value["layer_information"]
                    layer_info = {
                        k: v
                        for k, v in layer_info.items()
                        if v["inference_time"] is not None
                    }
                else:
                    raise ValueError(
                        "Cannot integrate inference_dict without 'layer_information' field"
                    )
                return
            self.inner_dict[key] = value

    def get(self, key: str):
        with self.lock:
            return self.inner_dict.get(key)

    def update(self, new_info: dict, by_value=True):
        if by_value:
            new_info = obtain(new_info)
        with self.lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)

    def get_transmission_latency(
        self, inference_id: str, split_layer: str, mb_per_s: float = 4.0
    ) -> int:
        inf_data = self.inner_dict[inference_id]
        split_layer = split_layer
        # TODO: fix hardcoding
        if split_layer == 20:
            return 0
        send_layer = int(split_layer - 1)  # type: ignore
        if split_layer == 0:
            sent_output_size_bytes = 602112
        else:
            sent_output_size_bytes = inf_data["layer_information"][send_layer][
                "output_bytes"
            ]
        bytes_per_second = mb_per_s * 1e6
        latency_s = sent_output_size_bytes / bytes_per_second
        latency_ns = int(latency_s * 1e9)
        return latency_ns

    def get_total_inference_time(self, inference_id: str) -> tuple[int, int]:
        inf_data = self.inner_dict[inference_id]

        # layer_times = [
        #     layer["inference_time"]
        #     for layer in inf_data["layer_information"].values()
        #     if layer["inference_time"]
        # ]

        elayer_times = [
            int(layer["inference_time"])
            for layer in inf_data["layer_information"].values()
            if layer["inference_time"] and layer["completed_by_node"] == "EDGE1"
        ]
        clayer_times = [
            int(layer["inference_time"])
            for layer in inf_data["layer_information"].values()
            if layer["inference_time"] and layer["completed_by_node"] == "CLIENT1"
        ]
        return int(sum(clayer_times)), int(sum(elayer_times))

    def get_split_layer(self, inference_id: str) -> int:
        inf_data = self.inner_dict[inference_id]
        layer_ids = sorted(list(inf_data["layer_information"].keys()))
        start_node = inf_data["layer_information"][0]["completed_by_node"]
        for layer_id in layer_ids:
            if (
                inf_data["layer_information"][layer_id]["completed_by_node"]
                != start_node
            ):
                return layer_id  # type: ignore
        # TODO: fix hardcoding
        if start_node == "CLIENT1":
            return 0
        else:
            return 20

    def calculate_supermetrics(
        self, inference_id: str
    ) -> tuple[int, int, int, int, int]:
        split_layer = self.get_split_layer(inference_id)
        transmission_latency = self.get_transmission_latency(inference_id, split_layer)  # type: ignore
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
        flattened_data = []
        layer_attrs = []

        for superfields in self.inner_dict.values():
            inf_id = superfields["inference_id"]
            (
                split_layer,
                trans_latency,
                inf_time_client,
                inf_time_edge,
                total_time_to_result,
            ) = self.calculate_supermetrics(inf_id)
            for subdict in superfields["layer_information"].values():
                layer_id = subdict.pop("layer_id")
                if not layer_attrs:
                    layer_attrs = [key for key in subdict.keys()]
                row = (
                    inf_id,
                    split_layer,
                    total_time_to_result,
                    inf_time_client,
                    inf_time_edge,
                    trans_latency,
                    layer_id,
                    *[subdict[k] for k in layer_attrs],
                )
                flattened_data.append(row)

        flattened_data.sort(key=lambda tup: (tup[0], tup[1]))

        columns = [
            "inference_id",
            "split_layer",
            "total_time_ns",
            "inf_time_client",
            "inf_time_edge",
            "transmission_latency_ns",
            "layer_id",
            *layer_attrs,
        ]
        df = pd.DataFrame(flattened_data, columns=columns)

        return df

    def to_pickle(self):
        return pickle.dumps(self.inner_dict)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, newvalue: dict):
        return self.set(key, newvalue)
