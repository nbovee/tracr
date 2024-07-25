import pytest
import pandas as pd
from src.tracr.app_api.master_dict import MasterDict


@pytest.fixture
def master_dict():
    return MasterDict()


@pytest.fixture
def sample_data():
    return {
        "inference_1": {
            "inference_id": "inference_1",
            "layer_information": {
                0: {
                    "layer_id": 0,
                    "completed_by_node": "CLIENT1",
                    "inference_time": 100,
                    "output_bytes": 1000,
                },
                1: {
                    "layer_id": 1,
                    "completed_by_node": "EDGE1",
                    "inference_time": 200,
                    "output_bytes": 2000,
                },
            },
        },
        "inference_2": {
            "inference_id": "inference_2",
            "layer_information": {
                0: {
                    "layer_id": 0,
                    "completed_by_node": "CLIENT1",
                    "inference_time": 150,
                    "output_bytes": 1500,
                },
                1: {
                    "layer_id": 1,
                    "completed_by_node": "CLIENT1",
                    "inference_time": 250,
                    "output_bytes": 2500,
                },
                2: {
                    "layer_id": 2,
                    "completed_by_node": "EDGE1",
                    "inference_time": 300,
                    "output_bytes": 3000,
                },
            },
        },
    }


def test_set_and_get(master_dict, sample_data):
    master_dict.set("inference_1", sample_data["inference_1"])
    assert master_dict.get("inference_1") == sample_data["inference_1"]
    assert master_dict.get("non_existent") is None


def test_set_invalid_data(master_dict):
    with pytest.raises(ValueError):
        master_dict.set("invalid", {"no_layer_info": True})


def test_update(master_dict, sample_data):
    master_dict.update(sample_data)
    assert master_dict.get("inference_1") == sample_data["inference_1"]
    assert master_dict.get("inference_2") == sample_data["inference_2"]


def test_get_transmission_latency(master_dict, sample_data):
    master_dict.update(sample_data)
    latency = master_dict.get_transmission_latency("inference_1", 1)
    assert isinstance(latency, int)
    assert latency > 0


def test_get_total_inference_time(master_dict, sample_data):
    master_dict.update(sample_data)
    client_time, edge_time = master_dict.get_total_inference_time("inference_1")
    assert client_time == 100
    assert edge_time == 200


def test_get_split_layer(master_dict, sample_data):
    master_dict.update(sample_data)
    assert master_dict.get_split_layer("inference_1") == 1
    assert master_dict.get_split_layer("inference_2") == 2


def test_calculate_supermetrics(master_dict, sample_data):
    master_dict.update(sample_data)
    metrics = master_dict.calculate_supermetrics("inference_1")
    assert len(metrics) == 5
    assert all(isinstance(m, int) for m in metrics)


def test_to_dataframe(master_dict, sample_data):
    master_dict.update(sample_data)
    df = master_dict.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5  # Total number of layers in sample_data


def test_to_pickle(master_dict, sample_data):
    master_dict.update(sample_data)
    pickled = master_dict.to_pickle()
    assert isinstance(pickled, bytes)


def test_getitem_setitem(master_dict, sample_data):
    master_dict["inference_1"] = sample_data["inference_1"]
    assert master_dict["inference_1"] == sample_data["inference_1"]


def test_thread_safety(master_dict, sample_data):
    import threading

    def update_dict():
        for _ in range(1000):
            master_dict.update(sample_data)

    threads = [threading.Thread(target=update_dict) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(master_dict._data) == 2
    assert all(
        len(master_dict._data[key]["layer_information"])
        == len(sample_data[key]["layer_information"])
        for key in sample_data
    )
