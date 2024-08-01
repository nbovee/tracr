"""Basic invocation script for model_hooked module"""

from pathlib import Path
import os
import logging
import torch

from src.tracr.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]), "model_test.yaml"
)
m = WrappedModel(config_path=yaml_file_path)
m2 = WrappedModel(config_path=yaml_file_path)

test_image = torch.randn(1, *m.input_size)

for i in range(1, m.layer_count):
    logging.info(f"Switch at: {i}")
    res = m(test_image, end=i)
    logging.info("Switched.")
    m2(res, start=i)


""" CURRENT TASK
do NOT save layers if they would not be needed upstream
completing model should update its self.drop_save_dict file based on what it receives
cleanup hooks and potentially create generator class"""
