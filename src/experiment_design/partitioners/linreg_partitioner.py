# src/experiment_design/partitioners/linreg_partitioner.py

import logging
import os
import csv
import pickle
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .partitioner import Partitioner

logger = logging.getLogger(__name__)


class LinearRegression(nn.Module):
    """A simple linear regression model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.training_iterations = 20
        self.loss_history: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        assert isinstance(x, torch.Tensor)
        return self.model(x.unsqueeze(0))

    def train_step(self, y: torch.Tensor, pred: torch.Tensor) -> None:
        """Perform a single training step."""
        assert isinstance(y, torch.Tensor) and isinstance(pred, torch.Tensor)
        loss = self.criterion(pred, y.unsqueeze(0))
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_weights(self, weight: float, bias: float) -> None:
        """Manually set the weights of the model."""
        with torch.no_grad():
            self.model.weight.fill_(weight)
            self.model.bias.fill_(bias)

    def scale_bias(self, scale_factor: float) -> None:
        """Scale the bias of the model."""
        with torch.no_grad():
            self.model.bias.mul_(scale_factor)


class RegressionPartitioner(Partitioner):
    _TYPE = "regression"

    def __init__(self, num_breakpoints: int, clip_min_max: bool = True) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.clip = clip_min_max
        self.regression: Dict[str, LinearRegression] = {}
        self.module_sequence: List[Tuple[str, int, int]] = []
        self.num_modules: int = 0
        self._dir = (
            "src/tracr/app_api/test_cases/alexnetsplit/partitioner_datapoints/local/"
        )
        self.server_regression = None
        logger.info(
            f"Initialized RegressionPartitioner with {num_breakpoints} breakpoints"
        )

    def pass_regression_copy(self) -> bytes:
        """Return a pickled copy of the regression models."""
        return pickle.dumps(self.regression)

    def add_server_module(self, server_modules: Any) -> None:
        """Add server modules to the partitioner."""
        self.server_regression = server_modules

    def estimate_split_point(self, starting_layer: int) -> int:
        logger.debug(f"Estimating split point from layer {starting_layer}")
        for module, param_bytes, output_bytes in self.module_sequence:
            local_time_est_s = (
                self.regression[module].forward(torch.tensor(float(param_bytes))).item()
                * 1e-9
            )
            server_time_est_s = (
                0
                if self.server_regression is None
                else self.server_regression[module]
                .forward(torch.tensor(float(param_bytes)))
                .item()
                * 1e-9
            )
            output_transfer_time = output_bytes / self._get_network_speed_bytes()

            if local_time_est_s < output_transfer_time + server_time_est_s:
                starting_layer += 1
            else:
                logger.info(f"Estimated split point: {starting_layer}")
                return starting_layer
        logger.info(f"Reached end of modules. Split point: {starting_layer}")
        return starting_layer

    def create_data(self, model: Any, iterations: int = 10) -> None:
        logger.info(f"Creating data for regression analysis. Iterations: {iterations}")
        for f in os.listdir(self._dir):
            os.remove(os.path.join(self._dir, f))

        for i in range(iterations):
            logger.debug(f"Data creation iteration {i + 1}/{iterations}")
            model(torch.randn(1, *model.base_input_size), inference_id="profile")
            from_model = model.master_dict.pop("profile")["layer_information"].values()
            self._process_model_data(from_model)

    def _process_model_data(self, data: Any) -> None:
        """Process and save model data for regression analysis."""
        output_bytes = None
        for datapoint in data:
            if len(self.module_sequence) < self.breakpoints:
                self.module_sequence.append(
                    (
                        datapoint["class"],
                        datapoint["parameter_bytes"],
                        datapoint["output_bytes"],
                    )
                )

            selected_value = (
                datapoint["parameter_bytes"]
                if datapoint["parameter_bytes"] != 0
                else output_bytes
            )
            self._save_datapoint(
                datapoint["class"], selected_value, datapoint["inference_time"]
            )
            output_bytes = datapoint["output_bytes"]

    def _save_datapoint(self, class_name: str, x: float, y: float) -> None:
        """Save a single datapoint to a CSV file."""
        with open(os.path.join(self._dir, f"{class_name}.csv"), "a") as f:
            f.write(f"{x}, {y}\n")

    def update_regression(self) -> None:
        """Update regression models based on collected data."""
        for layer_type in os.listdir(self._dir):
            x, y = self._load_data(os.path.join(self._dir, layer_type))
            self.regression[layer_type.split(".")[0]] = self._train_regression(x, y)

    def _load_data(self, filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from a CSV file."""
        x, y = [], []
        with open(filepath) as f:
            reader = csv.reader(f)
            for line in reader:
                x.append(float(line[0]))
                y.append(float(line[1]))
        return torch.tensor(x), torch.tensor(y)

    def _train_regression(self, x: torch.Tensor, y: torch.Tensor) -> LinearRegression:
        model = LinearRegression()
        if x.max() == x.min():
            logger.warning("Insufficient data for regression, setting w=0 b=median")
            model.set_weights(0, torch.median(y))
        else:
            mmax = max(x.max(), y.max())
            x_norm, y_norm = x / mmax, y / mmax
            for i in range(model.training_iterations):
                for v, z in zip(x_norm, y_norm):
                    pred = model.forward(v)
                    model.train_step(z, pred)
                logger.debug(
                    f"Training iteration {i + 1}/{model.training_iterations}, Loss: {model.loss_history[-1]}"
                )
            model.scale_bias(mmax)
        return model

    def _get_network_speed_bytes(self, artificial_value: int = 4 * 1024**2) -> int:
        """Get the network speed in bytes per second."""
        logger.debug("Getting network speed (artificial value)")
        return artificial_value  # TODO: Implement actual network speed measurement

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        logger.debug("Calling RegressionPartitioner to estimate split point")
        return self.estimate_split_point(starting_layer=0)
