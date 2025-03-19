"""Tests for the hooks module in experiment_design.models.hooks."""

import os
import sys
import unittest
from unittest.mock import MagicMock
import logging
import torch
import torch.nn as nn

# Fix the path to include the project root, not just the parent directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiment_design.models.hooks import (   # noqa: E402
    EarlyOutput,
    HookExitException,
    create_forward_prehook,
    create_forward_posthook,
)

# Setup test logger
logging.basicConfig(level=logging.ERROR)


class SimpleModel(nn.Module):
    """Simple model for testing hooks."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TestEarlyOutput(unittest.TestCase):
    """Tests for the EarlyOutput wrapper class."""

    def test_early_output_with_dict(self):
        """Test EarlyOutput with dictionary input."""
        data = {"layer0": torch.ones(1, 1), "layer1": torch.zeros(1, 1)}
        early_output = EarlyOutput(data)

        # Test callable behavior
        result = early_output()
        self.assertEqual(data, result)
        self.assertIs(data, result)

        # Test that shape property returns None for dict
        self.assertIsNone(early_output.shape)

    def test_early_output_with_tensor(self):
        """Test EarlyOutput with tensor input."""
        tensor = torch.ones(1, 3, 224, 224)
        early_output = EarlyOutput(tensor)

        # Test callable behavior
        result = early_output()
        self.assertTrue(torch.equal(tensor, result))
        self.assertIs(tensor, result)

        # Test shape property
        self.assertEqual(tensor.shape, early_output.shape)


class TestHookExitException(unittest.TestCase):
    """Tests for the HookExitException class."""

    def test_exception_carries_result(self):
        """Test that the exception properly carries the result."""
        data = {"layer0": torch.ones(1, 1)}

        with self.assertRaises(HookExitException) as context:
            raise HookExitException(data)

        exception = context.exception
        self.assertIs(data, exception.result)


class TestHookCreation(unittest.TestCase):
    """Tests for hook creation functions."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()
        self.wrapped_model = MagicMock()
        self.wrapped_model.start_i = 0
        self.wrapped_model.stop_i = 2
        self.wrapped_model.log = True
        self.wrapped_model.layer_times = {}
        self.wrapped_model.metrics_collector = MagicMock()
        self.wrapped_model.banked_output = {}
        self.wrapped_model.forward_info = {0: {}, 1: {}, 2: {}, 3: {}}
        self.wrapped_model.save_layers = {}

        # Create a sample tensor
        self.input_tensor = torch.ones(1, 3, 32, 32)
        self.device = "cpu"

    def test_create_forward_prehook(self):
        """Test creation and execution of forward pre-hook."""
        # Create a pre-hook for layer 0
        pre_hook = create_forward_prehook(
            self.wrapped_model, 0, "Conv2d", self.input_tensor.shape, self.device
        )

        # Execute the pre-hook
        module = self.model.conv
        hook_output = pre_hook(module, (self.input_tensor,))

        # Verify the hook output (using torch.equal instead of assertIs)
        # Since tuple may be reconstructed, we need to check the tensor content
        self.assertEqual(len(hook_output), 1)
        self.assertTrue(torch.equal(hook_output[0], self.input_tensor))

        # Verify metrics collection was started
        self.wrapped_model.metrics_collector.start_layer_measurement.assert_called_once_with(
            0
        )

    def test_create_forward_prehook_cloud_mode(self):
        """Test pre-hook in cloud mode (start_i > 0)."""
        # Set up wrapped model for cloud mode
        self.wrapped_model.start_i = 1
        self.wrapped_model.input_size = (3, 32, 32)

        # Mock banked output that will be used by the hook
        banked_output = {"0": torch.ones(1, 16, 32, 32)}

        # Create a function that returns the banked output (similar to EarlyOutput behavior)
        tensor_func = lambda: banked_output # noqa: E731

        # Create a pre-hook for layer 0
        pre_hook = create_forward_prehook(
            self.wrapped_model, 0, "Conv2d", self.input_tensor.shape, self.device
        )

        # Execute the pre-hook
        module = self.model.conv
        hook_output = pre_hook(module, (tensor_func,))

        # Verify the result is a tensor with the right dimensions
        self.assertIsInstance(hook_output, torch.Tensor)
        self.assertEqual(hook_output.shape, (1, 3, 32, 32))

    def test_create_forward_posthook(self):
        """Test creation and execution of forward post-hook."""
        # Create a post-hook for layer 0
        post_hook = create_forward_posthook(
            self.wrapped_model, 0, "Conv2d", self.input_tensor.shape, self.device
        )

        # Create an output tensor
        output_tensor = torch.ones(1, 16, 30, 30)

        # Execute the post-hook
        module = self.model.conv
        result = post_hook(module, (self.input_tensor,), output_tensor)

        # Verify the output
        self.assertIs(result, output_tensor)

        # Verify metrics were collected - only pass layer_idx and tensor output
        # This matches the actual function signature in the implementation
        self.wrapped_model.metrics_collector.end_layer_measurement.assert_called_once_with(
            0, output_tensor
        )

    def test_posthook_banking_and_exit(self):
        """Test that post-hook banks output and exits at stop_i."""
        # Create a post-hook for layer 2 (which is at our stop_i)
        post_hook = create_forward_posthook(
            self.wrapped_model, 2, "ReLU", self.input_tensor.shape, self.device
        )

        # Create an output tensor
        output_tensor = torch.ones(1, 16, 30, 30)

        # Execute the post-hook, expect exception
        module = self.model.relu
        with self.assertRaises(HookExitException) as context:
            post_hook(module, (self.input_tensor,), output_tensor)

        # Verify the banked output (using torch.equal for tensor comparison)
        self.assertTrue(torch.equal(self.wrapped_model.banked_output[2], output_tensor))

        # Verify the exception carries our banked output
        self.assertIs(context.exception.result, self.wrapped_model.banked_output)

    def test_posthook_cloud_mode(self):
        """Test post-hook in cloud mode (start_i > 0)."""
        # Set up wrapped model for cloud mode
        self.wrapped_model.start_i = 1
        self.wrapped_model.banked_output = {
            2: torch.ones(1, 16, 15, 15)  # Pre-stored output
        }

        # Create a post-hook for layer 2
        post_hook = create_forward_posthook(
            self.wrapped_model, 2, "ReLU", self.input_tensor.shape, self.device
        )

        # Create a different output tensor
        output_tensor = torch.zeros(1, 16, 15, 15)  # This should be overridden

        # Execute the post-hook
        module = self.model.relu
        result = post_hook(module, (self.input_tensor,), output_tensor)

        # Verify we got the banked output instead of the actual output
        self.assertIs(result, self.wrapped_model.banked_output[2])
        self.assertTrue(torch.equal(result, torch.ones(1, 16, 15, 15)))


if __name__ == "__main__":
    unittest.main()
