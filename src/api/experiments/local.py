"""Local experiment implementation"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .base import BaseExperiment, ProcessingTimes

logger = logging.getLogger("split_computing_logger")


class LocalExperiment(BaseExperiment):
    """Experiment implementation for local (non-networked) computing."""

    def __init__(
        self, config: Dict[str, Any], host: str = None, port: int = None
    ) -> None:
        """Initialize local experiment with configuration parameters."""
        super().__init__(config, host, port)

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Optional[Path],
    ) -> Optional[ProcessingTimes]:
        """Process a single image through the complete model locally.

        Unlike networked implementations, all computation occurs on a single device
        with no tensor transmission over the network.
        """
        try:
            start_time = time.time()
            with torch.no_grad():
                # Move input tensor to model's device (CPU/GPU) for inference
                inputs = inputs.to(self.device, non_blocking=True)
                # Forward pass through the entire model at once
                output = self.model(inputs)
                if isinstance(output, tuple):
                    output = output[0]

            # Move input back to CPU for post-processing
            original_image = self._get_original_image(inputs.cpu(), image_file)

            # Ensure output tensor is on CPU before post-processing
            output_cpu = (
                output.cpu() if output.device != torch.device("cpu") else output
            )
            processed_result = self.post_processor.process_output(
                output_cpu,
                self.post_processor.get_input_size(original_image),
            )
            total_time = time.time() - start_time

            # Only save visualization if output_dir is provided
            if output_dir and self.config.get("default", {}).get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            # Return timing with zeros for network-related metrics
            return ProcessingTimes(
                host_time=total_time, travel_time=0.0, server_time=0.0
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test model performance with all computation occurring locally.

        The split_layer parameter is maintained for API compatibility with networked
        implementations but has no effect on the execution as no actual splitting occurs.
        """
        # Create directory for saving split-specific outputs if configured
        split_dir = None
        if self.paths and self.paths.images_dir:
            split_dir = self.paths.images_dir / f"split_{split_layer}"
            split_dir.mkdir(exist_ok=True)
            logger.info(f"Saving split layer images to {split_dir}")
        else:
            logger.warning("No output directory configured. Images won't be saved.")

        # Process all images in the dataset with optimizations enabled
        with (
            torch.no_grad(),  # Disable gradient calculation
            torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ),  # Use mixed precision if CUDA available
        ):
            times = [
                result
                for batch in tqdm(self.data_loader, desc="Processing locally")
                for input_tensor, class_idx, image_file in zip(*batch)
                if (
                    result := self.process_single_image(
                        input_tensor.unsqueeze(0),  # Add batch dimension
                        class_idx,
                        image_file,
                        split_layer,
                        split_dir,
                    )
                )
                is not None
            ]

        # Calculate and report performance metrics
        if times:
            total_host = sum(t.host_time for t in times)
            avg_host = total_host / len(times)

            if self.collect_metrics:
                logger.info(
                    f"Local processing: average time per image = {avg_host:.4f}s"
                )
                self._log_performance_summary(avg_host, 0.0, 0.0)

            return split_layer, total_host, 0.0, 0.0

        return split_layer, 0.0, 0.0, 0.0
