"""Local experiment implementation."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .base import BaseExperiment, ProcessingTimes

logger = logging.getLogger("split_computing_logger")


class LocalExperiment(BaseExperiment):
    """Experiment implementation for local (non-networked) computing.

    This class handles experiments where the entire model runs on
    the local device, without any network interaction.
    """

    def __init__(
        self, config: Dict[str, Any], host: str = None, port: int = None
    ) -> None:
        """Initialize local experiment.

        Args:
            config: Dictionary containing experiment configuration.
            host: Unused, included for API compatibility.
            port: Unused, included for API compatibility.
        """
        super().__init__(config, host, port)

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Optional[Path],
    ) -> Optional[ProcessingTimes]:
        """Process a single image locally.

        Args:
            inputs: The input tensor for the model.
            class_idx: Class index for ground truth.
            image_file: Name of the image file.
            split_layer: Index of the layer (unused for local processing).
            output_dir: Directory to save intermediate results, can be None.

        Returns:
            ProcessingTimes object with timing measurements, or None if an error occurs.
        """
        try:
            start_time = time.time()
            with torch.no_grad():
                # Move input tensor to the proper device.
                inputs = inputs.to(self.device, non_blocking=True)
                output = self.model(inputs)
                if isinstance(output, tuple):
                    output = output[0]

            # Reconstruct the original image (or load it from dataset).
            original_image = self._get_original_image(inputs.cpu(), image_file)
            # Process the output using the post processor.
            processed_result = self.post_processor.process_output(
                output.cpu() if output.device != torch.device("cpu") else output,
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

            # Return timing even if metrics collection is disabled to maintain function signature
            return ProcessingTimes(
                host_time=total_time, travel_time=0.0, server_time=0.0
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test local computing performance for a given split layer.

        For local experiments, the split layer is not actually used, as all
        computation happens on the local device.

        Args:
            split_layer: Index of the layer (included for API compatibility).

        Returns:
            Tuple of (split_layer, host_time, 0.0, 0.0).
        """
        # Check if paths is configured before using it
        split_dir = None
        if self.paths and self.paths.images_dir:
            split_dir = self.paths.images_dir / f"split_{split_layer}"
            split_dir.mkdir(exist_ok=True)
            logger.info(f"Saving split layer images to {split_dir}")
        else:
            logger.warning("No output directory configured. Images won't be saved.")

        with (
            torch.no_grad(),
            torch.cuda.amp.autocast(enabled=torch.cuda.is_available()),
        ):
            times = [
                result
                for batch in tqdm(self.data_loader, desc="Processing locally")
                for input_tensor, class_idx, image_file in zip(*batch)
                if (
                    result := self.process_single_image(
                        input_tensor.unsqueeze(0),
                        class_idx,
                        image_file,
                        split_layer,
                        split_dir,
                    )
                )
                is not None
            ]

        if times:
            total_host = sum(t.host_time for t in times)
            avg_host = total_host / len(times)

            if self.collect_metrics:
                logger.info(
                    f"Local processing: average time per image = {avg_host:.4f}s"
                )
                self._log_performance_summary(avg_host, 0.0, 0.0)

            return split_layer, avg_host, 0.0, 0.0

        return split_layer, 0.0, 0.0, 0.0
