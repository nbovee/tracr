"""Transforms module for dataset preprocessing operations."""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image

from .exceptions import DatasetTransformError

logger = logging.getLogger("split_computing_logger")


class TransformType(Enum):
    """Enumeration of supported transform types."""

    IMAGENET = "imagenet"
    ONION = "onion"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class NormalizationParams:
    """Standard normalization parameters for various datasets."""

    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Default values for datasets without standard normalization
    DEFAULT_MEAN = [0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5]

    @classmethod
    def get_params(
        cls, transform_type: Union[str, TransformType]
    ) -> Tuple[List[float], List[float]]:
        """Get mean and std normalization parameters for a transform type.

        Args:
            transform_type: The type of transform to get parameters for

        Returns:
            Tuple containing mean and std normalization parameters
        """
        if isinstance(transform_type, str):
            transform_type = TransformType(transform_type.lower())

        if transform_type == TransformType.IMAGENET:
            return cls.IMAGENET_MEAN, cls.IMAGENET_STD
        else:
            return cls.DEFAULT_MEAN, cls.DEFAULT_STD


class TransformFactory:
    """Factory class for creating transform pipelines."""

    DEFAULT_TRANSFORMS: Dict[TransformType, Callable] = {}

    @classmethod
    def get_transform(
        cls,
        transform_type: Union[str, TransformType] = TransformType.IMAGENET,
        **kwargs: Any,
    ) -> Callable:
        """Get a transform pipeline for the specified transform type.

        Args:
            transform_type: The type of transform to create
            **kwargs: Additional arguments to customize the transform

        Returns:
            A callable transform pipeline

        Raises:
            DatasetTransformError: If the transform type is not supported
        """
        try:
            if isinstance(transform_type, str):
                transform_type = TransformType(transform_type.lower())

            # Initialize default transforms dict if not already populated
            if not cls.DEFAULT_TRANSFORMS:
                cls._initialize_default_transforms()

            if transform_type == TransformType.CUSTOM:
                return cls._create_custom_transform(**kwargs)
            else:
                return cls.DEFAULT_TRANSFORMS[transform_type]
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid transform type: {transform_type}")
            raise DatasetTransformError(
                f"Invalid transform type: {transform_type}",
                transform_name=str(transform_type),
            ) from e

    @classmethod
    def _initialize_default_transforms(cls) -> None:
        """Initialize the dictionary of default transforms."""
        # ImageNet standard preprocessing
        cls.DEFAULT_TRANSFORMS[TransformType.IMAGENET] = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=NormalizationParams.IMAGENET_MEAN,
                    std=NormalizationParams.IMAGENET_STD,
                ),
            ]
        )

        # Onion dataset preprocessing
        cls.DEFAULT_TRANSFORMS[TransformType.ONION] = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        # Minimal preprocessing (just convert to tensor)
        cls.DEFAULT_TRANSFORMS[TransformType.MINIMAL] = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    @classmethod
    def _create_custom_transform(cls, **kwargs: Any) -> Callable:
        """Create a custom transform pipeline based on provided parameters.

        Args:
            **kwargs: Configuration parameters for the custom transform

        Returns:
            A custom transform pipeline

        Raises:
            DatasetTransformError: If required parameters are missing
        """
        transforms_list = []

        # Apply resize if specified
        if "resize" in kwargs:
            size = kwargs["resize"]
            if isinstance(size, (tuple, list)) and len(size) == 2:
                transforms_list.append(T.Resize(size))
            elif isinstance(size, int):
                transforms_list.append(T.Resize(size))
            else:
                logger.warning(
                    f"Invalid resize parameter: {size}, expected int or (height, width) tuple"
                )

        # Apply center crop if specified
        if "crop_size" in kwargs:
            crop_size = kwargs["crop_size"]
            if isinstance(crop_size, int):
                transforms_list.append(T.CenterCrop(crop_size))
            else:
                logger.warning(
                    f"Invalid crop_size parameter: {crop_size}, expected int"
                )

        # Apply data augmentation if specified
        if kwargs.get("augment", False):
            transforms_list.extend(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                ]
            )

        # Always convert to tensor
        transforms_list.append(T.ToTensor())

        # Apply normalization if specified
        if kwargs.get("normalize", True):
            norm_type = kwargs.get("norm_type", "imagenet")
            mean, std = NormalizationParams.get_params(norm_type)
            transforms_list.append(T.Normalize(mean=mean, std=std))

        if not transforms_list:
            logger.warning("No transforms specified, using minimal transform")
            return cls.DEFAULT_TRANSFORMS[TransformType.MINIMAL]

        return T.Compose(transforms_list)


class ImageTransformer:
    """Utility class for applying transformations to images."""

    @staticmethod
    def apply_transform(
        image: Image.Image, transform: Optional[Callable] = None
    ) -> torch.Tensor:
        """Apply a transform to an image.

        Args:
            image: The PIL image to transform
            transform: The transform to apply, or None to use minimal transform

        Returns:
            Transformed image as a torch.Tensor

        Raises:
            DatasetTransformError: If the transform fails
        """
        if transform is None:
            transform = TransformFactory.get_transform(TransformType.MINIMAL)

        try:
            return transform(image)
        except Exception as e:
            logger.error(f"Error applying transform: {e}")
            raise DatasetTransformError(
                f"Failed to apply transform: {str(e)}",
                transform_name=transform.__class__.__name__,
            ) from e

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor back to a PIL image.

        Args:
            tensor: Image tensor to convert [C,H,W]

        Returns:
            PIL Image

        Raises:
            DatasetTransformError: If the conversion fails
        """
        try:
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected 3D tensor [C,H,W], got shape {tensor.shape}"
                )

            # Ensure values are in [0,1] range for F.to_pil_image
            tensor = tensor.clamp(0, 1)
            return F.to_pil_image(tensor)
        except Exception as e:
            logger.error(f"Error converting tensor to PIL image: {e}")
            raise DatasetTransformError(
                f"Failed to convert tensor to PIL image: {str(e)}"
            )


# Predefined transform instances for common use cases
IMAGENET_TRANSFORM = TransformFactory.get_transform(TransformType.IMAGENET)
ONION_TRANSFORM = TransformFactory.get_transform(TransformType.ONION)
MINIMAL_TRANSFORM = TransformFactory.get_transform(TransformType.MINIMAL)
