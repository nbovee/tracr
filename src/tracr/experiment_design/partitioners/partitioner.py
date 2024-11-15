# src/experiment_design/partitioners/partitioner.py

import abc
import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class Partitioner(abc.ABC):
    _TYPE: str = "base"
    subclasses: Dict[str, Type["Partitioner"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._TYPE in cls.subclasses:
            raise ValueError(f"_TYPE alias '{cls._TYPE}' is already reserved.")
        cls.subclasses[cls._TYPE] = cls
        logger.info(
            f"Registered partitioner subclass: {cls.__name__} with _TYPE: {cls._TYPE}"
        )

    @classmethod
    def create(cls, class_type: str, *args, **kwargs) -> "Partitioner":
        if class_type not in cls.subclasses:
            raise ValueError(f"Unknown partitioner type: {class_type}")
        logger.info(f"Creating partitioner of type: {class_type}")
        return cls.subclasses[class_type](*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Calling partitioner of type: {self._TYPE}")
        raise NotImplementedError(
            "Partitioner __call__ method must be implemented by subclasses"
        )
