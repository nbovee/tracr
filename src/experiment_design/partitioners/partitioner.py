import abc
from typing import Any


class Partitioner:
    """
    Factory class for the method of determining split location in a model. Custom partitioners
    can be written in their own module and dropped into this directory for automatic import.
    """

    _TYPE: str = "base"

    subclasses = {}

    # @classmethod implicit
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._TYPE in cls.subclasses:
            raise ValueError("_TYPE alias already reserved.")
        cls.subclasses[cls._TYPE] = cls

    @classmethod
    def create(cls, class_type, *args, **kwargs):
        if class_type not in cls.subclasses:
            raise ValueError(
                "Bad or unknown type {}. Does the subclass specify _TYPE ?".format(
                    class_type
                )
            )
        return cls.subclasses[class_type](*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
