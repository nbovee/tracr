import abc
from typing import Any, Dict, Type


class Partitioner(abc.ABC):
    """
    Abstract base class for model partitioning strategies.

    This class serves as a factory for creating different partitioning methods.
    Custom partitioners can be implemented by subclassing Partitioner and
    specifying a unique _TYPE attribute.

    Attributes:
        _TYPE (str): A unique identifier for the partitioner type.
        subclasses (Dict[str, Type['Partitioner']]): A dictionary of registered partitioner subclasses.
    """

    _TYPE: str = "base"
    subclasses: Dict[str, Type["Partitioner"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register subclasses of Partitioner.

        Raises:
            ValueError: If the _TYPE attribute is already registered.
        """
        super().__init_subclass__(**kwargs)
        if cls._TYPE in cls.subclasses:
            raise ValueError(f"_TYPE alias '{cls._TYPE}' is already reserved.")
        cls.subclasses[cls._TYPE] = cls

    @classmethod
    def create(cls, class_type: str, *args, **kwargs) -> "Partitioner":
        """
        Create an instance of a specific partitioner subclass.

        Args:
            class_type (str): The _TYPE of the partitioner to create.
            *args: Positional arguments to pass to the partitioner constructor.
            **kwargs: Keyword arguments to pass to the partitioner constructor.

        Returns:
            Partitioner: An instance of the specified partitioner subclass.

        Raises:
            ValueError: If the specified class_type is not registered.
        """
        if class_type not in cls.subclasses:
            raise ValueError(f"Unknown partitioner type: {class_type}")
        return cls.subclasses[class_type](*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the partitioning strategy.

        This method should be implemented by subclasses to define the specific
        partitioning behavior.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
