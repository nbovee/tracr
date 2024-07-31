from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class ModelInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        config_path: Optional[str] = None,
        master_dict: Any = None,
        flush_buffer_size: int = 100,
    ):
        pass

    @abstractmethod
    def forward(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Union[int, float] = float("inf"),
        log: bool = True,
    ) -> Any:
        pass

    @abstractmethod
    def update_master_dict(self) -> None:
        pass

    @abstractmethod
    def parse_input(self, _input: Any) -> Any:
        pass

    @abstractmethod
    def warmup(self, iterations: int = 50, force: bool = False) -> None:
        pass


class ModelFactoryInterface(ABC):
    @abstractmethod
    def create_model(
        self,
        config_path: Optional[str] = None,
        master_dict: Any = None,
        flush_buffer_size: int = 100,
    ) -> ModelInterface:
        pass
