from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExperiment(ABC):
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
