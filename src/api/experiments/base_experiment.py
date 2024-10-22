from typing import Any, Dict

class BaseExperiment:
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port

    def initialize_model(self) -> Any:
        raise NotImplementedError("initialize_model method must be implemented")

    def process_data(self, data: Any) -> Any:
        raise NotImplementedError("process_data method must be implemented")

    def setup_socket(self):
        raise NotImplementedError("setup_socket method must be implemented")

    def run(self):
        raise NotImplementedError("run method must be implemented")

    def receive_data(self, conn: Any) -> Any:
        raise NotImplementedError("receive_data method must be implemented")

    def send_result(self, conn: Any, result: Any):
        raise NotImplementedError("send_result method must be implemented")
