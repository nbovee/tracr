# src/experiment_design/partitioners/iter_partitioner.py

import logging
from typing import Any, Generator
from itertools import cycle
from .partitioner import Partitioner

logger = logging.getLogger(__name__)


class CyclePartitioner(Partitioner):
    _TYPE: str = "cycle"

    def __init__(
        self, num_breakpoints: int, clip_min_max: bool = True, repeats: int = 1
    ) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.repeats = max(repeats, 1)
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints + 1))
        logger.info(
            f"Initialized CyclePartitioner with {num_breakpoints} breakpoints, clip_min_max={clip_min_max}, repeats={repeats}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Generator[int, None, None]:
        logger.debug("CyclePartitioner called")
        value = next(self.counter)
        for i in range(self.repeats):
            logger.debug(f"Yielding value {value} (repeat {i + 1}/{self.repeats})")
            yield value
