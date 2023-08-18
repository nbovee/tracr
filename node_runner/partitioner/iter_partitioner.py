from .partitioner import Partitioner
from itertools import cycle
from typing import Any


class CyclePartitioner(Partitioner):
    _TYPE = "cycle"

    def __init__(self, num_breakpoints, clip_min_max=True, repeats = 1) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.repeats = repeats if repeats > 0 else 1
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints + 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        res = next(self.counter)
        for i in range(self.repeats):
            yield res
