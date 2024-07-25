from typing import Any, Generator
from itertools import cycle
from .partitioner import Partitioner


class CyclePartitioner(Partitioner):
    """
    A partitioner that cycles through a range of breakpoints.

    Attributes:
        _TYPE (str): The type identifier for this partitioner.
        breakpoints (int): The number of breakpoints to cycle through.
        repeats (int): The number of times to yield each value before moving to the next.
        counter (cycle): An iterator that cycles through the breakpoint range.
    """

    _TYPE: str = "cycle"

    def __init__(
        self, num_breakpoints: int, clip_min_max: bool = True, repeats: int = 1
    ) -> None:
        """
        Initialize the CyclePartitioner.

        Args:
            num_breakpoints (int): The number of breakpoints to cycle through.
            clip_min_max (bool): If True, exclude 0 and num_breakpoints from the cycle.
            repeats (int): The number of times to yield each value before moving to the next.
        """
        super().__init__()
        self.breakpoints = num_breakpoints
        self.repeats = max(repeats, 1)
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints + 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Generator[int, None, None]:
        """
        Generate breakpoint values in a cyclic manner.

        Yields:
            int: The next breakpoint value in the cycle.
        """
        value = next(self.counter)
        for _ in range(self.repeats):
            yield value
