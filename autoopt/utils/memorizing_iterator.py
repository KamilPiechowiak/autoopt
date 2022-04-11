from typing import Any, Iterator


class MemorizingIterator:

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator:
        self.iterator = iter(self.dataset)
        return self

    def __next__(self) -> Any:
        self._current = next(self.iterator)
        return self._current

    def current(self) -> Any:
        return self._current
