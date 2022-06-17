from typing import Any, Iterator, Tuple


class DataLoaderTransferWrapper:

    def __init__(self, dataloader, device) -> None:
        self.dataloader = dataloader
        self.device = device

    def __iter__(self) -> Iterator:
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Tuple[Any]:
        current = next(self.iterator)
        return (*[x.to(self.device) for x in current],)
