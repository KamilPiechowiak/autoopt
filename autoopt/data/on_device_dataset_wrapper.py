from typing import Any
import torch
from torch.utils import data


class OnDeviceDatasetWrapper(data.Dataset):

    def __init__(self, dataset: data.Dataset, device: torch.device) -> None:
        self.device = device
        self.data = [
            (*[torch.tensor(x_i, device=device) for x_i in x],) for x in dataset
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Any:
        return self.data[i]
