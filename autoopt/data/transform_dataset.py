from typing import Callable, Tuple
import torch


class TransformDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            transform: Callable
    ) -> None:
        super(TransformDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.dataset[i]
        return self.transform(img), target
