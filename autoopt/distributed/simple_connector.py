from typing import Callable, List, Dict, Tuple

import torch

from .base_connector import BaseConnector


class SimpleConnector(BaseConnector):

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def is_master(self) -> bool:
        return True

    def rendezvous(self, name) -> None:
        pass

    def get_samplers(self, train_dataset, val_dataset, shuffle_val=False) \
            -> Tuple[torch.utils.data.Sampler, torch.utils.data.Sampler]:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset
        )
        val_sampler = torch.utils.data.SequentialSampler(
            val_dataset
        )
        return train_sampler, val_sampler

    def wrap_data_loader(self, data_loader, device) -> torch.utils.data.DataLoader:
        return data_loader

    def optimizer_step(self, opt: torch.optim.Optimizer, **kwargs: Dict):
        opt.step(**self._filter_optimizer_kwargs(opt, kwargs))

    def reduce_gradients(self, opt: torch.optim.Optimizer) -> None:
        return

    def all_avg(self, arr: List):
        pass

    def print(self, msg, flush=False):
        print(msg, flush=flush)

    def save(self, obj: Dict, path: str):
        torch.save(obj, path)

    def run(self, function: Callable, args: List, nprocs: int):
        function(0, *args)

    def step(self):
        pass

    def all_gather(self, arr: List) -> List:
        return arr
