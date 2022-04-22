from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Tuple
import inspect
import torch


class BaseConnector(ABC):

    @abstractmethod
    def get_device(self) -> torch.device:
        return

    @abstractmethod
    def is_master(self) -> bool:
        return

    @abstractmethod
    def rendezvous(self, name) -> None:
        return

    @abstractmethod
    def get_samplers(self, train_dataset, val_dataset, shuffle_val=False) \
            -> Tuple[torch.utils.data.Sampler, torch.utils.data.Sampler]:
        return

    @abstractmethod
    def wrap_data_loader(self, data_loader, device) -> torch.utils.data.DataLoader:
        return

    @abstractmethod
    def optimizer_step(self, opt: torch.optim.Optimizer, **kwargs: Dict):
        return

    @abstractmethod
    def reduce_gradients(self, opt: torch.optim.Optimizer) -> None:
        return

    def _filter_optimizer_kwargs(self, opt: torch.optim.Optimizer, kwargs) -> Dict:
        available_kwargs = {}
        if hasattr(opt, "step_kwargs"):
            available_kwargs = opt.step_kwargs
        return {key: value for key, value in kwargs.items() if key in available_kwargs}

    @abstractmethod
    def all_avg(self, arr: List):
        return

    @abstractmethod
    def print(self, msg, flush):
        return

    @abstractmethod
    def save(self, obj: Dict, path: str):
        return

    @abstractmethod
    def run(self, function: Callable, args: List, nprocs: int):
        return

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def all_gather(self, arr: List) -> List:
        pass
