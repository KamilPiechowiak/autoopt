from modulefinder import Module
from typing import Dict, List
from torch import optim
import inspect
import copy


class SchedulersFactory:

    def __init__(self):
        self.schedulers = self._get_all_classes_in_a_module(optim.lr_scheduler)

    def _get_all_classes_in_a_module(self, module: Module) -> Dict[str, type]:
        res = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                res[name] = obj
        return res

    def get_scheduler(self, config: Dict, optimizer: optim.Optimizer) \
            -> optim.lr_scheduler._LRScheduler:
        config = copy.deepcopy(config)
        name = config["name"]
        del config["name"]
        if name not in self.schedulers:
            raise RuntimeError(f"{name} scheduler was not found. Check if the name is " +
                               "specified correctly.")
        clazz = self.schedulers[name]
        return clazz(optimizer, **config)
