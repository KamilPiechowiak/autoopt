from modulefinder import Module
from typing import Dict, List
from torch import optim
import inspect
import sys
import copy

from autoopt.optimizers.armijo_line_search import ArmijoLineSearch
from autoopt.optimizers.distributed_armijo_line_search import DistributedArmijoLineSearch


class OptimizersFactory:

    def __init__(self):
        self.optimizers = {
            **self._get_all_classes_in_a_module(sys.modules[__name__]),
            **self._get_all_classes_in_a_module(optim)
        }

    def _get_all_classes_in_a_module(self, module: Module) -> Dict[str, type]:
        res = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                res[name] = obj
        return res

    def get_optimizer(self, config: Dict, model_parameters: List) -> optim.Optimizer:
        config = copy.deepcopy(config)
        model_parameters = list(model_parameters)
        name = config["name"]
        del config["name"]
        if name not in self.optimizers:
            raise RuntimeError(f"{name} optimizer was not found. Check if the name is " +
                               "specified correctly.")
        clazz = self.optimizers[name]
        if "inner_optimizer" in config:
            config["inner_optimizer"] = self.get_optimizer(config["inner_optimizer"],
                                                           model_parameters)
        return clazz(model_parameters, **config)
