import random
from typing import Dict

import sys
import inspect
from torch import nn
import copy
import timm
import torch
import numpy as np
from .sls import *
from .small import *
from .lstm import *


class ModelsFactory:
    def __init__(self):
        self.available_models = {}
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            self.available_models[name] = obj

    def get_model(self, config: Dict) -> nn.Module:
        config = copy.deepcopy(config)
        name = config["name"]
        del config["name"]
        if config.get("seed"):
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            del config["seed"]
        if name in self.available_models.keys():
            return self.available_models[name](**config)
        return timm.create_model(name, **config)
