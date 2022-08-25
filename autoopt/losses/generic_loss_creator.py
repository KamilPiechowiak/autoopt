from typing import Callable
import torch
from torch import nn


class GenericLossCreator:

    def __init__(self) -> None:
        pass

    def get_loss_function_on_minibatch(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) \
            -> Callable:
        raise NotImplementedError
