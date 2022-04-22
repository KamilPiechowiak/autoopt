from typing import Callable, List
import torch
from torch import nn
import time

from .armijo_line_search import ArmijoLineSearch


class DistributedArmijoLineSearch(ArmijoLineSearch):
    def __init__(self, params: List[torch.Tensor], **kwargs):
        super(DistributedArmijoLineSearch, self).__init__(params, **kwargs)
        import torch_xla.core.xla_model as xm
        self.xm = xm

    def _evaluate_model(self, model: nn.Module,
                        loss_func: Callable[[torch.Tensor, torch.Tensor], float],
                        X: torch.Tensor, y: torch.Tensor, average: bool = True) -> torch.Tensor:
        y_predicted = model(X)
        loss = loss_func(y_predicted, y)
        if average:
            xm = self.xm
            loss = xm.all_reduce(xm.REDUCE_SUM, loss, scale=1.0/xm.xrt_world_size())
        return loss

    def _backpropagate_gradients(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.xm.reduce_gradients(self)

    def _reduce_average(self, value: torch.Tensor) -> torch.Tensor:
        xm = self.xm
        value = xm.all_reduce(xm.REDUCE_SUM, value, scale=1.0/xm.xrt_world_size())
        return value

    def _mark_step(self) -> None:
        self.xm.mark_step()
