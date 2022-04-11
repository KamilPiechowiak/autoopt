from typing import Callable, List
import torch
from torch import nn

from .armijo_line_search import ArmijoLineSearch


class DistributedArmijoLineSearch(ArmijoLineSearch):
    def __init__(self, params: List[torch.Tensor], **kwargs):
        super(DistributedArmijoLineSearch, self).__init__(params, **kwargs)
        import torch_xla.core.xla_model as xm

    def _evaluate_model(self, model: nn.Module,
                        loss_func: Callable[[torch.Tensor, torch.Tensor], float],
                        X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_predicted = model(X)
        loss = loss_func(y_predicted, y)
        loss = xm.all_reduce(xm.REDUCE_SUM, loss) / xm.xrt_world_size()
        return loss

    def _backpropagate_gradients(self, loss: torch.Tensor):
        loss.backward()
        xm.reduce_gradients(self)
