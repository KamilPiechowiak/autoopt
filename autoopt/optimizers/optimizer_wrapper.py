import copy
from lib2to3.pgen2.token import OP
from typing import List
from numpy import inner
import torch
from torch import optim, nn

from autoopt.optimizers.extended_optimizer import ExtendedOptimizer


class OptimizerWrapper(ExtendedOptimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer):
        super(OptimizerWrapper, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.params_current = None
        self.direction = None
        self.history = []

    @torch.no_grad()
    def step(self) -> None:
        if self.params_current is None:
            self.params_current = copy.deepcopy(self.param_groups)
        self.inner_optimizer.step()
        params_next = copy.deepcopy(self.inner_optimizer.param_groups)
        direction_next = self._get_direction(self.params_current, params_next)
        if self.direction is not None:
            _, neg_cos = self._gradient_vector_dot_product(
                self.direction, direction_next, is_first_grad=False)
            self.history.append(-neg_cos.item())
        self.params_current = params_next
        self.direction = direction_next
