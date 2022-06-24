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
    def step(self, simulate: bool = False) -> None:
        if self.params_current is None:
            self.params_current = copy.deepcopy(self.param_groups)
        if simulate:
            step = self.get_state_key('step')
            exp_avg = self.get_state_key('exp_avg')
            exp_avg_sq = self.get_state_key('exp_avg_sq')
        self.inner_optimizer.step()
        if simulate:
            self.set_state_key('step', step)
            self.set_state_key('exp_avg', exp_avg)
            self.set_state_key('exp_avg_sq', exp_avg_sq)
        params_next = copy.deepcopy(self.inner_optimizer.param_groups)
        direction_next = self._get_direction(self.params_current, params_next)
        if self.direction is not None:
            _, neg_cos = self._gradient_vector_dot_product(
                self.direction, direction_next, is_first_grad=False)
            self.history.append(-neg_cos.item())
        self.params_current = params_next
        self.direction = direction_next

    def get_state_key(self, key: str) -> List[torch.Tensor]:
        data = []
        for group in self.inner_optimizer.param_groups:
            for p in group['params']:
                if key in self.inner_optimizer.state[p]:
                    data.append(self.inner_optimizer.state[p][key])
                else:
                    data.append(None)
        return copy.deepcopy(data)

    def set_state_key(self, key: str, data: List[torch.Tensor]):
        i = 0
        for group in self.inner_optimizer.param_groups:
            for p in group['params']:
                if data[i] is None:
                    if isinstance(self.inner_optimizer.state[p][key], int):
                        self.inner_optimizer.state[p][key] = 0
                    else:
                        self.inner_optimizer.state[p][key][:] = 0
                else:
                    self.inner_optimizer.state[p][key] = data[i]
                i += 1
