from typing import Callable, List
from warnings import WarningMessage

import numpy as np
from scipy import stats
import torch
from torch import optim, nn
import copy
import warnings
from autoopt.optimizers.extended_optimizer import ExtendedOptimizer
from autoopt.utils.memorizing_iterator import MemorizingIterator


class ExpertsLinearized(ExtendedOptimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 alpha: float, lr_min: float, lr_max: float,
                 num_experts: int, is_stationary: bool, initial_distritbution: str = 'lognormal',
                 expecation_on_logarithms: bool = False, cumulative_loss_decay: float = 1) -> None:
        super(ExpertsLinearized, self).__init__(params, {})
        assert initial_distritbution in {'loguniform', 'lognormal'}
        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        device = self.param_groups[0]['params'][0].device
        self.learning_rates = torch.exp(
            torch.linspace(np.log(lr_min), np.log(lr_max), num_experts, device=device)
        )
        if initial_distritbution == 'loguniform':
            self.probabilities = torch.ones(num_experts, device=device)/num_experts
        else:
            if num_experts < 100:
                warnings.warn("Normal distribution will be approximated numerically." +
                              " Consider increasing number of experts")
            x = np.linspace(-3, 3, num_experts)
            self.probabilities = torch.tensor(stats.norm.pdf(x), device=device)
            self.probabilities = self.probabilities/torch.sum(self.probabilities)

        self.cumulative_losses = torch.zeros(num_experts, device=device)/num_experts
        self.previous_direction = None
        self.total_steps = 0
        self.cumulative_loss_decay = cumulative_loss_decay
        self.expecation_on_logarithms = expecation_on_logarithms
        self.is_stationary = is_stationary
        if not self.is_stationary:
            self.initial_params = copy.deepcopy(self.param_groups)

    @torch.no_grad()
    def step(self) -> None:
        self.total_steps += 1
        params_current = copy.deepcopy(self.param_groups)
        self.inner_optimizer.step()
        params_next = copy.deepcopy(self.inner_optimizer.param_groups)
        direction = self._get_direction(params_current, params_next)

        # print(self._gradient_vector_dot_product(self.param_groups, direction))

        if self.previous_direction is not None:
            neg_dot_product, _ = self._gradient_vector_dot_product(self.param_groups,
                                                                   self.previous_direction)
            self.cumulative_losses *= self.cumulative_loss_decay
            self.cumulative_losses += neg_dot_product*self.learning_rates

        probabilities = self.probabilities * \
            torch.exp(-self.alpha * (self.cumulative_losses-self.cumulative_losses.min()))
        probabilities /= torch.sum(probabilities)
        if self.expecation_on_logarithms:
            expected_lr = torch.exp(torch.sum(probabilities*torch.log(self.learning_rates)))
        else:
            expected_lr = torch.sum(probabilities*self.learning_rates)
        # self.state['prob'] = probabilities.cpu().numpy()
        self.state['lr'] = expected_lr.item()
        if self.is_stationary:
            self.previous_direction = direction
            self._assign_new_params(params_current, direction, lr=expected_lr)
        else:
            if self.previous_direction is None:
                self.previous_direction = direction
            else:
                self._add_direction_to_vector(self.previous_direction, self.previous_direction,
                                              direction, lr=1)
            self._assign_new_params(self.initial_params, self.previous_direction, lr=expected_lr)
        # import matplotlib.pyplot as plt
        # plt.plot(torch.log(self.learning_rates).numpy(), probabilities.numpy())
        # plt.show()
        # print(expected_lr, self.cumulative_losses.max(),
        #       self.cumulative_losses.median(), self.cumulative_losses.min())
