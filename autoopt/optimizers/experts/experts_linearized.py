from typing import Callable, List
from warnings import WarningMessage

import numpy as np
from scipy import stats, special
import torch
from torch import optim, nn
import copy
import warnings
from autoopt.optimizers.extended_optimizer import ExtendedOptimizer
from autoopt.utils.memorizing_iterator import MemorizingIterator


class ExpertsLinearized(ExtendedOptimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 alpha: float, lr_min: float, lr_max: float,
                 is_stationary: bool, num_experts: int = None,
                 initial_distritbution: str = 'lognormal',
                 expecation_on_logarithms: bool = False,
                 cumulative_loss_decay: float = 1,
                 normalize: str = None,
                 fixed_share_alpha: float = 0.0) -> None:
        super(ExpertsLinearized, self).__init__(params, {})
        assert initial_distritbution in {'loguniform', 'lognormal', 'exponential'}
        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        device = self.param_groups[0]['params'][0].device

        self.initial_distritbution = initial_distritbution
        self.lr_min = lr_min
        self.lr_max = lr_max

        if initial_distritbution == 'lognormal':
            self.learning_rates = torch.exp(
                torch.linspace(np.log(lr_min), np.log(lr_max), num_experts, device=device)
            )
            if num_experts < 100:
                warnings.warn("Normal distribution will be approximated numerically." +
                              " Consider increasing the number of experts")
            x = np.linspace(-3, 3, num_experts)
            self.probabilities = torch.tensor(stats.norm.pdf(x), device=device)
            self.probabilities = self.probabilities/torch.sum(self.probabilities)
        elif initial_distritbution == 'exponential':
            self.exponential_lambda = 1 / np.exp((np.log(lr_min) + np.log(lr_max))/2.0)

        self.cumulative_loss = torch.tensor(0, device=device, dtype=torch.float32)
        self.previous_direction = None
        self.total_steps = 0
        self.cumulative_loss_decay = cumulative_loss_decay
        self.expecation_on_logarithms = expecation_on_logarithms
        self.is_stationary = is_stationary
        if not self.is_stationary:
            self.initial_params = copy.deepcopy(self.param_groups)
        self.normalize = normalize
        self.fixed_share_alpha = fixed_share_alpha

    @torch.no_grad()
    def step(self) -> None:
        self.total_steps += 1
        params_current = copy.deepcopy(self.param_groups)
        self.inner_optimizer.step()
        params_next = copy.deepcopy(self.inner_optimizer.param_groups)
        direction = self._get_direction(params_current, params_next)

        if self.normalize:
            self._normalize_direction(direction)

        # print(self._gradient_vector_dot_product(self.param_groups, direction))

        if self.previous_direction is not None:
            neg_dot_product, _ = self._gradient_vector_dot_product(self.param_groups,
                                                                   self.previous_direction)
            self.cumulative_loss *= self.cumulative_loss_decay
            self.cumulative_loss += neg_dot_product

        if self.initial_distritbution == 'lognormal':
            cumulative_losses = self.learning_rates*self.cumulative_loss
            probabilities = self.probabilities * \
                torch.exp(-self.alpha * (cumulative_losses-cumulative_losses.min()))
            probabilities /= torch.sum(probabilities)
            n = probabilities.shape[0]
            probabilities = (1-self.fixed_share_alpha)*probabilities + self.fixed_share_alpha/n
            expected_lr = torch.sum(probabilities*self.learning_rates).cpu().item()
        elif self.initial_distritbution == 'loguniform':
            ln_r_t = (-self.alpha * self.cumulative_loss).cpu().item()
            if np.abs(ln_r_t) < 1e-6:
                expected_lr = (self.lr_max-self.lr_min)/np.log(self.lr_max/self.lr_min)
            else:
                expected_lr = (np.exp(ln_r_t*self.lr_max) - np.exp(ln_r_t*self.lr_min)) / \
                    ln_r_t / (special.expi(ln_r_t*self.lr_max) - special.expi(ln_r_t*self.lr_min))
        else:  # exponential
            ln_r_t = -self.alpha * self.cumulative_loss
            expected_lr = 1.0 / (self.exponential_lambda - ln_r_t)
            expected_lr = expected_lr.cpu().item()
        # print(expected_lr)

        self.state['lr'] = expected_lr
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
