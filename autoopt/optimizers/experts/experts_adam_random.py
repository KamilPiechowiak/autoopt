import copy
import math
from typing import Dict, List, Tuple
from attr import has
import numpy as np
from scipy import stats

import torch
from torch import optim

from autoopt.optimizers.experts.experts_adam import ExpertsAdam


class ExpertsAdamRandom(ExpertsAdam):
    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 alpha: float, ranges: Dict[str, Tuple[float, float]],
                 cumulative_loss_decay: float = 1,
                 num_experts: int = None,
                 initial_distribution: str = 'loguniform') -> None:

        super(ExpertsAdam, self).__init__(params, {})
        assert initial_distribution in {'loguniform', 'lognormal'}
        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        device = self.param_groups[0]['params'][0].device
        self.cumulative_loss_decay = cumulative_loss_decay

        self.cumulative_losses = torch.zeros(num_experts, device=device, dtype=torch.float)
        self.ranges = {}

        for hyperparameter, (h_min, h_max) in ranges.items():
            if initial_distribution == 'loguniform':
                samples = torch.rand(num_experts, device=device)
            else:  # lognormal
                samples = torch.randn(num_experts, device=device) / 3.0
            samples = np.log(h_min) * (1-samples) + np.log(h_max) * samples
            samples = torch.exp(samples)
            if hyperparameter in ['beta1', 'beta2']:
                samples = 1-samples
            self.ranges[hyperparameter] = samples
            # print(hyperparameter, samples)

    @torch.no_grad()
    def step(self) -> None:

        if hasattr(self, 'previous_stats'):
            self.cumulative_losses *= self.cumulative_loss_decay
            for group_opt, group_prev in zip(self.inner_optimizer.param_groups,
                                             self.previous_stats):
                hyperparams = {
                    'lr': group_opt['lr'],
                    'beta1': group_opt['betas'][0],
                    'beta2': group_opt['betas'][1],
                    'eps': group_opt['eps'],
                    'weight_decay': group_opt['weight_decay']
                }
                for hyperparameter in self.ranges.keys():
                    hyperparams[hyperparameter] = self.ranges[hyperparameter].reshape(-1, 1)
                for p_opt, p_prev in zip(group_opt['params'], group_prev['params']):
                    losses_update = self._get_losses(p_opt, p_prev, hyperparams)
                    self.cumulative_losses -= losses_update

        expected_hyperparameters = self._get_expected_hyperparamters()
        self._update_state(expected_hyperparameters)

        self.inner_optimizer.step()

    def _get_expected_hyperparamters(self) -> Dict[str, float]:
        expected_hyperparameters = {}
        probabilities = torch.exp(-self.alpha * (
            self.cumulative_losses - self.cumulative_losses.min()
        ))
        probabilities /= torch.sum(probabilities)
        for hyperparameter in self.ranges.keys():
            expected_hyperparameters[hyperparameter] = torch.sum(
                probabilities*self.ranges[hyperparameter]).cpu().item()
            # print(self.cumulative_losses)
            # print(hyperparameter, expected_hyperparameters[hyperparameter])
            self.state[hyperparameter] = expected_hyperparameters[hyperparameter]
        return expected_hyperparameters
