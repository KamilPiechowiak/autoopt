import copy
import math
from typing import Dict, List, Tuple
from attr import has
import numpy as np
from scipy import stats

import torch
from torch import optim
from autoopt.optimizers.extended_optimizer import ExtendedOptimizer


class ExpertsAdam(ExtendedOptimizer):
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

        self.probabilities = {}
        self.cumulative_losses = {}
        self.ranges = {}

        for hyperparameter, (h_min, h_max) in ranges.items():
            if initial_distribution == 'loguniform':
                self.probabilities[hyperparameter] = torch.ones(num_experts, device=device) \
                    / num_experts
            else:  # lognormal
                x = np.linspace(-3, 3, num_experts)
                self.probabilities[hyperparameter] = torch.tensor(stats.norm.pdf(x), device=device)
                self.probabilities[hyperparameter] = self.probabilities[hyperparameter] \
                    / torch.sum(self.probabilities[hyperparameter])
            self.cumulative_losses[hyperparameter] = torch.zeros(num_experts, device=device,
                                                                 dtype=torch.float)
            self.ranges[hyperparameter] = torch.exp(
                torch.linspace(np.log(h_min), np.log(h_max), num_experts, device=device)
            )

    @torch.no_grad()
    def step(self) -> None:

        for hyperparameter in self.probabilities.keys():
            if not hasattr(self, 'previous_stats'):
                continue
            self.cumulative_losses[hyperparameter] *= self.cumulative_loss_decay
            for group_opt, group_prev in zip(self.inner_optimizer.param_groups,
                                             self.previous_stats):
                hyperparams = {
                    'lr': group_opt['lr'],
                    'beta1': group_opt['betas'][0],
                    'beta2': group_opt['betas'][1],
                    'eps': group_opt['eps'],
                    'weight_decay': group_opt['weight_decay']
                }
                hyperparams[hyperparameter] = self.ranges[hyperparameter].reshape(-1, 1)
                if hyperparameter in ['beta1', 'beta2']:  # distribution on 1-\beta rather than \beta
                    hyperparams[hyperparameter] = 1 - hyperparams[hyperparameter]
                for p_opt, p_prev in zip(group_opt['params'], group_prev['params']):
                    losses_update = self._get_losses(p_opt, p_prev, hyperparams)
                    self.cumulative_losses[hyperparameter] -= losses_update

        # if hasattr(self, 'direction'):
        #     print(self._gradient_vector_dot_product(self.direction, computed_direction, is_first_grad=False))
        #     print(self._gradient_vector_dot_product(self.direction, self.direction, is_first_grad=False))
        #     print(self._gradient_vector_dot_product(computed_direction, computed_direction, is_first_grad=False))

        expected_hyperparameters = self._get_expected_hyperparamters()
        self._update_state(expected_hyperparameters)

        self.inner_optimizer.step()

    def _get_losses(self, p_opt: torch.Tensor, p_prev: Dict[str, torch.Tensor],
                    hyperparams: Dict[str, torch.Tensor]) -> None:
        bias_correction1 = 1 - hyperparams['beta1'] ** p_prev['step']
        bias_correction2 = 1 - hyperparams['beta2'] ** p_prev['step']
        prev_grad = p_prev['grad'] + p_prev['param']*hyperparams['weight_decay']
        numerator = hyperparams['beta1'] * p_prev['exp_avg'] \
            + (1-hyperparams['beta1']) * prev_grad
        denominator = hyperparams['beta2'] * p_prev['exp_avg_sq'] \
            + (1-hyperparams['beta2']) * prev_grad**2
        denominator /= bias_correction2
        denominator = denominator.sqrt() + hyperparams['eps']
        direction = hyperparams['lr'] * numerator / bias_correction1 / denominator
        # print(direction.shape)
        # print(direction[:10])
        losses_update = torch.sum(direction * p_opt.grad.flatten(), dim=-1)
        return losses_update

    def _get_expected_hyperparamters(self) -> Dict[str, float]:
        expected_hyperparameters = {}
        for hyperparameter in self.probabilities.keys():
            probabilities = torch.exp(-self.alpha * (
                self.cumulative_losses[hyperparameter]
                - self.cumulative_losses[hyperparameter].min()
            ))
            probabilities *= self.probabilities[hyperparameter]
            probabilities /= torch.sum(probabilities)
            expected_hyperparameters[hyperparameter] = torch.sum(
                probabilities*self.ranges[hyperparameter]).cpu().item()
            # print(self.cumulative_losses)
            # print(hyperparameter, expected_hyperparameters[hyperparameter])
            self.state[hyperparameter] = expected_hyperparameters[hyperparameter]
        return expected_hyperparameters

    def _update_state(self, expected_hyperparameters: Dict[str, float]) -> None:
        self.previous_stats = []
        for group in self.inner_optimizer.param_groups:
            if 'lr' in self.ranges.keys():
                group['lr'] = expected_hyperparameters['lr']
            if 'beta1' in self.ranges.keys() and 'beta2' in self.ranges.keys():
                group['betas'] = (expected_hyperparameters['beta1'], expected_hyperparameters['beta2'])
            if 'eps' in self.ranges.keys():
                group['eps'] = expected_hyperparameters['eps']
            if 'weight_decay' in self.ranges.keys():
                group['weight_decay'] = expected_hyperparameters['weight_decay']
            previous_stats = {
                'params': []
            }
            for p in group['params']:
                state = self.inner_optimizer.state[p]
                if len(state) == 0:
                    state = {
                        'step': torch.ones((1,), device=p.device),
                        'exp_avg': torch.zeros_like(p, memory_format=torch.preserve_format),
                        'exp_avg_sq': torch.zeros_like(p, memory_format=torch.preserve_format),
                    }
                previous_stats['params'].append({
                    'step': copy.deepcopy(state['step'])+1,
                    'grad': copy.deepcopy(p.grad).flatten(),
                    'param': copy.deepcopy(p).flatten(),
                    'exp_avg': copy.deepcopy(state['exp_avg']).flatten(),
                    'exp_avg_sq': copy.deepcopy(state['exp_avg_sq']).flatten()
                })
            self.previous_stats.append(previous_stats)