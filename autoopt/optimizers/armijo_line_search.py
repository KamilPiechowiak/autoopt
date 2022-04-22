from decimal import ExtendedContext
from typing import Any, Callable, Dict, List
import inspect
import logging

import torch
from torch import optim, nn
import copy
import time
from autoopt.optimizers.extended_optimizer import ExtendedOptimizer

from autoopt.utils.memorizing_iterator import MemorizingIterator
from autoopt.utils.profile import profile


class ArmijoLineSearch(ExtendedOptimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 max_lr: float = 100.0, beta: float = 0.9, c: float = 0.1,
                 reset_strategy: str = 'keep', search_strategy: str = 'armijo',
                 batch_strategy: str = 'single', gamma: float = 2.0, max_iterations: int = 100,
                 min_cosine: float = 0.01, min_lr: float = 1e-5, goldstein_c: float = None) -> None:
        # self.step_kwargs = set(inspect.getfullargspec(self.step).args)
        self.step_kwargs = {"model", "loss_func", "data_iterator", "loss_value"}
        super(ArmijoLineSearch, self).__init__(params, {})
        assert reset_strategy in ['max', 'keep', 'increase']
        assert search_strategy in ['armijo', 'goldstein']
        assert batch_strategy in ['single', 'double']
        if search_strategy == 'goldstein' and c >= 0.5:
            raise ValueError("If search strategy is goldstein, c has to be smaller than 0.5")
        self.state['lr'] = 1
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.beta = beta
        self.c = c
        if goldstein_c is not None:
            assert goldstein_c > c
            self.goldstein_c = goldstein_c
        else:
            self.goldstein_c = 1-self.c
        self.reset_strategy = reset_strategy
        self.search_strategy = search_strategy
        self.batch_strategy = batch_strategy
        self.inner_optimizer = inner_optimizer
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.min_cosine = min_cosine
        self.total_steps = 0
        self.cosine_breaks = 0

    # @profile
    def step(self, model: nn.Module, loss_func: Callable[[torch.Tensor, torch.Tensor], float],
             data_iterator: MemorizingIterator, loss_value: torch.Tensor) -> None:
        with torch.no_grad():
            self.total_steps += 1
            params_current = copy.deepcopy(self.param_groups)
            self.inner_optimizer.step()
            params_next = copy.deepcopy(self.inner_optimizer.param_groups)
            direction = self._get_direction(params_current, params_next)

            if self.batch_strategy == 'single':
                X, y = data_iterator.current()
            else:
                self._assign_new_params(params_current, direction, lr=0)
                self.zero_grad()
                with torch.enable_grad():
                    X, y = next(data_iterator)
                    loss_value = self._evaluate_model(model, loss_func, X, y, average=False)
                    self._backpropagate_gradients(loss_value)
            loss_value = self._reduce_average(loss_value)
            dot_product, cosine = self._gradient_vector_dot_product(self.param_groups, direction)
            # logging.debug(f"Cosine: {cosine}")
            lr = self._reset_lr(self.state['lr'])
            self._mark_step()
            # tmp = []
            # lr = self.max_lr  # FIXME
            for i in range(self.max_iterations):
                self._assign_new_params(params_current, direction, lr=lr)
                new_loss_value = self._evaluate_model(model, loss_func, X, y)

                if cosine < self.min_cosine:  # FIXME
                    self.cosine_breaks += 1
                    i = -1
                    # logging.debug(f"Cosine too small. Break {self.cosine_breaks/self.total_steps}")
                    break
                upper_bound_condition = (
                    new_loss_value <= loss_value + self.c*lr*dot_product
                )
                lower_bound_condition = (
                    new_loss_value >= loss_value + self.goldstein_c*lr*dot_product
                )
                # logging.debug(f"{upper_bound_condition} {lower_bound_condition}")
                if self.search_strategy == 'armijo':
                    if upper_bound_condition:
                        break
                    else:
                        lr *= self.beta
                else:  # goldstein
                    if upper_bound_condition and lower_bound_condition:
                        break
                    elif upper_bound_condition:
                        lr = lr / self.beta
                    elif lower_bound_condition:
                        lr = lr * self.beta
                    else:
                        raise RuntimeError("Something very weird happened to loss value:" +
                                           f"{new_loss_value}")
                if lr < self.min_lr:
                    lr = self.min_lr
                    break
                if lr > self.max_lr:
                    lr = self.max_lr
                    break
                self._mark_step()
                # print(f"{i} {lr} {new_loss_value.item()} {loss_value.item()} " +
                #       f"{(loss_value+self.c*lr*dot_product).item()} " +
                #       f"{(loss_value + self.goldstein_c*lr*dot_product).item()}", flush=True)
                # tmp.append([lr, new_loss_value.item(), (loss_value+self.c*lr*dot_product).item(),
                #            (loss_value + self.goldstein_c*lr*dot_product).item()])
                # lr *= self.beta
            self.state['num_iterations'] = i
            self.state['lr'] = lr
            self.state['cosine'] = cosine.cpu().item()
            # import numpy as np
            # import matplotlib.pyplot as plt
            # tmp = np.array(tmp)
            # plt.ylim((1.5, loss_value.item()))
            # plt.scatter(tmp[:,0], tmp[:,1])
            # plt.scatter(tmp[:,0], tmp[:,2])
            # plt.scatter(tmp[:,0], tmp[:,3])
            # plt.show()
            # exit(0)

    def _reset_lr(self, lr: float) -> float:
        if self.reset_strategy == 'max':
            return self.max_lr
        elif self.reset_strategy == 'keep':
            return lr
        else:
            return min(lr * self.gamma, self.max_lr)
