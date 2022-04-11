from typing import Any, Callable, Dict, List
import inspect
import logging

import torch
from torch import optim, nn
import copy

from autoopt.utils.memorizing_iterator import MemorizingIterator


class ArmijoLineSearch(optim.Optimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 max_lr: float = 1.0, beta: float = 0.9, c: float = 0.1,
                 reset_strategy: str = 'keep', search_strategy: str = 'armijo',
                 batch_strategy: str = 'single', gamma: float = 2.0, max_iterations: int = 100,
                 min_cosine: float = 0.01) -> None:
        # self.step_kwargs = set(inspect.getfullargspec(self.step).args)
        self.step_kwargs = {"model", "loss_func", "data_iterator", "loss_value"}
        print(self.step_kwargs)
        super(ArmijoLineSearch, self).__init__(params, {})
        assert reset_strategy in ['max', 'keep', 'increase']
        assert search_strategy in ['armijo', 'goldstein']
        assert batch_strategy in ['single', 'double']
        if search_strategy == 'goldstein' and c >= 0.5:
            raise ValueError("If search strategy is goldstein, c has to be smaller than 0.5")
        self.state['lr'] = max_lr
        self.max_lr = max_lr
        self.beta = beta
        self.c = c
        self.reset_strategy = reset_strategy
        self.search_strategy = search_strategy
        self.batch_strategy = batch_strategy
        self.inner_optimizer = inner_optimizer
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.min_cosine = min_cosine
        self.total_steps = 0
        self.cosine_breaks = 0

    def state_dict(self) -> dict:
        self.state['inner_optimizer'] = self.inner_optimizer.state_dict()
        return super(ArmijoLineSearch, self).state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        super(ArmijoLineSearch, self).load_state_dict(state_dict)
        self.inner_optimizer.load_state_dict(self.state['inner_optimizer'])

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
                    loss_value = self._evaluate_model(model, loss_func, X, y)
                    self._backpropagate_gradients(loss_value)
            dot_product, cosine = self._gradient_vector_dot_product(self.param_groups, direction)
            logging.debug(f"Cosine: {cosine}")
            lr = self._reset_lr(self.state['lr'])

            for i in range(self.max_iterations):
                self._assign_new_params(params_current, direction, lr=lr)
                new_loss_value = self._evaluate_model(model, loss_func, X, y)
                logging.debug(f"{i} {lr} {new_loss_value} {loss_value} " +
                              f"{loss_value+self.c*lr*dot_product} " +
                              f"{loss_value + (1-self.c)*lr*dot_product}")

                if cosine < self.min_cosine:
                    self.cosine_breaks += 1
                    i = -1
                    logging.debug(f"Cosine too small. Break {self.cosine_breaks/self.total_steps}")
                    break

                upper_bound_condition = (
                    new_loss_value <= loss_value + self.c*lr*dot_product
                )
                lower_bound_condition = (
                    new_loss_value >= loss_value + (1-self.c)*lr*dot_product
                )
                logging.debug(f"{upper_bound_condition} {lower_bound_condition}")
                if self.search_strategy == 'armijo':
                    if upper_bound_condition:
                        break
                    else:
                        lr *= self.beta
                else:  # goldstein
                    if upper_bound_condition and lower_bound_condition:
                        break
                    elif upper_bound_condition:
                        lr = min(lr / self.beta, self.max_lr)
                    elif lower_bound_condition:
                        lr = lr * self.beta
                    else:
                        raise RuntimeError("Something very weird happened to loss value:" +
                                           f"{new_loss_value}")
            self.state['num_iterations'] = i
            self.state['lr'] = lr
            self.state['cosine'] = cosine.item()

    def _get_direction(self, params_a: List[Dict[str, Any]], params_b: List[Dict[str, Any]]) \
            -> List[Dict[str, List[torch.Tensor]]]:
        res = []
        for group_a, group_b in zip(params_a, params_b):
            group_res = []
            lr = group_b['lr']
            for p_a, p_b in zip(group_a['params'], group_b['params']):
                group_res.append(
                    (p_b-p_a)/lr
                )
            res.append({
                'params': group_res
            })
        return res

    def _gradient_vector_dot_product(self, gradient: List[Dict[str, List[torch.Tensor]]],
                                     vector: List[Dict[str, List[torch.Tensor]]]) -> float:
        dot_product = 0.0
        grad_len = 0.0
        vector_len = 0.0
        for group_gradient, group_vector in zip(gradient, vector):
            for p_a, p_b in zip(group_gradient['params'], group_vector['params']):
                dot_product += torch.sum(p_a.grad*p_b)
                grad_len += torch.sum(p_a.grad*p_a.grad)
                vector_len += torch.sum(p_b*p_b)

        return dot_product, -dot_product/(grad_len*vector_len)**0.5

    def _assign_new_params(self, params_current: List[Dict[str, Any]],
                           direction: List[Dict[str, List[torch.Tensor]]], lr: float) -> None:
        for group_model, group_current, group_direction in \
                zip(self.param_groups, params_current, direction):
            for p_model, p_current, p_direction in \
                    zip(group_model['params'], group_current['params'], group_direction['params']):
                p_model.data = p_current + lr*p_direction

    def _evaluate_model(self, model: nn.Module,
                        loss_func: Callable[[torch.Tensor, torch.Tensor], float],
                        X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_predicted = model(X)
        loss = loss_func(y_predicted, y)
        return loss

    def _backpropagate_gradients(self, loss: torch.Tensor) -> None:
        loss.backward()

    def _reset_lr(self, lr: float) -> float:
        if self.reset_strategy == 'max':
            return self.max_lr
        elif self.reset_strategy == 'keep':
            return lr
        else:
            return min(lr * self.gamma, self.max_lr)
