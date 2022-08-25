import contextlib
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import optim, nn

from autoopt.losses.generic_loss_creator import GenericLossCreator
from autoopt.utils.memorizing_iterator import MemorizingIterator


class ExtendedOptimizer(optim.Optimizer):

    def __init__(self, *args, **kwargs):
        super(ExtendedOptimizer, self).__init__(*args, **kwargs)

    def state_dict(self) -> dict:
        self.state['inner_optimizer'] = self.inner_optimizer.state_dict()
        return super(ExtendedOptimizer, self).state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        super(ExtendedOptimizer, self).load_state_dict(state_dict)
        self.inner_optimizer.load_state_dict(self.state['inner_optimizer'])

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

    def _normalize_direction(self, direction: List[Dict[str, Any]]) -> None:
        dot_product, _ = self._gradient_vector_dot_product(direction, direction,
                                                           is_first_grad=False)
        length = dot_product**0.5
        for group_direction in direction:
            for p in group_direction['params']:
                p.data /= length
        self.state['direction_length'] = length.cpu().item()

    def _gradient_vector_dot_product(self, gradient: List[Dict[str, List[torch.Tensor]]],
                                     vector: List[Dict[str, List[torch.Tensor]]],
                                     is_first_grad: bool = True) -> Tuple[float, float]:
        dot_product = 0.0
        grad_len = 0.0
        vector_len = 0.0
        for group_gradient, group_vector in zip(gradient, vector):
            for p_a, p_b in zip(group_gradient['params'], group_vector['params']):
                if is_first_grad:
                    dot_product += torch.sum(p_a.grad*p_b)
                    grad_len += torch.sum(p_a.grad*p_a.grad)
                else:
                    dot_product += torch.sum(p_a*p_b)
                    grad_len += torch.sum(p_a*p_a)
                vector_len += torch.sum(p_b*p_b)

        return dot_product, -dot_product/(grad_len*vector_len)**0.5

    def _assign_new_params(self, params_current: List[Dict[str, Any]],
                           direction: List[Dict[str, List[torch.Tensor]]], lr: float) -> None:
        for group_model, group_current, group_direction in \
                zip(self.param_groups, params_current, direction):
            for p_model, p_current, p_direction in \
                    zip(group_model['params'], group_current['params'], group_direction['params']):
                p_model.data = p_current + lr*p_direction

    def _add_direction_to_vector(self, destination: List[Dict[str, Any]],
                                 source: List[Dict[str, Any]], direction: List[Dict[str, Any]],
                                 lr: float) -> None:
        for group_dest, group_source, group_direction in \
                zip(destination, source, direction):
            for p_dest, p_source, p_direction in \
                    zip(group_dest['params'], group_source['params'], group_direction['params']):
                p_dest.data = p_source + lr*p_direction

    def _update_loss_function(self, model: nn.Module, loss_creator: GenericLossCreator,
                              data_iterator: MemorizingIterator) -> None:
        X, y = data_iterator.current()
        self.loss_func = loss_creator.get_loss_function_on_minibatch(model, X, y)
        self.seed = time.time()

    def _evaluate_model(self, backward: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.random_seed_torch(int(self.seed)):
            return self.loss_func(backward)

    def _backpropagate_gradients(self, loss: torch.Tensor) -> None:
        loss.backward()

    def _reduce_average(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def _mark_step(self) -> None:
        pass

    @contextlib.contextmanager
    def random_seed_torch(self, seed):
        """
        source: https://github.com/IssamLaradji/sls/
        """
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            gpu_rng_state = torch.cuda.get_rng_state(0)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(gpu_rng_state)

    def _set_batchnorms_momentum(self, model, batchnorm_momentum):
        if isinstance(model, nn.BatchNorm2d):
            model.momentum = batchnorm_momentum
        else:
            for child in model.children():
                self._set_batchnorms_momentum(child, batchnorm_momentum)

    def _turn_off_batchnorms_accumulation(self, model):
        self._set_batchnorms_momentum(model, 0.0)

    def _turn_on_batchnorms_accumulation(self, model):
        self._set_batchnorms_momentum(model, 0.1)
