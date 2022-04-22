from typing import Any, Callable, Dict, List, Tuple

from typing import List
import torch
from torch import optim, nn


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

    def _evaluate_model(self, model: nn.Module,
                        loss_func: Callable[[torch.Tensor, torch.Tensor], float],
                        X: torch.Tensor, y: torch.Tensor, average: bool = True) -> torch.Tensor:
        y_predicted = model(X)
        loss = loss_func(y_predicted, y)
        return loss

    def _backpropagate_gradients(self, loss: torch.Tensor) -> None:
        loss.backward()

    def _reduce_average(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def _mark_step(self) -> None:
        pass
