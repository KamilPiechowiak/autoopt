from typing import Callable, List, Tuple

import torch
from torch import optim, nn
import copy
from autoopt.losses.generic_loss_creator import GenericLossCreator
from autoopt.optimizers.extended_optimizer import ExtendedOptimizer
from autoopt.utils.memorizing_iterator import MemorizingIterator


class ExpertsNonLinearizedStationary(ExtendedOptimizer):

    def __init__(self, params: List[torch.Tensor], inner_optimizer: optim.Optimizer,
                 alpha: float, learning_rates: List[float], cumulative_loss_decay: float = 1,
                 expecation_on_logarithms: bool = False) -> None:
        super(ExpertsNonLinearizedStationary, self).__init__(params, {})
        self.step_kwargs = {"model", "loss_creator", "data_iterator"}
        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        device = self.param_groups[0]['params'][0].device
        self.learning_rates = torch.tensor(learning_rates, device=device)
        self.probabilities = torch.ones(len(learning_rates), device=device)/len(learning_rates)
        self.cumulative_losses = torch.zeros(len(learning_rates), device=device)
        self.previous_params = None
        self.previous_direction = None
        self.total_steps = 0
        self.cumulative_loss_decay = cumulative_loss_decay
        self.expecation_on_logarithms = expecation_on_logarithms

    @torch.no_grad()
    def step(self, model: nn.Module, loss_creator: GenericLossCreator,
             data_iterator: MemorizingIterator) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.enable_grad():
            self._update_loss_function(model, loss_creator, data_iterator)
            loss_value, y_pred = self._evaluate_model(backward=True)

        self._turn_off_batchnorms_accumulation(model)

        self.total_steps += 1
        params_current = copy.deepcopy(self.param_groups)
        self.inner_optimizer.step()
        params_next = copy.deepcopy(self.inner_optimizer.param_groups)
        direction = self._get_direction(params_current, params_next)
        X, y = data_iterator.current()

        if self.previous_params is not None:
            for i, lr in enumerate(self.learning_rates):
                self._assign_new_params(self.previous_params, self.previous_direction, lr=lr)
                self.cumulative_losses[i] = self.cumulative_loss_decay * self.cumulative_losses[i]
                self.cumulative_losses[i] += self._evaluate_model(backward=False)[0]

        probabilities = self.probabilities * \
            torch.exp(-self.alpha * (self.cumulative_losses-self.cumulative_losses.min()))
        probabilities /= torch.sum(probabilities)
        if self.expecation_on_logarithms:
            expected_lr = torch.exp(torch.sum(probabilities*torch.log(self.learning_rates)))
        else:
            expected_lr = torch.sum(probabilities*self.learning_rates)
        self.state['prob'] = probabilities.cpu().numpy()
        self.state['lr'] = expected_lr.item()
        self._assign_new_params(params_current, direction, lr=expected_lr)

        self.previous_params = params_current
        self.previous_direction = direction

        self._turn_on_batchnorms_accumulation(model)

        return loss_value, y_pred
        # print(self.cumulative_losses/self.total_steps, probabilities, expected_lr)
