from turtle import back
from typing import Callable
import torch
from torch import nn

from autoopt.losses.generic_loss_creator import GenericLossCreator


class ClassificationLossCreator(GenericLossCreator):

    def get_loss_function_on_minibatch(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) \
            -> Callable:
        def loss(backward=False):
            y_flat = y.flatten()
            y_pred = model(X)
            loss_value = nn.CrossEntropyLoss()(y_pred, y_flat)
            if backward:
                loss_value.backward()
            return loss_value, y_pred
        return loss
