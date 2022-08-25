from turtle import back
from typing import Callable
import torch
from torch import nn

from autoopt.losses.generic_loss_creator import GenericLossCreator


class VAELossCreator(GenericLossCreator):

    def get_loss_function_on_minibatch(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) \
            -> Callable:
        X = X[:, 0].unsqueeze(1)
        X = 0.2860+X*0.3530
        X = X*0.8 + 0.1
        single_image_size = X.shape[-1]*X.shape[-2]
        target_flat = X.reshape(-1, single_image_size)

        def loss(backward=False):
            output, mean, std = model(X)
            output_flat = output.reshape(-1, single_image_size)
            image_loss = torch.mean((output_flat - target_flat).pow(2).sum(dim=1))
            latent_loss = -0.5 * torch.mean(
                (1 + 2 * std - mean.pow(2) - torch.exp(2 * std)).sum(dim=1)
            )
            total_loss = image_loss + latent_loss
            if backward:
                total_loss.backward()
            return total_loss, (output, mean, std)
        return loss
