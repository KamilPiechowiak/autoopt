import copy
from typing import Dict, List, Callable
import numpy as np
import torch
from torch import nn
from torchvision.datasets import VisionDataset


from autoopt.distributed.base_connector import BaseConnector
from autoopt.optimizers.optimizer_wrapper import OptimizerWrapper
from autoopt.utils.file_utils import save_json


class ProfileStats:

    def __init__(self, batch_sizes: List[int], num_workers: int, path: str,
                 connector: BaseConnector, lr_min: float = 1e-5, lr_max: float = 10,
                 n_steps: int = 61, samples_limit: int = 2, epochs: int = None):
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.path = path
        self.connector = connector
        self.samples_limit = samples_limit
        self.lrs = np.exp(np.linspace(np.log(lr_min), np.log(lr_max), n_steps))
        self.stats = {}
        self.possible_directions = ["gradient", "optimizer", "random"]
        for batch_size in self.batch_sizes:
            for stage in ["train", "val"]:
                for direction in self.possible_directions:
                    for x in ["real", "approx", "length"]:
                        self.stats[f"{batch_size}_{direction}_{x}/{stage}"] = []

        self.previous_optimizer_direction = None
        self.stats['optimizer_epoch_directions'] = []

    def analyse_profiles(self, optimizer: OptimizerWrapper, model: nn.Module, loss_func: Callable,
                         dataset: VisionDataset, is_training: bool = True):
        if is_training:
            stage = "train"
        else:
            stage = "val"
        optimizer.zero_grad()
        params_current = copy.deepcopy(optimizer.param_groups)
        for batch_size in self.batch_sizes:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(2147483647)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.RandomSampler(dataset, generator=generator),
                num_workers=self.num_workers,
                drop_last=True)
            loader = self.connector.wrap_data_loader(loader, self.connector.get_device())
            for possible_direction in self.possible_directions:
                self.stats[f"{batch_size}_{possible_direction}_real/{stage}"].append([])
                self.stats[f"{batch_size}_{possible_direction}_approx/{stage}"].append([])
                self.stats[f"{batch_size}_{possible_direction}_length/{stage}"].append([])
            for j, (x, y) in enumerate(loader):
                print(y[::8])
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step(simulate=True)
                with torch.no_grad():
                    directions = [self.extract_negative_gradient(optimizer.param_groups)]
                    if optimizer.direction is not None:
                        directions.append(optimizer.direction)
                    directions.append(self.get_random_vector(optimizer.param_groups))
                    for i, direction in enumerate(directions):
                        print(batch_size, j, i, flush=True)
                        dot_product, cosine = optimizer._gradient_vector_dot_product(
                            optimizer.param_groups, direction)
                        print(dot_product, cosine)
                        real_losses = []
                        linear_approx = []
                        for lr in self.lrs:
                            optimizer._assign_new_params(params_current, direction, lr)
                            new_loss_value = loss_func(model(x), y)
                            real_losses.append((new_loss_value).detach())
                            linear_approx.append((loss + lr*dot_product).detach())
                            self.connector.step()
                            # print(real_losses[-1].detach(), flush=True)
                        self.connector.step()
                        self.stats[f"{batch_size}_{self.possible_directions[i]}_real/{stage}"][-1] \
                            .append([x.item() for x in real_losses])
                        self.stats[f"{batch_size}_{self.possible_directions[i]}_approx/{stage}"][-1] \
                            .append([x.item() for x in linear_approx])
                        self.connector.step()

                        dot_product, cosine = optimizer._gradient_vector_dot_product(
                            direction, direction, is_first_grad=False)
                        self.stats[f"{batch_size}_{self.possible_directions[i]}_length/{stage}"][-1] \
                            .append((dot_product**0.5).cpu().item())
                        
                    optimizer._assign_new_params(params_current, direction, 0)
                optimizer.zero_grad()
                if j+1 == self.samples_limit:
                    break
        with torch.no_grad():
            optimizer._assign_new_params(params_current, direction, 0)

            self.stats['optimizer_step_directions'] = optimizer.history
            if is_training:
                if self.previous_optimizer_direction is not None:
                    self.stats['optimizer_epoch_directions'].append(
                        -optimizer._gradient_vector_dot_product(
                            self.previous_optimizer_direction,
                            optimizer.direction,
                            is_first_grad=False
                        )[1].item())
                self.previous_optimizer_direction = optimizer.direction
            save_json(self.stats, f"{self.path}/profiles_{self.connector.get_rank()}.json")

    def extract_negative_gradient(self, params: List[Dict[str, List[torch.Tensor]]]) \
            -> List[Dict[str, List[torch.Tensor]]]:
        res = []
        for group in params:
            group_res = []
            for p in group['params']:
                group_res.append(-p.grad)
            res.append({
                'params': group_res
            })
        return res

    def get_random_vector(self, params: List[Dict[str, List[torch.Tensor]]]) \
            -> List[Dict[str, List[torch.Tensor]]]:
        res = []
        for group in params:
            group_res = []
            for p in group['params']:
                group_res.append(torch.rand_like(p))
            res.append({
                'params': group_res
            })
        return res