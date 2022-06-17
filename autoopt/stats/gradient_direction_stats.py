from typing import Callable, Dict

import torch
from torch import nn
from torchvision.datasets import VisionDataset

from autoopt.distributed.base_connector import BaseConnector
from autoopt.utils.file_utils import save_json


class GradientDirectionStats:

    def __init__(self, config: Dict, path: str, connector: BaseConnector):
        self.batch_sizes = config['batch_sizes']
        self.num_workers = config['num_workers']
        self.path = path
        self.connector = connector
        self.stats = {}
        for batch_size in self.batch_sizes:
            for stage in ["train", "val"]:
                self.stats[f"{batch_size}_cos_avg/{stage}"] = []
                self.stats[f"{batch_size}_cos_pairs/{stage}"] = []
                self.stats[f"{batch_size}_lengths/{stage}"] = []

    def analyse_gradients(self, model: nn.Module, loss_func: Callable,
                          dataset: VisionDataset, sampler: torch.utils.data.Sampler, seed: int,
                          is_training: bool = True) -> None:
        if is_training:
            stage = "train"
        else:
            stage = "val"
        model.zero_grad()
        for batch_size in self.batch_sizes:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,  # FIXME
                # sampler=torch.utils.data.SequentialSampler(dataset),
                num_workers=self.num_workers,
                drop_last=True)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(seed)

            avg_gradient = None
            wrapped_loader = self.connector.wrap_data_loader(loader, self.connector.get_device())
            for i, (x, y) in enumerate(wrapped_loader):
                # print("first", batch_size)
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                loss.backward()
                if avg_gradient is None:
                    avg_gradient = self.get_gradient(model)
                else:
                    avg_gradient += self.get_gradient(model)
                # if i == 10:  # FIXME
                #     break
            avg_gradient_arr = [avg_gradient]
            self.connector.all_avg(avg_gradient_arr)
            avg_gradient = avg_gradient_arr[0]
            # print("Ensure all equal:", avg_gradient[:7], flush=True)
            self.connector.step()

            cosines = []
            pairwise_cosines = []
            lengths = []
            previous_gradient = None

            wrapped_loader = self.connector.wrap_data_loader(loader, self.connector.get_device())
            for i, (x, y) in enumerate(wrapped_loader):
                # print("second", batch_size)
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                loss.backward()
                gradient = self.get_gradient(model)
                lengths.append(torch.linalg.norm(gradient).item())
                cosines.append(self.get_cosine(avg_gradient, gradient))
                if previous_gradient is None:
                    previous_gradient = gradient
                else:
                    pairwise_cosines.append(self.get_cosine(previous_gradient, gradient))
                    previous_gradient = None
                # if i == 10:  # FIXME
                #     break
            cosines = self.connector.all_gather(cosines)
            pairwise_cosines = self.connector.all_gather(pairwise_cosines)
            lengths = self.connector.all_gather(lengths)
            self.connector.print(cosines[:7], flush=True)
            self.connector.step()

            self.stats[f"{batch_size}_cos_avg/{stage}"].append(cosines)
            self.stats[f"{batch_size}_cos_pairs/{stage}"].append(pairwise_cosines)
            self.stats[f"{batch_size}_lengths/{stage}"].append(lengths)

        self.save()

    def save(self):
        if self.connector.is_master():
            save_json(self.stats, f"{self.path}/gradient_directions.json")

    def get_gradient(self, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
            model.zero_grad()
        return gradient

    def get_cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        with torch.no_grad():
            a_len = torch.sum(a**2)**0.5
            b_len = torch.sum(b**2)**0.5
            return (torch.sum(a*b)/a_len/b_len).item()
