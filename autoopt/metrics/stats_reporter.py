import os
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


class StatsReporter:
    def __init__(self, metrics: Dict, path: str) -> None:
        self.metric_values = {}
        self.phases = ['train', 'val', 'test']
        for metric in metrics.keys():
            for phase in self.phases:
                self.metric_values[f'{metric}/{phase}'] = []
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.writer = SummaryWriter(self.path)

    def add_metrics(self, metrics: List[str]) -> None:
        for metric in metrics:
            for phase in self.phases:
                self.metric_values[f'{metric}/{phase}'] = []

    def update(self, metric_values: Dict, training_phase: str, dump: bool = True) \
            -> None:
        assert (training_phase in self.phases)
        suf = f'/{training_phase}'
        for metric, value in metric_values.items():
            name = f'{metric}{suf}'
            if metric == 'profile':
                self.metric_values[name].append(value)
                continue
            if not isinstance(value, np.ndarray):
                value = [value]
            for v in value:
                self.writer.add_scalar(name, v, len(self.metric_values[name]))
                self.metric_values[name].append(float(v))

            if not dump:
                continue
            if len(value) == 1:
                print(name, value[0])
            plt.clf()
            for plot_name in [f'{metric}/train', f'{metric}/val']:
                n = len(self.metric_values[plot_name])
                plt.plot(np.arange(n), self.metric_values[plot_name], label=plot_name)
                plt.legend()
            plt.savefig(f'{self.path}/{metric}')
        if dump:
            with open(f'{self.path}/stats.json', 'w') as f:
                json.dump(self.metric_values, f)
