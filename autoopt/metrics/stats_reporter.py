import os
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class StatsReporter:
    def __init__(self, metrics: Dict, path: str) -> None:
        self.metric_values = {}
        for metric in metrics.keys():
            self.metric_values[f'{metric}/train'] = []
            self.metric_values[f'{metric}/val'] = []
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.writer = SummaryWriter(self.path)

    def add_metrics(self, metrics: List[str]) -> None:
        for metric in metrics:
            self.metric_values[f'{metric}/train'] = []
            self.metric_values[f'{metric}/val'] = []

    def update(self, metric_values: Dict, is_training: bool = False, dump: bool = True) \
            -> None:
        suf = '/train' if is_training else '/val'
        for metric, value in metric_values.items():
            name = f'{metric}{suf}'
            self.writer.add_scalar(name, value, len(self.metric_values[name]))
            self.metric_values[name].append(value)

            if not dump:
                continue

            print(name, value)
            plt.clf()
            for plot_name in [f'{metric}/train', f'{metric}/val']:
                n = len(self.metric_values[plot_name])
                plt.plot(np.arange(n), self.metric_values[plot_name], label=plot_name)
                plt.legend()
            plt.savefig(f'{self.path}/{metric}')
            with open(f'{self.path}/stats.json', 'w') as f:
                json.dump(self.metric_values, f)
