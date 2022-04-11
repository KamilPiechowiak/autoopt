from datetime import datetime
from typing import Callable, Dict
import torch
from torch import nn
import logging
import os
import shutil

from autoopt.data.datasets_factory import DatasetsFactory
from autoopt.distributed.base_connector import BaseConnector
from autoopt.models.models_factory import ModelsFactory
from autoopt.optimizers.optimizers_factory import OptimizersFactory
from autoopt.metrics.stats_reporter import StatsReporter
from autoopt.utils.memorizing_iterator import MemorizingIterator
from autoopt.utils.path_utils import get_path


def _single_epoch(epoch: int, device: torch.device, connector: BaseConnector,
                  model: nn.Module, loader: torch.utils.data.DataLoader, loss_func: Callable,
                  opt: torch.optim.Optimizer = None, stats: StatsReporter = None,
                  metrics: Dict = {}, gradeBy: str = 'bce', grad_acc: int = 1):
    start_time = datetime.now()
    assert gradeBy in metrics.keys()
    is_training = (opt is not None)
    metric_values = {}
    for metric in metrics.keys():
        metric_values[metric] = []

    loader_iterator = MemorizingIterator(loader)

    for i, (x, y) in enumerate(loader_iterator):
        y_pred = model(x)
        loss = loss_func(y_pred, y)

        for metric, f in metrics.items():
            metric_values[metric].append(f(y_pred, y).detach())
        if is_training:
            loss.backward()
            if i % grad_acc == 0:
                connector.optimizer_step(opt, model=model, loss_func=loss_func,
                                         data_iterator=loader_iterator, loss_value=loss)
                opt.zero_grad()
                if connector.is_master() and stats is not None:
                    keys = ['num_iterations', 'lr', 'cosine']
                    for key in keys:
                        if key in opt.state:
                            stats.update({key: opt.state[key]}, is_training=True, dump=False)
        break  # TODO remove
    if is_training and i % grad_acc != 0:
        connector.optimizer_step(opt, model=model, loss_func=loss_func,
                                 data_iterator=loader_iterator)
        opt.zero_grad()

    metric_keys = list(metric_values.keys())
    metric_list = []
    for metric in metric_keys:
        metric_list.append(torch.tensor(metric_values[metric], device=device).mean())

    connector.all_avg(metric_list)
    for i, metric in enumerate(metric_keys):
        metric_values[metric] = metric_list[i].item()
    if connector.is_master() and stats is not None:
        stats.update(metric_values, is_training=is_training)

    end_time = datetime.now()
    connector.print(end_time - start_time, flush=True)
    return metric_values[gradeBy]


def _save_model(connector: BaseConnector, model: nn.Module,
                optimizer: torch.optim.Optimizer, path: str):
    connector.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def train(config: Dict, connector: BaseConnector) -> None:
    path = get_path(config)
    device = connector.get_device()

    if not connector.is_master():
        connector.rendezvous('download_only_once')
    train_dataset, val_dataset = DatasetsFactory().get_dataset(config["dataset"])
    if connector.is_master():
        connector.rendezvous('download_only_once')

    model = ModelsFactory().get_model(config['model'])
    model.to(device)
    optimizer = OptimizersFactory().get_optimizer(config['optimizer'], model.parameters())

    loss_func = nn.CrossEntropyLoss()
    metrics = {
        'loss': loss_func,
        'acc': lambda input, target:
            (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
        'acc5': lambda input, target:
            (torch.sort(input, 1, descending=True)[1][:, :5] == target.unsqueeze(-1)).sum() /
            float(target.shape[0])
    }
    if connector.is_master():
        stats_reporter = StatsReporter(metrics, path)
        stats_reporter.add_metrics(['num_iterations', 'lr', 'cosine'])
    else:
        stats_reporter = None

    train_sampler, val_sampler = connector.get_samplers(train_dataset, val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        drop_last=True)

    best_loss = 1e10
    for epoch in range(config['epochs']):
        if hasattr(train_sampler, "set_epoch"):
            logging.info(f"Setting epoch {epoch}")
            train_sampler.set_epoch(epoch + config['repeat'] * 2137)
        connector.print(f'EPOCH: {epoch}')
        model.train()
        _single_epoch(epoch, device, connector, model,
                      connector.wrap_data_loader(train_loader, device), loss_func, optimizer,
                      stats=stats_reporter, metrics=metrics, gradeBy='loss',
                      grad_acc=config.get("grad_acc", 1))

        with torch.no_grad():
            model.eval()
            loss = _single_epoch(epoch, device, connector, model,
                                 connector.wrap_data_loader(val_loader, device), loss_func,
                                 stats=stats_reporter, metrics=metrics, gradeBy='loss',
                                 grad_acc=config.get("grad_acc", 1))
            if loss < best_loss:
                best_loss = loss
                if config.get("checkpoint", True):
                    _save_model(connector, model, optimizer, f'{path}/best.pt')

        if config.get("checkpoint", True) is True:
            _save_model(connector, model, optimizer, f'{path}/current.pt')
            # if epoch % config['persist_state_every'] == config['persist_state_every'] - 1 and \
            #         connector.is_master() and os.path.exists(f'{path}/best.pt'):
            #     shutil.copy(f'{path}/best.pt', f'{path}/{epoch}_checkpoint.pt')

    if connector.is_master() and config.get('gcp', True):
        os.system(f'gsutil cp -r {path} {config["bucket_path"]}')
