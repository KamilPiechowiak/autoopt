from datetime import datetime
import sched
from turtle import backward
from typing import Callable, Dict
import torch
from torch import nn
import os

from autoopt.data.datasets_factory import DatasetsFactory
from autoopt.data.on_device_dataset_wrapper import OnDeviceDatasetWrapper
from autoopt.distributed.base_connector import BaseConnector
from autoopt.losses.generic_loss_creator import GenericLossCreator
from autoopt.losses.loss_creators_factory import LossCreatorsFactory
from autoopt.models.models_factory import ModelsFactory
from autoopt.optimizers.optimizer_wrapper import OptimizerWrapper
from autoopt.optimizers.optimizers_factory import OptimizersFactory
from autoopt.metrics.stats_reporter import StatsReporter
from autoopt.optimizers.schedulers_factory import SchedulersFactory
from autoopt.stats.profile_stats import ProfileStats
from autoopt.utils.memorizing_iterator import MemorizingIterator
from autoopt.utils.path_utils import get_path
from autoopt.stats.gradient_direction_stats import GradientDirectionStats

STOP_FAST = False


def _single_epoch(epoch: int, device: torch.device, connector: BaseConnector,
                  model: nn.Module, loader: torch.utils.data.DataLoader,
                  loss_creator: GenericLossCreator,
                  opt: torch.optim.Optimizer = None, stats: StatsReporter = None,
                  metrics: Dict = {}, gradeBy: str = 'bce',
                  training_phase: str = None):
    start_time = datetime.now()
    assert gradeBy in metrics.keys()
    is_training = (opt is not None)
    if training_phase is None:
        if is_training:
            training_phase = 'train'
        else:
            training_phase = 'val'
    metric_values = {}
    for metric in metrics.keys():
        metric_values[metric] = []

    print(training_phase)

    loader_iterator = MemorizingIterator(loader)

    for i, (x, y) in enumerate(loader_iterator):
        y = y.flatten()

        if is_training:
            if hasattr(opt, "step_kwargs"):
                loss, y_pred = connector.optimizer_step(
                    opt, model=model, loss_creator=loss_creator,
                    data_iterator=loader_iterator)
            else:
                loss, y_pred = loss_creator.get_loss_function_on_minibatch(
                    model, x, y)(backward=True)
                connector.optimizer_step(opt)
            opt.zero_grad()
            if connector.is_master() and stats is not None:
                keys = ['num_iterations', 'lr', 'cosine', 'prob', 'beta1', 'beta2', 'eps',
                        'weight_decay', 'direction_length', 'profile', 'profile_type']
                for key in keys:
                    if key in opt.state:
                        stats.update({key: opt.state[key]}, training_phase, dump=False)
        else:
            loss, y_pred = loss_creator.get_loss_function_on_minibatch(
                model, x, y)(backward=False)

        for metric, f in metrics.items():
            metric_values[metric].append(f(y_pred, y, loss).detach())

        if STOP_FAST and i == 10:
            break

    metric_keys = list(metric_values.keys())
    arr = []
    for metric in metric_keys:
        metric_values[metric] = torch.tensor(metric_values[metric], device=device)
        arr.append(metric_values[metric])

    connector.all_avg(arr)

    for i, metric in enumerate(metric_keys):
        metric_values[metric] = arr[i]

    for metric in metric_keys:
        connector.step()
        if connector.is_master() and stats is not None:
            stats.update({
                metric: metric_values[metric].mean().item(),
                f"{metric}_detailed": metric_values[metric].cpu().numpy()
            }, training_phase)

    end_time = datetime.now()
    connector.print(end_time - start_time, flush=True)
    return metric_values[gradeBy].mean().item()


def _save_model(connector: BaseConnector, model: nn.Module,
                optimizer: torch.optim.Optimizer, path: str):
    connector.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def train(config: Dict, connector: BaseConnector) -> None:
    if config.get('local', False):
        global STOP_FAST
        STOP_FAST = True

    path = get_path(config)
    device = connector.get_device()

    if not connector.is_master():
        connector.rendezvous('download_only_once')
    datasets = DatasetsFactory().get_dataset(config["dataset"])
    if len(datasets) == 3:
        train_dataset, val_dataset, test_dataset = datasets
    else:
        train_dataset, val_dataset = datasets
        test_dataset = None

    if connector.is_master():
        connector.rendezvous('download_only_once')

    if config["dataset"].get("wrap"):
        train_dataset = OnDeviceDatasetWrapper(train_dataset, device)
        val_dataset = OnDeviceDatasetWrapper(val_dataset, device)
        if test_dataset is not None:
            test_dataset = OnDeviceDatasetWrapper(test_dataset, device)

    model = ModelsFactory().get_model(config['model'])
    model.to(device)
    optimizer = OptimizersFactory().get_optimizer(config['optimizer'], model.parameters())
    if config.get('scheduler'):
        scheduler = SchedulersFactory().get_scheduler(config['scheduler'], optimizer)
    else:
        scheduler = None

    loss_creator = LossCreatorsFactory().get_loss_creator(config.get('task', 'classification'))
    if config.get('task', 'classification') == 'classification':
        loss_func = nn.CrossEntropyLoss()
        metrics = {
            'loss': lambda input, target, loss: loss,
            'acc': lambda input, target, loss:
                (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
            'acc5': lambda input, target, loss:
                (torch.sort(input, 1, descending=True)[1][:, :5] == target.unsqueeze(-1)).sum() /
                float(target.shape[0]),
        }
        grade_by = 'acc'
    else:
        metrics = {
            'loss': lambda input, target, loss: loss
        }
        grade_by = 'loss'
    if connector.is_master():
        stats_reporter = StatsReporter(metrics, path)
        stats_reporter.add_metrics(['num_iterations', 'lr', 'cosine', 'loss_detailed',
                                    'acc_detailed', 'acc5_detailed', 'prob',
                                    'beta1', 'beta2', 'eps', 'weight_decay', 'direction_length',
                                    'profile', 'profile_type'])
    else:
        stats_reporter = None

    if 'stats' in config and 'gradient_direction' in config['stats']:
        gradient_direction_stats = GradientDirectionStats(config['stats']['gradient_direction'],
                                                          path, connector)
    else:
        gradient_direction_stats = None

    if 'stats' in config and 'profile' in config['stats']:
        profile_stats = ProfileStats(**config['stats']['profile'],
                                     path=path, connector=connector)
    else:
        profile_stats = None

    train_sampler, val_sampler = connector.get_samplers(
        train_dataset, val_dataset,
        (gradient_direction_stats is not None) or (profile_stats is not None)
    )

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

    best_acc = -1000000.0
    for epoch in range(config['epochs']):
        connector.print(f'EPOCH: {epoch}')
        model.train()
        if gradient_direction_stats is not None:
            gradient_direction_stats.analyse_gradients(model, loss_func, train_dataset,
                                                       train_sampler, 0, True)
            gradient_direction_stats.analyse_gradients(model, loss_func, val_dataset,
                                                       val_sampler, 0, False)
        if profile_stats is not None and epoch in config['stats']['profile']['epochs']:
            wrapped_optimizer = OptimizerWrapper(model.parameters(), optimizer.inner_optimizer)
            print(wrapped_optimizer)
            profile_stats.analyse_profiles(wrapped_optimizer, model, loss_func, train_dataset,
                                           True)
            profile_stats.analyse_profiles(wrapped_optimizer, model, loss_func, val_dataset,
                                           False)

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch + config['repeat'] * 2137)
        _single_epoch(epoch, device, connector, model,
                      connector.wrap_data_loader(train_loader, device), loss_creator, optimizer,
                      stats=stats_reporter, metrics=metrics, gradeBy=grade_by)

        with torch.no_grad():
            model.eval()
            acc = _single_epoch(epoch, device, connector, model,
                                connector.wrap_data_loader(val_loader, device), loss_creator,
                                stats=stats_reporter, metrics=metrics, gradeBy=grade_by)
            if grade_by != 'acc':
                acc *= -1
            if acc > best_acc:
                print(f"Saving {acc} > {best_acc}")
                best_acc = acc
                _save_model(connector, model, optimizer, 'best.pt')

        if config.get("checkpoint", False):
            _save_model(connector, model, optimizer, f'{path}/{epoch}.pt')
            # if epoch % config['persist_state_every'] == config['persist_state_every'] - 1 and \
            #         connector.is_master() and os.path.exists(f'{path}/best.pt'):
            #     shutil.copy(f'{path}/best.pt', f'{path}/{epoch}_checkpoint.pt')

        if scheduler is not None:
            stats_reporter.update({'lr': optimizer.param_groups[0]['lr']},
                                  'train', dump=False)
            scheduler.step()

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SequentialSampler(
                test_dataset
            ),
            num_workers=config['num_workers'],
            drop_last=True)
        model.load_state_dict(torch.load('best.pt')['model'])
        with torch.no_grad():
            model.eval()
            _single_epoch(epoch, device, connector, model,
                          connector.wrap_data_loader(test_loader, device), loss_creator,
                          stats=stats_reporter, metrics=metrics, gradeBy=grade_by,
                          training_phase='test')

    if connector.is_master() and config.get('gcp', True):
        os.system(f'gsutil cp -r {path} {config["bucket_path"]}')
