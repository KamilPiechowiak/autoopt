import os
from typing import Dict, List
from torch import nn
import torch

from autoopt.data.datasets_factory import DatasetsFactory
from autoopt.distributed.base_connector import BaseConnector
from autoopt.metrics.stats_reporter import StatsReporter
from autoopt.models.models_factory import ModelsFactory
from autoopt.training.train import _single_epoch
from autoopt.utils.file_utils import read_json
from autoopt.utils.path_utils import get_path


def validate_experts(config: Dict, connector: BaseConnector) -> None:
    """Validate interpolation of independent experts"""
    path = get_path(config)
    device = connector.get_device()

    if not connector.is_master():
        connector.rendezvous('download_only_once')
    train_dataset, val_dataset = DatasetsFactory().get_dataset(config["dataset"])
    if connector.is_master():
        connector.rendezvous('download_only_once')

    model = ModelsFactory().get_model(config['model'])
    model.to(device)
    source_models = []
    for _ in range(len(config['source_models'])):
        source_models.append(ModelsFactory().get_model(config['model']))
        source_models[-1].to(device)

    loss_func = nn.CrossEntropyLoss()
    metrics = {
        'loss': loss_func,
        'acc': lambda input, target:
            (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
        'acc5': lambda input, target:
            (torch.sort(input, 1, descending=True)[1][:, :5] == target.unsqueeze(-1)).sum() /
            float(target.shape[0])
    }

    stats_reporters = []
    for alpha in config['alphas']:
        stats_reporters.append(
            StatsReporter(metrics, os.path.join(path, str(alpha)))
        )
        stats_reporters[-1].add_metrics(['loss_detailed', 'acc_detailed', 'acc5_detailed'])

    train_sampler, val_sampler = connector.get_samplers(
        train_dataset, val_dataset)

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

    cumulative_loss_decay = config['cumulative_loss_decay']
    sources_losses = load_sources_losses(config['source_models'])
    cumulative_losses = torch.zeros(len(source_models), device=device)

    with torch.no_grad():
        for epoch in range(config['epochs']):
            connector.print(f'EPOCH: {epoch}')

            for i, losses in enumerate(sources_losses):
                n = len(losses)
                d = n // config['epochs']
                print(epoch*d, (epoch+1)*d)
                for loss in losses[epoch*d:(epoch+1)*d]:
                    cumulative_losses[i] *= cumulative_loss_decay
                    cumulative_losses[i] += loss

            load_models(source_models, config['source_models'], epoch)
            for stats_reporter, alpha in zip(stats_reporters, config['alphas']):
                update_model(model, source_models, cumulative_losses, alpha)
                model.eval()
                _single_epoch(epoch, device, connector, model,
                              connector.wrap_data_loader(train_loader, device), loss_func,
                              stats=stats_reporter, metrics=metrics, gradeBy='loss',
                              is_training_set=True)
                _single_epoch(epoch, device, connector, model,
                              connector.wrap_data_loader(val_loader, device), loss_func,
                              stats=stats_reporter, metrics=metrics, gradeBy='loss',
                              is_training_set=False)

    if connector.is_master() and config.get('gcp', True):
        os.system(f'gsutil cp -r {path} {config["bucket_path"]}')


def load_sources_losses(source_models_paths: List[str]) -> List[List[List[float]]]:
    losses = []
    for source_model_path in source_models_paths:
        stats = read_json(f'{source_model_path}/stats.json')
        losses.append(stats['loss_detailed/train'])
    return losses


def load_models(source_models: List[nn.Module], source_models_paths: List[str], epoch: int) -> None:
    for source_model, path in zip(source_models, source_models_paths):
        source_model.load_state_dict(torch.load(f'{path}/{epoch}.pt')['model'])


def update_model(model: nn.Module, source_models: List[nn.Module],
                 cumulative_losses: torch.Tensor, alpha: float):
    probabilites = torch.exp(-alpha*(cumulative_losses-cumulative_losses.min()))
    probabilites /= torch.sum(probabilites)

    source_params = [source_model.parameters() for source_model in source_models]
    for params in zip(model.parameters(), *source_params):
        stacked_params = torch.stack([prob * x.data for prob, x in zip(probabilites, params[1:])])
        params[0].data[:] = torch.sum(stacked_params, dim=0)
