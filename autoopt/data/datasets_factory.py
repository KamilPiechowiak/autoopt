from typing import Callable, Dict, Tuple
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms, datasets
from autoopt.data import tolstoi

from autoopt.data.transform_dataset import TransformDataset


class DatasetsFactory:

    def __init__(self):
        pass

    def get_dataset(self, config: Dict) -> Tuple[VisionDataset, VisionDataset]:
        stack = transforms.Lambda(lambda img: torch.cat([img, img, img], axis=0)
                                  if img.shape[0] == 1 else img)
        if config["name"] in ['FASHION_MNIST', 'MNIST']:
            normalize = transforms.Normalize(mean=0.2860, std=0.3530)
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if config["resolution"] < 100:
            scale = (0.5, 1)
        else:
            scale = (0.1, 1)

        if config.get("noaugment", False):
            preprocess_train = transforms.Compose([
                transforms.Resize(config["resolution"]),
                transforms.CenterCrop(config["resolution"]),
                transforms.ToTensor(),
                stack,
                normalize
            ])
        else:
            preprocess_train = transforms.Compose([
                transforms.RandomResizedCrop(config["resolution"], scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                stack,
                normalize
            ])

        preprocess_val = transforms.Compose([
            transforms.Resize(config["resolution"]),
            transforms.CenterCrop(config["resolution"]),
            transforms.ToTensor(),
            stack,
            normalize
        ])

        def prepare_dataset(name: str, path: str, train: bool, transform: torch.nn.Module):
            return {
                'FASHION_MNIST': lambda: datasets.FashionMNIST(path, train=train,
                                                               transform=transform, download=True),
                'CIFAR10': lambda: datasets.CIFAR10(path, train=train,
                                                    transform=transform, download=True),
                'CIFAR100': lambda: datasets.CIFAR100(path, train=train,
                                                      transform=transform, download=True),
                'ImageNet': lambda: datasets.ImageNet(path, split='train' if train else 'val',
                                                      transform=transform),
                'Tolstoi': lambda: tolstoi.Tolstoi(path, train=train, download=True)
            }[name]()

        dataset_test = prepare_dataset(config['name'], config['path'], False, preprocess_val)
        if config.get('test', False):
            dataset_train_val = prepare_dataset(config['name'], config['path'], True, None)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(2147483647)
            n = len(dataset_train_val)
            n_train = (4*n)//5
            dataset_train, dataset_val = torch.utils.data.random_split(
                dataset_train_val, [n_train, n-n_train], generator)
            dataset_train = TransformDataset(dataset_train, preprocess_train)
            dataset_val = TransformDataset(dataset_val, preprocess_val)
            print(len(dataset_train), len(dataset_val), len(dataset_test))
            print([dataset_train[i][1] for i in range(10)])
            print([dataset_val[i][1] for i in range(10)])
            return dataset_train, dataset_val, dataset_test
        else:
            dataset_train = prepare_dataset(config['name'], config['path'], True, preprocess_train)
            return dataset_train, dataset_test
