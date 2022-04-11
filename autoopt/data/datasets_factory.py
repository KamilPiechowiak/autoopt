from typing import Callable, Dict, Tuple
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms, datasets


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
                'FASHION_MNIST': lambda: datasets.FashionMNIST(path, train=True,
                                                               transform=transform, download=True),
                'CIFAR10': lambda: datasets.CIFAR10(path, train=train,
                                                    transform=transform, download=True),
                'CIFAR100': lambda: datasets.CIFAR100(path, train=train,
                                                      transform=transform, download=True),
                'ImageNet': lambda: datasets.ImageNet(path, split='train' if train else 'val',
                                                      transform=transform)
            }[name]()

        return prepare_dataset(config['name'], config['path'], True, preprocess_train), \
            prepare_dataset(config['name'], config['path'], False, preprocess_val)
