import collections
import os
import codecs
from typing import Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class Tolstoi(Dataset):

    url = "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt"
    encoding = "utf-8"

    def __init__(self, path: str, train: bool = True, download: bool = False,
                 seq_length: int = 50, test_size: int = 0.2) -> None:
        self.path = os.path.join(path, 'tolstoi')
        os.makedirs(self.path, exist_ok=True)
        file = 'train.npy' if train else 'test.npy'
        file_path = os.path.join(self.path, file)
        if not os.path.exists(file_path) and download:
            self.download(test_size)
        data = np.load(file_path)
        num_samples = (data.shape[0]-1) // seq_length
        x = data[:num_samples*seq_length].reshape(num_samples, seq_length)
        y = data[1:num_samples*seq_length+1].reshape(num_samples, seq_length)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def download(self, test_size) -> None:
        input_file = os.path.join(self.path, 'input.txt')
        os.system(f"wget -O {input_file} {self.url}")

        with codecs.open(input_file, "r", encoding=self.encoding) as inp_file:
            data = inp_file.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        print("Vocab size", len(chars))
        vocab = dict(zip(chars, range(len(chars))))

        # Create array of character ids
        array = np.array(list(map(vocab.get, data)))

        # Split in train and test and save to .npy files
        train_size = int(np.ceil((1.0 - test_size) * np.size(array)))
        train = array[0:train_size]
        test = array[train_size:]
        np.save(os.path.join(self.path, 'train.npy'), train)
        np.save(os.path.join(self.path, 'test.npy'), test)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
