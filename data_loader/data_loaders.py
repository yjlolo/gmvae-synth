import numpy as np
import torch
from torchvision import datasets, transforms
from base import BaseDataLoader, SoLBase
from audio_process import transformers
from audio_process.datasets import CollectSOLSpec


class SolDataLoader(SoLBase):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        trsfm = transforms.Compose([
            transformers.LoadTensor(),
            transformers.ChunkSpec(sr=22050, hop_size=512, duration=0.5),
            transformers.PickFirstChunk(),
            transformers.MinMaxNorm(-1, 1)
        ])
        self.data_dir = data_dir
        self.dataset = CollectSOLSpec(self.data_dir, transform=trsfm)
        super(SolDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NormDataLoader(SoLBase):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        trsfm = transforms.Compose([
            transformers.LoadTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = CollectSOLSpec(self.data_dir, transform=trsfm)
        super(NormDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class PickFirstFrame:
    def __call__(self, x):
        # assume x is a tensor [n_band, time]
        return x[:, 0]


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


if __name__ == "__main__":
    # data_dir = '/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/cqt'
    data_dir = '/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/melspec_256-first_chunk-include_onset-fix_piano-normalize'
    # dl = SolDataLoader(data_dir, 128, True, 0.1, 1)
    dl = NormDataLoader(data_dir, 128, True, 0.1, 0)
    # d = next(iter(dl))
    for (data, target, f) in dl:
        print(data.size(), target[0].size())#, f.size())
