import random

import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

from train_utils import *

# high - 2
# low - 1
# none - 0


class EegNetDataset(Dataset):

    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname).item()

        self.X = self.data['X']
        self.y = self.data['y']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])
        label = to_long_tensor([self.y[index]])  # careful

        return sample, label


class EegOnlyDataset(Dataset):

    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname).item()

        self.X = self.data['X']
        self.y = self.data['y']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])
        eeg = sample[0:32]

        label = to_long_tensor([self.y[index]])  # careful

        return eeg, label


class AuthDataset(Dataset):
    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname, allow_pickle=True).item()

        self.X = self.data['X']
        self.y = self.data['y']

        if 'High' in fname:
            self.context = [2] * len(self.X)
        elif 'Low' in fname:
            self.context = [1] * len(self.X)
        elif 'resting' in fname:
            self.context = [0] * len(self.X)
        else:
            self.context = self.data['context']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        # return 33
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])
        context = to_float_tensor([self.context[index]])
        eeg = sample[0:32]
        muscle = sample[32:]
        label = to_long_tensor([self.y[index]])  # careful

        return eeg, muscle, context, label


class McannDataset(Dataset):
    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname).item()

        self.X = self.data['X']
        self.y = self.data['y']

        if 'High' in fname:
            self.context = [2] * len(self.X)
        elif 'Low' in fname:
            self.context = [1] * len(self.X)
        elif 'resting' in fname:
            self.context = [0] * len(self.X)
        else:
            self.context = self.data['context']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        # return 33
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])

        context = to_float_tensor([self.context[index]])

        eeg = sample[0:32]
        muscle = sample[32:]
        label = to_long_tensor([self.y[index]])  # careful

        return eeg, muscle, context, label


class McannDatasetEegEmg(Dataset):
    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname).item()

        self.X = self.data['X']
        self.y = self.data['y']

        if 'High' in fname:
            self.context = [3] * len(self.X)
        elif 'Low' in fname:
            self.context = [2] * len(self.X)
        elif 'resting' in fname:
            self.context = [1] * len(self.X)
        else:
            self.context = self.data['context']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        # return 33
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])

        context = to_float_tensor([self.context[index]])

        label = to_long_tensor([self.y[index]])  # careful

        return sample, context, label


class KaggleDataset(Dataset):
    def __init__(self, fname):

        super().__init__()

        self.data = np.load(fname).item()

        self.X = self.data['X']
        self.y = self.data['y']

        print("loaded dataset %s" % fname)
        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        # return 33
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])

        context = to_float_tensor([0])

        eeg = sample[0:56]
        muscle = sample[56:]
        label = to_long_tensor([self.y[index]])  # careful

        return eeg, muscle, context, label


class CVDataset(Dataset):
    def __init__(self, data):

        super().__init__()

        self.data = data

        self.X = self.data['X']
        self.y = self.data['y']

        print("total size %d" % len(self.data))
        print("input shape: %s" % str(self.X[0].shape))

    def __len__(self):
        # return 33
        return len(self.X)

    def __getitem__(self, index):
        # If mode is single_tw, sample is of size [3, 32, 32]; if mode is multiple_tw, sample is of size [7, 3, 32, 32].
        sample = to_float_tensor(self.X[index])

        context = to_float_tensor([0])

        eeg = sample[0:56]
        muscle = sample[56:]
        label = to_long_tensor([self.y[index]])  # careful

        return eeg, muscle, context, label
