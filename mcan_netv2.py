import torch.nn as nn
import torch
from torch.nn.functional import elu
import torch.nn.functional as F
from utils import calculate_same_padding
import sys

from train_utils import *


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                        maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

# Implementation of EEGNet.
# TODO: Test and train this model.

# Possible evaluation parameters
# all, emg, state, eeg


class McannV2(nn.Module):

    def __init__(self, T, F1, D, num_classes=3, dropout=0.25, channels=32,
                 evaluation_param="all", add_noise=True,
                 conditional_channels=8, is_kaggle=False, is_bci=False, noise=1.0):

        super().__init__()

        self.evaluation_param = evaluation_param
        self.add_noise = add_noise
        self.is_kaggle = is_kaggle
        self.is_bci = is_bci

        self.noise = noise

        self.use_gpu = torch.cuda.is_available(
        ) and os.environ['USE_CUDA'] == 'True'
        self.kernel_length = 10

        self.conditional_channels = conditional_channels

        self.channels = channels

        self.latent_dims = 64

        if is_bci:
            self.T = 384
            self.channels = 22
        elif is_kaggle:
            self.T = 160
            self.channels = 56
        else:
            self.T = T

        self.F1 = 10
        self.D = D
        self.F2 = int(F1 * D)
        F2 = self.F1 * D

        self.num_classes = num_classes

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        # Shared filters (EMG, EOG, EEG)
        self.conv_1 = nn.Conv2d(
            1, self.F1, (1, self.kernel_length), stride=1, bias=False,
            padding=(0, self.kernel_length // 2))

        self.pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.batch_norm_1 = nn.BatchNorm2d(
            F1, momentum=0.01, affine=True, eps=1e-3
        )

        self.conv_2 = nn.Conv2d(
            self.F1, self.F1, (1, self.kernel_length), stride=(1, 1), bias=False, groups=self.F1,
            padding=(0, self.kernel_length // 2)
        )
        self.batch_norm_2 = nn.BatchNorm2d(
            self.F1, momentum=0.01, affine=True, eps=1e-3
        )

        # EEG

        self.depth_conv_1 = Conv2dWithConstraint(
            self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, bias=False,
            groups=self.F1,
            padding=(0, 0))

        # context
        self.cond_conv_2 = nn.Conv2d(
            self.F1, self.F1, (1, self.kernel_length), stride=(1, 1), bias=False, groups=self.F1,
            padding=(0, self.kernel_length // 2)
        )
        self.depth_cond_conv_1 = Conv2dWithConstraint(
            self.F1, self.F1 * self.D, (self.conditional_channels, 1), max_norm=1, stride=1, bias=False,
            groups=self.F1,
            padding=(0, 0))

        self.batch_norm_cond = nn.BatchNorm2d(
            self.F1, momentum=0.01, affine=True, eps=1e-3)

        # FUSION
        # Method 1: Fuse using Linear layer + 1 for State information
        if self.evaluation_param == "all":
            self.fuse = nn.Linear(self.D * self.F1 * 2 + 1, self.D * self.F1)
        elif self.evaluation_param == "emg":
            self.fuse = nn.Linear(self.D * self.F1 * 2, self.D * self.F1)
        elif self.evaluation_param == "state":
            self.fuse = nn.Linear(self.D * self.F1 + 1, self.D * self.F1)
        elif self.evaluation_param == "eeg":
            self.fuse = nn.Linear(self.D * self.F1, self.D * self.F1)

        # Method 2: Add
        self.fuse_norm = nn.BatchNorm2d(
            self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)

        self.fuse_pool = nn.AvgPool2d(
            kernel_size=(1, 4), stride=(1, 4))

        # Block 2
        padding_h, padding_w = calculate_same_padding(1, T / 4, 1, 16)
        # +1 or not depends on even/odd
        self.separate_conv_layer_1 = nn.Conv2d(
            self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1, bias=False, groups=self.F1 * self.D,
            padding=(0, 16 // 2))

        self.separate_conv_layer_2 = nn.Conv2d(
            self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False,
            padding=(0, 0))

        self.batch_norm_3 = nn.BatchNorm2d(
            self.F2, momentum=0.01, affine=True, eps=1e-3)

        self.avg_pool_2 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8))

        # merge signals time wise
        if self.is_kaggle:
            self.depth_merge = nn.Linear(100, self.latent_dims)
        elif self.is_bci:
            self.depth_merge = nn.Linear(240, self.latent_dims)
        elif self.T == 512: # noisy dataset
            self.depth_merge = nn.Linear(320, self.latent_dims)
        elif self.T == 256:
            self.depth_merge = nn.Linear(160, self.latent_dims)
        elif self.T == 128:
            self.depth_merge = nn.Linear(80, self.latent_dims)
        elif self.T == 64:
            self.latent_dims = min(40, self.latent_dims)
            self.depth_merge = nn.Linear(40, self.latent_dims)
        else:
            self.depth_merge = nn.Linear(220, self.latent_dims)



        # TODO: Tensor cross product
        # self.classify = nn.Linear(F2 * 1 * (T // 32), self.num_classes)
        self.classify = nn.Linear(self.latent_dims, self.num_classes)

        # self.depth_merge = nn.Linear(32, self.latent_dims)
        # self.classify = nn.Linear(32, self.num_classes)

        # Decoder network
        self.un_merge = nn.Linear(self.latent_dims, F2 * 1 * (T // 32))

        self.unpool2 = nn.Linear(F2 * 1 * (T // 32), F2 * 1 * (T // 32) * 8)

        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.separable_deconv2 = nn.ConvTranspose2d(F2, F2, (1, 1))
        padding_h, padding_w = calculate_same_padding(1, T / 4, 1, 16)
        self.separable_deconv1 = nn.ConvTranspose2d(
            F2, F2, (1, 16), padding=(0, padding_w))

        if self.is_kaggle:
            self.unpool1 = nn.Linear(4 * (T//16), 16 * T//16 + 1)
        else:
            self.unpool1 = nn.Linear(4 * (T//16) + 1, 16 * T//16 + 1)

        self.depth_deconv1 = nn.ConvTranspose2d(
            self.F1*D, self.F1, (self.channels, 1), padding=0, groups=self.F1)
        self.deconv1 = nn.ConvTranspose2d(
            self.F1, 1, (1, self.kernel_length // 2), padding=(0, self.kernel_length // 4))

        self.deconv2 = nn.ConvTranspose2d(
            1, 1, (1, self.kernel_length), padding=(0, self.kernel_length // 2 - 1))

    def ctx_encoder(self, x):
        batch_sz = x.size(0)
        input_size = x.size()

        # ## add noise mean 0 variance 1
        # if self.training:
        #     x = x + to_float_tensor(torch.randn(x.size()) * 1)

        x = x.view(batch_sz, 1, self.conditional_channels, self.T)

        x = self.conv_1(x)  # temporal
        x = self.batch_norm_2(x)
        # x = self.cond_conv_2(x)
        x = self.cond_conv_2(x)
        x = self.pool(F.elu(x))
        # Input: (N, F1, C, T), Output: (N, D * F1, 1, T)
        x = self.depth_cond_conv_1(x)  # depth for sensor information

        return x

    def eeg_encoder(self, x):
        batch_sz = x.size(0)
        input_size = x.size()

        if self.training and self.add_noise:
            x = x + to_float_tensor(torch.randn(x.size()) * self.noise)

        x = x.view(batch_sz, 1, self.channels, self.T)

        x = self.conv_1(x)  # temporal
        x = self.batch_norm_2(x)
        x = self.conv_2(x)
        x = self.pool(F.elu(x))
        # Input: (N, F1, C, T), Output: (N, D * F1, 1, T)
        x = self.depth_conv_1(x)  # depth for sensor information

        return x

    def forward(self, x, muscle, user_context):
        batch_sz = x.size(0)
        input_size = x.size()

        eeg = self.eeg_encoder(x)

        if self.evaluation_param == "all":
            muscle = self.ctx_encoder(muscle)
            user_context = user_context.view(
                batch_sz, 1, 1, 1).repeat(1, 1, 1, eeg.size(-1))
            x = torch.cat([eeg, muscle, user_context], 1).squeeze(2)
        elif self.evaluation_param == "emg":
            muscle = self.ctx_encoder(muscle)
            x = torch.cat([eeg, muscle], 1).squeeze(2)
        elif self.evaluation_param == "state":
            user_context = user_context.view(
                batch_sz, 1, 1, 1).repeat(1, 1, 1, eeg.size(-1))
            x = torch.cat([eeg, user_context], 1).squeeze(2)
        elif self.evaluation_param == "eeg":
            x = torch.cat([eeg], 1).squeeze(2)

        x = F.elu(x)
        x = torch.transpose(x, 2, 1)

        x = self.fuse(x)
        x = torch.transpose(x, 1, 2).unsqueeze(2)
        x = F.elu(x)
        x = self.dropout(x)

        # block 2 of eegnet
        x = self.separate_conv_layer_1(x)
        x = self.separate_conv_layer_2(x)
        # x = self.batch_norm_3(x)
        x = F.elu(x)
        x = self.avg_pool_2(x)
        # x = F.dropout(x)
        x = self.dropout(x)
        latent = F.elu(self.depth_merge(x.view(batch_sz, -1)))

        probs = self.classify(latent).view(batch_sz, self.num_classes)

        return probs, latent

    def decode(self, latent, input_size):
        batch_sz = input_size[0]
        # x = latent.view(batch_sz, -1, self.latent_dims)

        x = F.elu(self.un_merge(latent))
        # print(x.size())

        x = F.elu(self.unpool2(x))

        # print(x.size())
        # x = self.batch_norm3(x.view(batch_sz, ))
        x = F.elu(self.separable_deconv2(x.view(batch_sz, self.F2, 1, -1)))
        # print(x.size())
        x = self.separable_deconv1(x)
        # print(x.size())
        x = F.elu(self.unpool1(x))
        # print(x.size())
        x = self.depth_deconv1(x)
        # print(x.size())
        x = F.elu(self.deconv1(x))
        reconstruction = self.deconv2(x).view(batch_sz, self.channels, -1)
        reconstruction = reconstruction[:, :, :input_size[-1]]
        # print(reconstruction.size())

        return reconstruction
