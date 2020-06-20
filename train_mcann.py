from multiprocessing import Pool, Process
from data.eegnet_dataset import *
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from train_utils import *
from models.mcan_netv2 import McannV2
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys

import argparse

sys.path.append('../src')


HOME_DIR = os.path.expanduser("~/.esac/eeg-auth")


# Epochs of patience
BATCH_SIZE = 256


class TwoClassDataset(Dataset):
    """wrapper for 2 class dataset"""

    def __init__(self, ds):
        super().__init__()

        self.ds = ds

    def __len__(self):
        # return 100
        return len(self.ds)

    def __getitem__(self, index):
        sample, muscle, label = self.ds[index]

        if label == 2:
            label = to_long_tensor([1])

        return sample, muscle, label


def reconstruction_weight(curr_epoch, max_epoch=5, max_w=1e-2, min_w=1e-5):
    # anneal over epochs (Linear)
    decr = ((max_w - min_w) / float(max_epoch))

    return max_w - float(min(curr_epoch, max_epoch)) * decr


def train_helper(model, optimizer, train_ds, valid_ds, name, num_classes=3,
                 max_epochs=200, cuda_dev_id=0, patience=20,
                 start_epoch=0,
                 model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/tom/"):
    print("----->")
    print("----->")
    print("----->")
    print("-----> Running trial: " + name)
    print("----->")
    print("----->")

    print(model)
    best_model_path = ""
    print("Num parameters: " + str(get_num_params(model)))

    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    use_gpu = torch.cuda.is_available() and os.environ['USE_CUDA'] == 'True'

    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("USE CUDA: " + str(use_gpu))

    if use_gpu:
        print("Using Cuda!")
        model = model.cuda()
        # optimizer = optimizer.cuda()
        criterion = criterion.cuda()
        mse_loss = mse_loss.cuda()

    early_stopping = EarlyStopping(patience=patience)

    best_epoch = 0

    since = time.time()

    val_losses = []
    val_accs = []

    best_acc = 0.0

    for epoch in range(start_epoch, max_epochs):

        print("Epoch {}/{}".format(epoch + 1, max_epochs))
        print("-" * 10)

        recon_weight = reconstruction_weight(epoch)
        print("Reconstruction weight: %f" % recon_weight)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(
            valid_ds, batch_size=BATCH_SIZE, shuffle=True)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        recon_losses = []
        recon_valid_losses = []

        ce_losses = []
        ce_valid_losses = []

        # Training
        model.train()
        num_correct = 0
        reconstruction = 0

        for eeg, muscle, context, labels in train_loader:
            # print(eeg.size())
            # print(muscle.size())
            labels = labels.view(eeg.size(0))

            if use_gpu:
                eeg = eeg.cuda()
                muscle = muscle.cuda()
                labels = labels.cuda()
                context = context.cuda()

            optimizer.zero_grad()

            probs, latent = model(eeg, muscle, context)
            ce_loss = criterion(probs, labels)
            ce_losses.append(ce_loss.item())

            reconstruction = model.decode(latent, eeg.size())
            recon_loss = recon_weight * mse_loss(reconstruction, eeg)
            recon_losses.append(recon_loss.item())

            loss = ce_loss + recon_loss
            loss = loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            train_losses.append(loss.item())

            # training statistics
            _, predicted = torch.max(probs.data, 1)
            num_correct += (predicted == labels).sum().item()

        train_loss = np.average(train_losses)
        train_acc = num_correct / float(len(train_ds))

        # Validation
        model.eval()
        num_correct = 0
        valid_loss = 0.0
        reconstruction = 0

        for eeg, muscle, context, labels in valid_loader:
            labels = labels.view(eeg.size(0))

            if use_gpu:
                eeg = eeg.cuda()
                muscle = muscle.cuda()
                labels = labels.cuda()
                context = context.cuda()

            probs, latent = model(eeg, muscle, context)
            ce_loss = criterion(probs, labels)
            ce_valid_losses.append(ce_loss.item())

            reconstruction = model.decode(latent, eeg.size())
            recon_loss = recon_weight * mse_loss(reconstruction, eeg)
            recon_valid_losses.append(recon_loss.item())

            loss = ce_loss + recon_loss
            loss = loss

            valid_losses.append(loss.item())

            _, predicted = torch.max(probs.data, 1)
            num_correct += (predicted == labels).sum().item()

        valid_loss = np.average(valid_losses)
        valid_acc = num_correct / float(len(valid_ds))

        epoch_len = len(str(max_epochs))

        print_msg = (f'[{epoch:>{5}}/{max_epochs:>{5}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        print(f'           ' +
              f'train_acc: {train_acc:.5f} ' +
              f'valid_acc: {valid_acc:.5f}')

        # print breakdown of epoch losses
        class_loss = np.average(ce_losses) / float(BATCH_SIZE)
        recon_loss = np.average(recon_losses) / float(BATCH_SIZE)
        print(f'           ' +
              f'train_class_loss: {class_loss:.5f} ' +
              f'train_recon_loss: {recon_loss:.5f}')

        # print breakdown of epoch losses
        class_loss = np.average(ce_valid_losses) / float(BATCH_SIZE)
        recon_loss = np.average(recon_valid_losses) / float(BATCH_SIZE)
        print(f'           ' +
              f'valid_class_loss: {class_loss:.5f} ' +
              f'valid_recon_loss: {recon_loss:.5f}')

        print("    Weights: " + str(get_param_norms(model.parameters())))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        recon_losses = []
        recon_valid_losses = []

        ce_losses = []
        ce_valid_losses = []

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_model_path = model_output_dir + "/%s_best" % name

            save_model(best_model_path, model, optimizer, epoch, valid_loss)

        epoch_model_path = model_output_dir + "/%s_epoch_%d" % (name, epoch)
        save_model(epoch_model_path, model, optimizer, epoch, valid_loss)

        print("\n")

        early_stopping(-valid_acc, model)

        if early_stopping.early_stop:
            print("Early Stopping.")
            break

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: %d" % best_epoch)

    final_model_path = model_output_dir + "/%s_final" % name
    save_model(final_model_path, model, optimizer, max_epochs, 0)

    return best_model_path


def eval_helper(model, optimizer, train_ds, valid_ds, name, num_classes=3,
                max_epochs=200, cuda_dev_id=0, patience=20,
                start_epoch=0,
                model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/tom/"):
    print("----->")
    print("----->")
    print("----->")
    print("-----> Running trial: " + name)
    print("----->")
    print("----->")

    print(model)
    best_model_path = ""
    print("Num parameters: " + str(get_num_params(model)))

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    # criterion = nn.CrossEntropyLoss(weight=to_float_tensor([8., 8., 1.]))
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    use_gpu = torch.cuda.is_available() and os.environ['USE_CUDA'] == 'True'

    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("USE CUDA: " + str(use_gpu))

    if use_gpu:
        print("Using Cuda!")
        model = model.cuda()
        # optimizer = optimizer.cuda()
        criterion = criterion.cuda()
        mse_loss = mse_loss.cuda()

    best_epoch = 0

    since = time.time()

    val_losses = []
    val_accs = []

    best_acc = 0.0

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    recon_losses = []
    recon_valid_losses = []

    ce_losses = []
    ce_valid_losses = []

    # Training
    model.train()
    num_correct = 0
    reconstruction = 0

    # Validation
    model.eval()
    num_correct = 0
    valid_loss = 0.0
    reconstruction = 0

    for eeg, muscle, context, labels in valid_loader:
        labels = labels.view(eeg.size(0))

        if use_gpu:
            eeg = eeg.cuda()
            muscle = muscle.cuda()
            labels = labels.cuda()
            context = context.cuda()

        probs, latent = model(eeg, muscle, context)
        ce_loss = criterion(probs, labels)
        ce_valid_losses.append(ce_loss.item())

        reconstruction = model.decode(latent, eeg.size())
        recon_loss = recon_weight * mse_loss(reconstruction, eeg)
        recon_valid_losses.append(recon_loss.item())

        loss = ce_loss + recon_loss
        loss = loss

        valid_losses.append(loss.item())

        _, predicted = torch.max(probs.data, 1)
        num_correct += (predicted == labels).sum().item()

    valid_loss = np.average(valid_losses)
    valid_acc = num_correct / float(len(valid_ds))

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

    return valid_acc


def train_helper_no_recon(model, optimizer, train_ds, valid_ds, name, num_classes=3,
                          max_epochs=200, cuda_dev_id=0, patience=20,
                          start_epoch=0,
                          model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/tom/"):
    print("----->")
    print("----->")
    print("----->")
    print("-----> Running trial: " + name)
    print("----->")
    print("----->")

    print(model)
    print("Num parameters: " + str(get_num_params(model)))

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    # criterion = nn.CrossEntropyLoss(weight=to_float_tensor([8., 8., 1.]))
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    use_gpu = torch.cuda.is_available() and os.environ['USE_CUDA'] == 'True'

    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("USE CUDA: " + str(use_gpu))

    if use_gpu:
        print("Using Cuda!")
        model = model.cuda()
        # optimizer = optimizer.cuda()
        criterion = criterion.cuda()
        mse_loss = mse_loss.cuda()

    early_stopping = EarlyStopping(patience=patience)

    best_epoch = 0
    best_model_path = ""

    since = time.time()

    val_losses = []
    val_accs = []

    best_acc = 0.0

    for epoch in range(start_epoch, max_epochs):

        print("Epoch {}/{}".format(epoch + 1, max_epochs))
        print("-" * 10)

        recon_weight = reconstruction_weight(epoch)
        print("Reconstruction weight: %f" % recon_weight)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(
            valid_ds, batch_size=BATCH_SIZE, shuffle=True)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        recon_losses = []
        recon_valid_losses = []

        ce_losses = []
        ce_valid_losses = []

        # Training
        model.train()
        num_correct = 0
        reconstruction = 0

        for eeg, muscle, context, labels in train_loader:
            # print(eeg.size())
            # print(muscle.size())
            labels = labels.view(eeg.size(0))

            if use_gpu:
                eeg = eeg.cuda()
                muscle = muscle.cuda()
                labels = labels.cuda()
                context = context.cuda()

            optimizer.zero_grad()

            probs, latent = model(eeg, muscle, context)
            ce_loss = criterion(probs, labels)
            ce_losses.append(ce_loss.item())

            # reconstruction = model.decode(latent, eeg.size())
            # recon_loss = recon_weight * mse_loss(reconstruction, eeg)
            # recon_losses.append(recon_loss.item())

            loss = ce_loss  # + recon_loss
            loss = loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            train_losses.append(loss.item())

            # training statistics
            _, predicted = torch.max(probs.data, 1)
            num_correct += (predicted == labels).sum().item()

        train_loss = np.average(train_losses)
        train_acc = num_correct / float(len(train_ds))

        # Validation
        model.eval()
        num_correct = 0
        valid_loss = 0.0
        reconstruction = 0

        for eeg, muscle, context, labels in valid_loader:
            labels = labels.view(eeg.size(0))

            if use_gpu:
                eeg = eeg.cuda()
                muscle = muscle.cuda()
                labels = labels.cuda()
                context = context.cuda()

            probs, latent = model(eeg, muscle, context)
            ce_loss = criterion(probs, labels)
            ce_valid_losses.append(ce_loss.item())

            # reconstruction = model.decode(latent, eeg.size())
            # recon_loss = recon_weight * mse_loss(reconstruction, eeg)
            # recon_valid_losses.append(recon_loss.item())

            loss = ce_loss  # + recon_loss
            loss = loss

            valid_losses.append(loss.item())

            _, predicted = torch.max(probs.data, 1)
            num_correct += (predicted == labels).sum().item()

        valid_loss = np.average(valid_losses)
        valid_acc = num_correct / float(len(valid_ds))

        epoch_len = len(str(max_epochs))

        print_msg = (f'[{epoch:>{5}}/{max_epochs:>{5}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        print(f'           ' +
              f'train_acc: {train_acc:.5f} ' +
              f'valid_acc: {valid_acc:.5f}')

        # print breakdown of epoch losses
        class_loss = np.average(ce_losses) / float(BATCH_SIZE)
        recon_loss = np.average(recon_losses) / float(BATCH_SIZE)
        print(f'           ' +
              f'train_class_loss: {class_loss:.5f} ' +
              f'train_recon_loss: {recon_loss:.5f}')

        # print breakdown of epoch losses
        class_loss = np.average(ce_valid_losses) / float(BATCH_SIZE)
        recon_loss = np.average(recon_valid_losses) / float(BATCH_SIZE)
        print(f'           ' +
              f'valid_class_loss: {class_loss:.5f} ' +
              f'valid_recon_loss: {recon_loss:.5f}')

        print("    Weights: " + str(get_param_norms(model.parameters())))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        recon_losses = []
        recon_valid_losses = []

        ce_losses = []
        ce_valid_losses = []

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_model_path = model_output_dir + "/%s_best" % name

            save_model(best_model_path, model, optimizer, epoch, valid_loss)

        epoch_model_path = model_output_dir + "/%s_epoch_%d" % (name, epoch)
        save_model(epoch_model_path, model, optimizer, epoch, valid_loss)

        print("\n")

        early_stopping(-valid_acc, model)

        if early_stopping.early_stop:
            print("Early Stopping.")
            break

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: %d" % best_epoch)

    final_model_path = model_output_dir + "/%s_final" % name
    save_model(final_model_path, model, optimizer, max_epochs, 0)

    return best_model_path


def eval_helper_no_recon(model, optimizer, train_ds, valid_ds, name, num_classes=3,
                         max_epochs=200, cuda_dev_id=0, patience=20,
                         start_epoch=0,
                         model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/tom/"):
    print("----->")
    print("----->")
    print("----->")
    print("-----> Running trial: " + name)
    print("----->")
    print("----->")

    print(model)
    print("Num parameters: " + str(get_num_params(model)))

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    # criterion = nn.CrossEntropyLoss(weight=to_float_tensor([8., 8., 1.]))
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    use_gpu = torch.cuda.is_available() and os.environ['USE_CUDA'] == 'True'

    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("USE CUDA: " + str(use_gpu))

    if use_gpu:
        print("Using Cuda!")
        model = model.cuda()
        # optimizer = optimizer.cuda()
        criterion = criterion.cuda()
        mse_loss = mse_loss.cuda()

    best_epoch = 0

    since = time.time()

    val_losses = []
    val_accs = []

    best_acc = 0.0
    reconstruction_weight = 0.0

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    recon_losses = []
    recon_valid_losses = []

    ce_losses = []
    ce_valid_losses = []

    # Training
    model.train()
    num_correct = 0
    reconstruction = 0

    # Validation
    model.eval()
    num_correct = 0
    valid_loss = 0.0
    reconstruction = 0

    for eeg, muscle, context, labels in valid_loader:
        labels = labels.view(eeg.size(0))

        if use_gpu:
            eeg = eeg.cuda()
            muscle = muscle.cuda()
            labels = labels.cuda()
            context = context.cuda()

        probs, latent = model(eeg, muscle, context)
        ce_loss = criterion(probs, labels)
        ce_valid_losses.append(ce_loss.item())

        loss = ce_loss
        loss = loss

        valid_losses.append(loss.item())

        _, predicted = torch.max(probs.data, 1)
        num_correct += (predicted == labels).sum().item()

    valid_loss = np.average(valid_losses)
    valid_acc = num_correct / float(len(valid_ds))

    return valid_acc


def main(name,
         num_classes=3,
         num_epochs=200,
         patience=200,
         cross_validate=False,
         exp_id=0,
         tom_ds_dir=HOME_DIR + "/data_models/eeg_classification/processed/noisy_bike_2hz/",
         evaluation_param="all",
         model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/"):

    trial_name = name + "_" + evaluation_param
    if cross_validate:
        trial_name += "_cv_" + str(exp_id)

    model_output_dir = os.path.join(model_output_dir, trial_name)

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    T = 512
    if "denoised" in tom_ds_dir:
        T = 128

    if "2hz" in tom_ds_dir:
        T = T // 2

    model = McannV2(T, 10, 2, num_classes=num_classes,
                    evaluation_param=evaluation_param)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    # datasets
    tom_train_fname = os.path.join(tom_ds_dir, "within.all.subj.train.npy")
    # tom_train_fname = os.path.join(
    #     tom_ds_dir, "sj04bikeLowRsvp1Hz.mat.train.npy")
    tom_test_fname = os.path.join(tom_ds_dir, "within.all.subj.test.npy")

    train_ds = McannDataset(tom_train_fname)

    if cross_validate:

        train_ds, valid_ds = train_valid_split(
            train_ds, split_fold=5, random_seed=exp_id)

        train_helper(model, optimizer, train_ds, valid_ds, trial_name,
                     max_epochs=num_epochs,
                     patience=patience,
                     model_output_dir=model_output_dir)
    else:
        test_ds = McannDataset(tom_test_fname)

        train_helper(model, optimizer, train_ds, test_ds,
                     trial_name,
                     max_epochs=num_epochs,
                     patience=patience,
                     model_output_dir=model_output_dir)


def main_cs(name,
            num_classes=3,
            num_epochs=200,
            patience=200,
            cross_validate=False,
            exp_id=0,
            tom_ds_dir=HOME_DIR + "/data_models/eeg_classification/processed/noisy_bike_2hz/",
            evaluation_param="all",
            model_output_dir=HOME_DIR + "/data_models/eeg_classification/models/"):

    root_dir = model_output_dir

    for i in range(12):
        trial_name = name + "_fold_" + str(i)

        model_output_dir = os.path.join(root_dir, trial_name)

        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        # 2hz, noisy only
        T = 512
        if "denoised" in tom_ds_dir:
            T = 128

        if "2hz" in tom_ds_dir:
            T = T // 2

        model = McannV2(T, 10, 2, num_classes=num_classes,
                        evaluation_param=evaluation_param)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        # datasets
        tom_train_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.train.npy" % i)
        tom_validate_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.valid.npy" % i)
        tom_test_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.test.npy" % i)

        train_ds = McannDataset(tom_train_fname)
        valid_ds = McannDataset(tom_validate_fname)
        #test_ds = McannDataset(tom_test_fname)

        train_helper(model, optimizer, train_ds, valid_ds, trial_name,
                     max_epochs=num_epochs,
                     patience=patience,
                     model_output_dir=model_output_dir)


def run_all_noisy_2hz():
    tom_ds_dir = HOME_DIR + "/data_models/eeg_classification/processed/noisy_bike_2hz/"

    main("mcann_2hz_noisy", evaluation_param="all", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="emg", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="eeg", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="state", tom_ds_dir=tom_ds_dir)

    # cross validation for early stopping
    main("mcann_2hz_noisy", evaluation_param="all",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="emg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="eeg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_noisy", evaluation_param="state",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)


def run_all_2hz():
    tom_ds_dir = HOME_DIR + "/data_models/eeg_classification/processed/denoised_bike_2hz/"

    main("mcann_2hz_denoised", evaluation_param="all", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="emg", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="eeg", tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="state", tom_ds_dir=tom_ds_dir)

    # cross validation for early stopping
    main("mcann_2hz_denoised", evaluation_param="all",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="emg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="eeg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_2hz_denoised", evaluation_param="state",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)


def run_all_1hz():
    tom_ds_dir = HOME_DIR + "/data_models/eeg_classification/processed/denoised_bike_1hz/"

    main("mcann_1hz_denoised", evaluation_param="all", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="emg", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="eeg", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="state", tom_ds_dir=tom_ds_dir)

    # cross validation for early stopping
    main("mcann_1hz_denoised", evaluation_param="all",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="emg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="eeg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_denoised", evaluation_param="state",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)


def run_all_noisy_1hz():
    tom_ds_dir = HOME_DIR + "/data_models/eeg_classification/processed/noisy_bike_1hz/"

    main("mcann_1hz_noisy", evaluation_param="all", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="emg", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="eeg", tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="state", tom_ds_dir=tom_ds_dir)

    # cross validation for early stopping
    main("mcann_1hz_noisy", evaluation_param="all",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="emg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="eeg",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)
    main("mcann_1hz_noisy", evaluation_param="state",
         cross_validate=True,
         tom_ds_dir=tom_ds_dir)


def auth_cs(name,
            num_classes=2,
            num_epochs=200,
            patience=200,
            cross_validate=False,
            exp_id=0,
            tom_ds_dir="/media/scratch/yding/lgfs/eeg_classification/raw/Bike_EEG_Data/Processed_Data_1Hz_Denoised",
            evaluation_param="all",
            model_output_dir=HOME_DIR + "/res/models/"):

    root_dir = model_output_dir

    for i in range(12):
        trial_name = name + "_fold_" + str(i)

        model_output_dir = os.path.join(root_dir, trial_name)

        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        # 2hz, noisy only
        T = 512
        if "denoised" in tom_ds_dir:
            T = 128

        if "2hz" in tom_ds_dir:
            T = T // 2

        model = McannV2(T, 10, 2, num_classes=num_classes,
                        evaluation_param=evaluation_param)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        # datasets
        tom_train_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.train.npy" % i)
        tom_validate_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.valid.npy" % i)
        tom_test_fname = os.path.join(
            tom_ds_dir, "leave.subj.1.fold%d.test.npy" % i)

        train_ds = AuthDataset(tom_train_fname)
        valid_ds = AuthDataset(tom_validate_fname)
        #test_ds = McannDataset(tom_test_fname)

        train_helper(model, optimizer, train_ds, valid_ds, trial_name,
                     max_epochs=num_epochs,
                     patience=patience,
                     model_output_dir=model_output_dir)


if __name__ == "__main__":
    # run_all_2hz()
    # run_all_1hz()
    # run_all_noisy_1hz()
    # run_all_noisy_2hz()
    print('hi')
    # main_cs("mcann_cross_subj_2hz_noisy_50", evaluation_param="all")


# BLA BLA BAL

    # load_model(train_with_model, model, optimizer)

    ########################################################################
    # LOAD training Datsets to memory
    # tom_train_fname = os.path.join(
    #     tom_ds_dir, "sj04bikeHighRsvp1Hz.mat.train.npy")
    # tom_train_fname = os.path.join(tom_ds_dir, "sj04bikeLowRsvp1Hz.mat.train.npy")
    # tom_train_fname = os.path.join(tom_ds_dir, "sj04restingRsvp1Hz.mat.train.npy")

    ### TESTTING DATASETS ########################################################################
    # tom_test_fname = os.path.join(
    #     tom_ds_dir, "sj04bikeHighRsvp1Hz.mat.test.npy")
    # tom_test_fname = os.path.join(tom_ds_dir, "sj04bikeLowRsvp1Hz.mat.test.npy")
    # tom_test_fname = os.path.join(tom_ds_dir, "sj04restingRsvp1Hz.mat.test.npy")

    ################################################################################################

    # cross_subj_training = McannDatasetEegEmg(tom_train_fname)
    # cross_subj_testing = McannDatasetEegEmg(tom_test_fname)
