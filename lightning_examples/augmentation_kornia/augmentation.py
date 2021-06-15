# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="x83-rnVKT8Wo"
# # GPU and batched data augmentation with Kornia and PyTorch-Lightning
#
# **NOTE:** Adaptation of the original post found in [Kornia tutorials](https://kornia-tutorials.readthedocs.io/en/latest/).

# %% [markdown] colab_type="text" id="iCsre0XmawoR"
# ## Setup
# We first need to install Kornia and PyTorch Lightning and for convenience also torchmetrics.

# %% colab={} colab_type="code" id="dEeUzX_5aLrX"
# ! pip install kornia pytorch_lightning torchmetrics -qU

# %% colab={} colab_type="code" id="Z_XTj-y1gYJL"
import os

import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

AVAIL_GPUS = min(1, torch.cuda.device_count())

# %% [markdown] colab_type="text" id="hA4-AFd6gKo-"
# ## Define Data Augmentations module
#
# [Kornia.org](https://www.kornia.org) is low level Computer Vision library that provides a dedicated module
# [`kornia.augmentation`](https://kornia.readthedocs.io/en/latest/augmentation.html) module implementing
# en extensive set of data augmentation techniques for image and video.
#
# Similar to Lightning, in Kornia it's promoted to encapsulate functionalities inside classes for readability
# and efficiency purposes. In this case, we define a data augmentaton pipeline subclassing a `nn.Module`
# where the augmentation_kornia (also subclassing `nn.Module`) are combined with other PyTorch components
# such as `nn.Sequential`.
#
# Checkout the different augmentation operators in Kornia docs and experiment yourself !


# %% colab={} colab_type="code" id="RvdMAbyXgRPh"
class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.75),
            K.augmentation.RandomChannelShuffle(p=0.75),
            K.augmentation.RandomThinPlateSpline(p=0.75),
        )

        self.jitter = K.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out


# %% [markdown] colab_type="text" id="CQ06HnPtkIlq"
# ## Define a Pre-processing module
#
# In addition to the `DataAugmentation` modudle that will sample random parameters during the training stage,
# we define a `Preprocess` class to handle the conversion of the image type to properly work with `torch.Tensor`.
#
# For this example we use `torchvision` CIFAR10 which return samples of `PIL.Image`, however,
# to take all the advantages of PyTorch and Kornia we need to cast the images into tensors.
#
# To do that we will use `kornia.image_to_tensor` which casts and permutes the images in the right format.


# %% colab={} colab_type="code" id="EjPPQjcXkNT3"
class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.


# %% [markdown] colab_type="text" id="LXGKqIZuTLMs"
# ## Define PyTorch Lightning model
#
# The next step is to define our `LightningModule` to have a proper organisation of our training pipeline.
# This is a simple example just to show how to structure your baseline to be used as a reference,
# do not expect a high performance.
#
# Notice that the `Preprocess` class is injected into the dataset and will be applied per sample.
#
# The interesting part in the proposed approach happens inside the `training_step` where with just a single
# line of code we apply the data augmentation in batch and no need to worry about the device.
# This means that our `DataAugmentation` pipeline will automatically executed in the GPU.


# %% colab={} colab_type="code" id="aDagOcKyZ_qh"
class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model: expereiment yourself
        self.model = torchvision.models.resnet18(pretrained=True)

        self.preprocess = Preprocess()  # per sample transforms

        self.transform = DataAugmentation()  # per batch augmentation_kornia

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return F.softmax(self.model(x))

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def show_batch(self, win_size=(10, 10)):

        def _to_vis(data):
            return K.utils.tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        # get a batch from the training set: try with `val_datlaoader` :)
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply transforms
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_aug = self.transform(x)  # => we perform GPU/Batched data augmentation
        y_hat = self(x_aug)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", self.accuracy(y_hat, y), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]

    def prepare_data(self):
        CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)

    def train_dataloader(self):
        dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=32)
        return loader

    def val_dataloader(self):
        dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=32)
        return loader


# %% [markdown] colab_type="text" id="yyLkgogITNtj"
# ## Visualize images

# %% colab={} colab_type="code" id="kr8cql-aaKnC"
# init model
model = CoolSystem()

# %% colab={} colab_type="code" id="O2Fq4bK4l9sS"
model.show_batch(win_size=(14, 14))

# %% [markdown] colab_type="text" id="baSvOQRNl5iw"
# ## Run training

# %% colab={} colab_type="code" id="O_AFCY3WlwKg"
# Initialize a trainer
trainer = Trainer(
    progress_bar_refresh_rate=20,
    gpus=AVAIL_GPUS,
    max_epochs=10,
    logger=pl.loggers.CSVLogger(save_dir='logs/', name="cifar10-resnet18")
)

# Train the model ⚡
trainer.fit(model)

# %% [markdown] colab_type="text" id="jrSNajrhx9E_"
# ### Visualize the training results

# %% colab={} colab_type="code" id="jXhH_0mbx_ye"
metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
print(metrics.head())

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[['train_loss', 'valid_loss']].plot(grid=True, legend=True)
df_metrics[['valid_acc', 'train_acc']].plot(grid=True, legend=True)

# %% [markdown] colab_type="text" id="t65M0xD4lgAh"
# ## Tensorboard

# %% colab={} colab_type="code" id="xg3c6UJQt0mw"
# Start tensorboard.
# # %load_ext tensorboard
# # %tensorboard --logdir lightning_logs/
