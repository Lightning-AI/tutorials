# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="HYpMlx7apuHq"
# ### Setup
# Lightning is easy to install. Simply `pip install pytorch-lightning`.
# Also check out [bolts](https://github.com/PyTorchLightning/lightning-bolts/) for pre-existing data modules and models.

# %% id="ziAQCrE-TYWG"
# ! pip install pytorch-lightning lightning-bolts -qU

# %% id="L-W_Gq2FORoU"
# Run this if you intend to use TPUs
# # !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# # !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

# %% id="wjov-2N_TgeS"
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
# %% id="54JMU1N-0y0g"
from torchmetrics.functional import accuracy

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

# %% [markdown] id="FA90qwFcqIXR"
# ### CIFAR10 Data Module
#
# Import the existing data module from `bolts` and modify the train and test transforms.

# %% id="S9e-W8CSa8nH"

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

# %% [markdown] id="SfCsutp3qUMc"
# ### Resnet
# Modify the pre-existing Resnet architecture from TorchVision. The pre-existing architecture is based on ImageNet
# images (224x224) as input. So we need to modify it for CIFAR10 images (32x32).


# %% id="GNSeJgwvhHp-"
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


# %% [markdown] id="HUCj5TKsqty1"
# ### Lightning Module
# Check out the [`configure_optimizers`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers)
# method to use custom Learning Rate schedulers. The OneCycleLR with SGD will get you to around 92-93% accuracy
# in 20-30 epochs and 93-94% accuracy in 40-50 epochs. Feel free to experiment with different
# LR schedules from https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate


# %% id="03OMrBa5iGtT"
class LitResnet(LightningModule):

    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


# %% id="3FFPgpAFi9KU"
model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=30,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger('lightning_logs/', name='resnet'),
    callbacks=[LearningRateMonitor(logging_interval='step')],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

# %% [markdown] id="lWL_WpeVIXWQ"
# ### Bonus: Use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) to get a boost on performance
#
# Use SWA from torch.optim to get a quick performance boost. Also shows a couple of cool features from Lightning:
# - Use `training_epoch_end` to run code after the end of every epoch
# - Use a pretrained model directly with this wrapper for SWA


# %% id="bsSwqKv0t9uY"
class SWAResnet(LitResnet):

    def __init__(self, trained_model, lr=0.01):
        super().__init__()

        self.save_hyperparameters('lr')
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)

    def forward(self, x):
        out = self.swa_model(x)
        return F.log_softmax(out, dim=1)

    def training_epoch_end(self, training_step_outputs):
        self.swa_model.update_parameters(self.model)

    def validation_step(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def on_train_end(self):
        update_bn(self.datamodule.train_dataloader(), self.swa_model, device=self.device)


# %% id="cA6ZG7C74rjL"
swa_model = SWAResnet(model.model, lr=0.01)
swa_model.datamodule = cifar10_dm

swa_trainer = Trainer(
    progress_bar_refresh_rate=20,
    max_epochs=20,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger('lightning_logs/', name='swa_resnet'),
)

swa_trainer.fit(swa_model, cifar10_dm)
swa_trainer.test(swa_model, datamodule=cifar10_dm)

# %% id="RRHMfGiDpZ2M"
# Start tensorboard.
# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/
