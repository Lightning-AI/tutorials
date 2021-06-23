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

# %% [markdown] id="WsWdLFMVKqbi"
# <a href="https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/06-tpu-training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="qXO1QLkbRXl0"
# # TPU training with PyTorch Lightning ⚡
#
# In this notebook, we'll train a model on TPUs. Changing one line of code is all you need to that.
#
# The most up to documentation related to TPU training can be found [here](https://pytorch-lightning.readthedocs.io/en/latest/advanced/tpu.html).
#
# ---
#
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)
#   - Ask a question on our [GitHub Discussions](https://github.com/PyTorchLightning/pytorch-lightning/discussions/)

# %% [markdown] id="UmKX0Qa1RaLL"
# ### Setup
#
# Lightning is easy to install. Simply ```pip install pytorch-lightning```

# %% id="vAWOr0FZRaIj"
# ! pip install pytorch-lightning -qU

# %% [markdown] id="zepCr1upT4Z3"
# ###  Install Colab TPU compatible PyTorch/TPU wheels and dependencies

# %% id="AYGWh10lRaF1"
# ! pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

# %% id="SNHa7DpmRZ-C"
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

# %% [markdown] id="rjo1dqzGUxt6"
# ### Defining The `MNISTDataModule`
#
# Below we define `MNISTDataModule`. You can learn more about datamodules in [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) and [datamodule notebook](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/notebooks/02-datamodules.ipynb).


# %% id="pkbrm3YgUxlE"
class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


# %% [markdown] id="nr9AqDWxUxdK"
# ### Defining the `LitModel`
#
# Below, we define the model `LitMNIST`.


# %% id="YKt0KZkOUxVY"
class LitModel(pl.LightningModule):

    def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * width * height, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# %% [markdown] id="Uxl88z06cHyV"
# ### TPU Training
#
# Lightning supports training on a single TPU core or 8 TPU cores.
#
# The Trainer parameters `tpu_cores` defines how many TPU cores to train on (1 or 8) / Single TPU core to train on [1].
#
# For Single TPU training, Just pass the TPU core ID [1-8] in a list. Setting `tpu_cores=[5]` will train on TPU core ID 5.

# %% [markdown] id="UZ647Xg2gYng"
# Train on TPU core ID 5 with `tpu_cores=[5]`.

# %% id="bzhJ8g_vUxN2"
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.size(), dm.num_classes)
# Init trainer
trainer = pl.Trainer(max_epochs=3, progress_bar_refresh_rate=20, tpu_cores=[5])
# Train
trainer.fit(model, dm)

# %% [markdown] id="slMq_0XBglzC"
# Train on single TPU core with `tpu_cores=1`.

# %% id="31N5Scf2RZ61"
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.size(), dm.num_classes)
# Init trainer
trainer = pl.Trainer(max_epochs=3, progress_bar_refresh_rate=20, tpu_cores=1)
# Train
trainer.fit(model, dm)

# %% [markdown] id="_v8xcU5Sf_Cv"
# Train on 8 TPU cores with `tpu_cores=8`. You might have to restart the notebook to run it on 8 TPU cores after training on single TPU core.

# %% id="EFEw7YpLf-gE"
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.size(), dm.num_classes)
# Init trainer
trainer = pl.Trainer(max_epochs=3, progress_bar_refresh_rate=20, tpu_cores=8)
# Train
trainer.fit(model, dm)

# %% [markdown] id="m2mhgEgpRZ1g"
# <code style="color:#792ee5;">
#     <h1> <strong> Congratulations - Time to Join the Community! </strong>  </h1>
# </code>
#
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the Lightning movement, you can do so in the following ways!
#
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool tools we're building.
#
# * Please, star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
#
# ### Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself and share your interests in `#general` channel
#
# ### Interested by SOTA AI models ! Check out [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# Bolts has a collection of state-of-the-art models, all implemented in [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and can be easily integrated within your own projects.
#
# * Please, star [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
#
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/lightning-bolts) GitHub Issues page and filter for "good first issue".
#
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
#
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
#
# <img src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png?raw=true" width="800" height="200" />
