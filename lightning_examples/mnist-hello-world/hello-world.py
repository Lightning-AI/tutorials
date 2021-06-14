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
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="2LODD6w9ixlT"
# ### Setup
# Lightning is easy to install. Simply ```pip install pytorch-lightning```

# %% colab={} colab_type="code" id="w4_TYnt_keJi"
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# %% [markdown] colab_type="text" id="EHpyMPKFkVbZ"
# ## Simplest example
#
# Here's the simplest most minimal example with just a training loop (no validation, no testing).
#
# **Keep in Mind** - A `LightningModule` *is* a PyTorch `nn.Module` - it just has a few more helpful features.


# %% colab={} colab_type="code" id="V7ELesz1kVQo"
class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# %% [markdown] colab_type="text" id="hIrtHg-Dv8TJ"
# By using the `Trainer` you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# %% colab={} colab_type="code" id="4Dk6Ykv8lI7X"
# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = pl.Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

# %% [markdown] colab_type="text" id="KNpOoBeIjscS"
# ## A more complete MNIST Lightning Module Example
#
# That wasn't so hard was it?
#
# Now that we've got our feet wet, let's dive in a bit deeper and write a more complete `LightningModule` for MNIST...
#
# This time, we'll bake in all the dataset specific pieces directly in the `LightningModule`.
# This way, we can avoid writing extra code at the beginning of our script every time we want to run it.
#
# ---
#
# ### Note what the following built-in functions are doing:
#
# 1. [prepare_data()](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.prepare_data) 💾
#     - This is where we can download the dataset. We point to our desired dataset and ask torchvision's `MNIST` dataset class to download if the dataset isn't found there.
#     - **Note we do not make any state assignments in this function** (i.e. `self.something = ...`)
#
# 2. [setup(stage)](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning-module.html#setup) ⚙️
#     - Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
#     - Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
#     - If you don't mind loading all your datasets at once, you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever `None` is passed to `stage` (or ignore it altogether and exclude any conditionals).
#     - **Note this runs across all GPUs and it *is* safe to make state assignments here**
#
# 3. [x_dataloader()](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning-module.html#data-hooks) ♻️
#     - `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` all return PyTorch `DataLoader` instances that are created by wrapping their respective datasets that we prepared in `setup()`


# %% colab={} colab_type="code" id="4DNItffri95Q"
class LitMNIST(pl.LightningModule):

    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * width * height, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

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


# %% colab={} colab_type="code" id="Mb0U5Rk2kLBy"
model = LitMNIST()
trainer = pl.Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)
trainer.fit(model)

# %% [markdown] colab_type="text" id="nht8AvMptY6I"
# ### Testing
#
# To test a model, call `trainer.test(model)`.
#
# Or, if you've just trained a model, you can just call `trainer.test()` and Lightning will automatically
# test using the best saved checkpoint (conditioned on val_loss).

# %% colab={} colab_type="code" id="PA151FkLtprO"
trainer.test()

# %% [markdown] colab_type="text" id="T3-3lbbNtr5T"
# ### Bonus Tip
#
# You can keep calling `trainer.fit(model)` as many times as you'd like to continue training

# %% colab={} colab_type="code" id="IFBwCbLet2r6"
trainer.fit(model)

# %% [markdown] colab_type="text" id="8TRyS5CCt3n9"
# In Colab, you can use the TensorBoard magic function to view the logs that Lightning has created for you!

# %% colab={} colab_type="code" id="wizS-QiLuAYo"
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
