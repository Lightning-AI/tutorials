# %% [markdown]
# ## Introduction
#
# First, we'll go over a regular `LightningModule` implementation without the use of a `LightningDataModule`

# %%
import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

# %% [markdown]
# ### Defining the LitMNISTModel
#
# Below, we reuse a `LightningModule` from our hello world tutorial that classifies MNIST Handwritten Digits.
#
# Unfortunately, we have hardcoded dataset-specific items within the model,
# forever limiting it to working with MNIST Data. 😢
#
# This is fine if you don't plan on training/evaluating your model on different datasets.
# However, in many cases, this can become bothersome when you want to try out your architecture with different datasets.


# %%
class LitMNIST(L.LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # We hardcode dataset specific stuff here.
        self.data_dir = data_dir
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Build model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

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
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=128)


# %% [markdown]
# ### Training the ListMNIST Model

# %%
model = LitMNIST()
trainer = L.Trainer(
    max_epochs=2,
    accelerator="auto",
    devices=1,
)
trainer.fit(model)

# %% [markdown]
# ## Using DataModules
#
# DataModules are a way of decoupling data-related hooks from the `LightningModule
# ` so you can develop dataset agnostic models.

# %% [markdown]
# ### Defining The MNISTDataModule
#
# Let's go over each function in the class below and talk about what they're doing:
#
# 1. ```__init__```
#     - Takes in a `data_dir` arg that points to where you have downloaded/wish to download the MNIST dataset.
#     - Defines a transform that will be applied across train, val, and test dataset splits.
#     - Defines default `self.dims`.
#
#
# 2. ```prepare_data```
#     - This is where we can download the dataset. We point to our desired dataset and ask torchvision's `MNIST` dataset class to download if the dataset isn't found there.
#     - **Note we do not make any state assignments in this function** (i.e. `self.something = ...`)
#
# 3. ```setup```
#     - Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
#     - Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
#     - If you don't mind loading all your datasets at once, you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever `None` is passed to `stage`.
#     - **Note this runs across all GPUs and it *is* safe to make state assignments here**
#
#
# 4. ```x_dataloader```
#     - `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` all return PyTorch `DataLoader` instances that are created by wrapping their respective datasets that we prepared in `setup()`


# %%
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


# %% [markdown]
# ### Defining the dataset agnostic `LitModel`
#
# Below, we define the same model as the `LitMNIST` model we made earlier.
#
# However, this time our model has the freedom to use any input data that we'd like 🔥.


# %%
class LitModel(L.LightningModule):
    def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# %% [markdown]
# ### Training the `LitModel` using the `MNISTDataModule`
#
# Now, we initialize and train the `LitModel` using the `MNISTDataModule`'s configuration settings and dataloaders.

# %%
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes)
# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)

# %% [markdown]
# ### Defining the CIFAR10 DataModule
#
# Lets prove the `LitModel` we made earlier is dataset agnostic by defining a new datamodule for the CIFAR10 dataset.


# %%
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE)


# %% [markdown]
# ### Training the `LitModel` using the `CIFAR10DataModule`
#
# Our model isn't very good, so it will perform pretty badly on the CIFAR10 dataset.
#
# The point here is that we can see that our `LitModel` has no problem using a different datamodule as its input data.

# %%
dm = CIFAR10DataModule()
model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
)
trainer.fit(model, dm)
