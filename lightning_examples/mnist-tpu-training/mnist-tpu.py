# %% [markdown]
# ###  Install Colab TPU compatible PyTorch/TPU wheels and dependencies

# %%
# ! pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

import lightning as L

# %%
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST

BATCH_SIZE = 1024

# %% [markdown]
# ### Defining The `MNISTDataModule`
#
# Below we define `MNISTDataModule`. You can learn more about datamodules
# in [docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).


# %%
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
# ### Defining the `LitModel`
#
# Below, we define the model `LitMNIST`.


# %%
class LitModel(L.LightningModule):
    def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        self.save_hyperparameters()

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# %% [markdown]
# ### TPU Training
#
# Lightning supports training on a single TPU core or 8 TPU cores.
#
# The Trainer parameter `devices` defines how many TPU cores to train on (1 or 8) / Single TPU core to train on [1]
# along with accelerator='tpu'.
#
# For Single TPU training, Just pass the TPU core ID [1-8] in a list.
# Setting `devices=[5]` will train on TPU core ID 5.

# %% [markdown]
# Train on TPU core ID 5 with `devices=[5]`.

# %%
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.size(), dm.num_classes)
# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="tpu",
    devices=[5],
)
# Train
trainer.fit(model, dm)

# %% [markdown]
# Train on single TPU core with `devices=1`.

# %%
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes)
# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="tpu",
    devices=1,
)
# Train
trainer.fit(model, dm)

# %% [markdown]
# Train on 8 TPU cores with `accelerator='tpu'` and `devices=8`.
# You might have to restart the notebook to run it on 8 TPU cores after training on single TPU core.

# %%
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes)
# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="tpu",
    devices=8,
)
# Train
trainer.fit(model, dm)
