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

# %% [markdown] id="x83-rnVKT8Wo"
# # GPU and batched data augmentation with Kornia and PyTorch-Lightning
#
# In this tutorial we will show how to combine both [Kornia.org](https://www.kornia.org) and PyTorch Lightning to perform efficient data augmentation to train a simpple model using the GPU in batch mode without additional effort.
#
# **NOTE:** Adaptation of the original post found in [Kornia tutorials](https://kornia-tutorials.readthedocs.io/en/latest/).

# %% [markdown] id="iCsre0XmawoR"
# ## Setup
# We first need to install Kornia and PyTorch Lightning and for convenience also torchmetrics.

# %% id="dEeUzX_5aLrX"
# ! pip install kornia pytorch_lightning torchmetrics -qU

# %% id="Z_XTj-y1gYJL"
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

# %% [markdown] id="hA4-AFd6gKo-"
# ## Define Data Augmentations module
#
# [Kornia.org](https://www.kornia.org) is low level Computer Vision library that provides a dedicated module [`kornia.augmentation`](https://kornia.readthedocs.io/en/latest/augmentation.html) module implementing en extensive set of data augmentation techniques for image and video.
#
# Similar to Lightning, in Kornia it's promoted to encapsulate functionalities inside classes for readability and efficiency purposes. In this case, we define a data augmentaton pipeline subclassing a `nn.Module` where the augmentations (also subclassing `nn.Module`) are combined with other PyTorch components such as `nn.Sequential`.
#
# Checkout the different augmentation operators in Kornia docs and experiment yourself !


# %% id="RvdMAbyXgRPh"
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


# %% [markdown] id="CQ06HnPtkIlq"
# ## Define a Pre-processing module
#
# In addition to the `DataAugmentation` modudle that will sample random parameters during the training stage, we define a `Preprocess` class to handle the conversion of the image type to properly work with `torch.Tensor`.
#
# For this example we use `torchvision` CIFAR10 which return samples of `PIL.Image`, however, to take all the advantages of PyTorch and Kornia we need to cast the images into tensors.
#
# To do that we will use `kornia.image_to_tensor` which casts and permutes the images in the right format.


# %% id="EjPPQjcXkNT3"
class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.


# %% [markdown] id="LXGKqIZuTLMs"
# ## Define PyTorch Lightning model
#
# The next step is to define our `LightningModule` to have a proper organisation of our training pipeline. This is a simple example just to show how to structure your baseline to be used as a reference, do not expect a high performance.
#
# Notice that the `Preprocess` class is injected into the dataset and will be applied per sample.
#
# The interesting part in the proposed approach happens inside the `training_step` where with just a single line of code we apply the data augmentation in batch and no need to worry about the device. This means that our `DataAugmentation` pipeline will automatically executed in the GPU.


# %% id="aDagOcKyZ_qh"
class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model: expereiment yourself
        self.model = torchvision.models.resnet18(pretrained=True)

        self.preprocess = Preprocess()  # per sample transforms

        self.transform = DataAugmentation()  # per batch augmentations

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


# %% [markdown] id="yyLkgogITNtj"
# ## Visualize images

# %% id="kr8cql-aaKnC"
# init model
model = CoolSystem()

# %% colab={"base_uri": "https://localhost:8080/", "height": 888} id="O2Fq4bK4l9sS" outputId="5711e7a2-07fa-4de8-b663-b5d9050247ce"
model.show_batch(win_size=(14, 14))

# %% [markdown] id="baSvOQRNl5iw"
# ## Run training

# %% id="O_AFCY3WlwKg" colab={"base_uri": "https://localhost:8080/", "height": 553, "referenced_widgets": ["709f1f18ae7b4057a564ebc20c30cf14", "efb1a69c4cb845ca8e42d6b06c8f49d2", "c811b82ef9354d7f83f47ec1546eb561", "4415c8b9a5b44c3d939728447fab9f4b", "7370c65f4bbf4d0eb5d81ed7a2d373af", "3089633e5f9f429684990ffb14e4c4b9", "9b0c91d7f2524a6eb8643118033ad30c", "2f7c9dc8c4394de2bbfc95ad24a9672a", "975dcd78a68247adbe3d305baece7c48", "c271876882a24ea6a7eb65816e34fa70", "1f753f12e0e54986bf101c21a4e5dfb8", "87a05880ab66478ab3b5cc849490cc06", "a9e8ee5e6a0d4f1fac52b4a361d35a66", "175d1617dbe14a61b924d3f44d0ace68", "14a50a956e0a4ed8b6e42d67ede920c7", "5911bec114fd4cbbba278e4d9c48711e", "893b98245d96477e887839494deb8b3d", "cc8ad4fbde744d7f9e43905929732cfa", "12c6b06aba1b4b93abbbd9ec5990ded6", "d5ed68ccedc14d88bc4bfa83015d500b", "c0d2b3e733b44dd0a60925e77b2c2843", "849740f2aa3f4b89b48ca9b75b4199ef", "7056185dcfa14145ba34704ccd31310e", "1594b1b3e13944e98ff6583e08b7da74", "60cd8ee70e83400ea0a340a9019c4613", "ec2e09636e4347bbb09eb5db626bb629", "a5602e74c0ef433d94af5ea32c0082d9", "c9006e61791e4a478c264ba3aa1ed735", "52a05b6e3531499cad49d8a6fbb21e87", "3bf765a4a77640518b91e529322163a3", "0b9dc57ed0984178be55834ed5ffaf6c", "9f0a2e194ab649a3af6ab6f1c5bddf40", "376c5e8cfdc44d2abab0056fbfd66d10", "0435543f907e4982993ad25c8045fd7b", "4946d46482ea4c979267f8947189f8ef", "6d29e2ad483b4ccabcba7f0adbb153a8", "be9607a996654cd1ac5953e3e9ccd9c9", "6e5e25f7a38d4aa2a31aae68ec57809c", "4e521a90685841fa8971afce65cf2082", "b6648c8e46fb49bd9d100b39d160ba6d", "6ebcfe11f51b49ebba4fba50a98bd272", "3e9d78e22b0a44e09503bbc4fd67c4e3", "80f22520340f41ce844f2a48b6ca26e6", "fcbbda9fd1c84687add33bbdb4876c19", "22aff830f48a4af59f0c1883b32329e6", "d46e2150726d43cd95cf9c59d2473ba7", "0c76afa96d044f839ecd6fbb4b20ca52", "1359309ae6e04b57b85941e8322207b1", "1b67cc25af184e7f9556c5de1dfd54cd", "98b4b51a2aaf49afaa8401e764d7833a", "a75282d841454fc399f626ad5602a74d", "dba0cfbbf338449b90a67e921e9b88b9", "6ac0a596086d44c69fcdf3ede8d6cf97", "79b7c0471b9f4361b7f163d9bb20547b", "d419cbe0111c4244a4147e960f8877d4", "96f4f3a8c31141a8834394db86f59f9a", "ec033e4dca90465c90fc2be25dd22f78", "aa5e83835b5140fe9b58a4d9294aaf26", "cad29c4d9f7545a29b4a90e89d9215aa", "212b03cac43443c194d1b07e009b7fa4", "35be5b0f9a0b4e52bdd3d3fee4973a2a", "a619744530a64286903d72570ae22e66", "0858b8de821d4a97925a1c2345384506", "aeb789428b5f48a380e842b276899aaf", "599c941c2a5748a1bd23c01997f89deb", "1159393a62f8445b9987cc5c6a0345b9", "0aa995b993044fc2859a0bd78bd0f574", "3435453a01004b8ab13aba41380669c0", "f1f82dafccc94d36b94235edc5e8e9a2", "e4ada31d4fa247d09f4902d24c0de8fc", "eba24180d9504770a215579e231f77c5", "fe78a1c28f4e48e580e90bb86c2cd974", "de0a418bdb7b4645a720e42c4360bc56", "5dc59cbf46d34facbf6bda28d901f23d", "77f480b972fa4a429985b665a703bca5", "1ad61d6d3ba742058f33ba42d0f7aee5", "40ac1628402c404397f5f7fe4a392ec4", "58907965ba6548698fd4ee09d8cc537f", "c6173d0bae434ad894bbcdb950978597", "5a19c9bd4631496c83c086c9a0790eb7", "36bc9e0864c041a3a9ba99138526dd0a", "9e5951cd14134a01a0989847610e5158", "b2ab723a9bd442caa234c31c3a0b37f2", "ba95f0a20a644817838d65454c68e62a", "6d1131a36b0643d0bbe361b1dce2530a", "2aad354dd58a4370ab02e9717b222527", "307eda2be3d44b6586e10cde67457eb1", "a86b460ff8b64039b3c1ad9c20fc509b", "dd8c0b79bc1e4543827f4c945e612afe", "98b00e759bd1480d8642e74dd513bb81", "8a1bc0325fa340c09908f90fdc5c3d3b", "0db90d9f3f8f4b369c902c4e1fb7d872", "6ec62f397aa24e08ae2d3333333e67e2", "6ad4eb9da10b4257aaae2edc87ed75bd", "013b59fa0c4d487eb71f900a4703c7bc", "aa6bdb26784c46e98dadcd8ccef5af45"]} outputId="12d1be70-3b5c-41e1-95bd-1f9d91bd07ae"
# Initialize a trainer
trainer = Trainer(
    progress_bar_refresh_rate=20,
    gpus=min(1, torch.cuda.device_count()),
    max_epochs=10,
    logger=pl.loggers.CSVLogger(save_dir='logs/', name="cifar10-resnet18")
)

# Train the model ⚡
trainer.fit(model)

# %% [markdown] id="jrSNajrhx9E_"
# ### Visualize the training results

# %% colab={"base_uri": "https://localhost:8080/", "height": 635} id="jXhH_0mbx_ye" outputId="d294929e-4af5-44d1-8b57-ccb378167fa3"
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

# %% [markdown] id="t65M0xD4lgAh"
# ## Tensorboard

# %% id="xg3c6UJQt0mw"
# Start tensorboard.
# # %load_ext tensorboard
# # %tensorboard --logdir lightning_logs/
