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

# %% [markdown] id="jKj5lgdr5j48"
# ---
# ### Setup
# Lightning is easy to use. Simply ```pip install pytorch-lightning```

# %% colab={"base_uri": "https://localhost:8080/", "height": 938} id="UGjilEHk4vb7" outputId="229670cf-ec26-446f-afe5-2432c4571030"
# ! pip install pytorch-lightning==0.8.3 --upgrade --silent

# %% id="zaVUShmQ5n8Y"
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

# %% [markdown] id="gEulmrbxwaYL"
# ---
# ## MNIST hello world

# %% [markdown] id="nbQAcRna5e_q"
# ## Simplest example
#
# Here's the simplest most minimal example with just a training loop (no validation, no testing).
#


# %% id="zM15oxCH5lo6"
class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# %% colab={"base_uri": "https://localhost:8080/", "height": 639, "referenced_widgets": ["bb305bb378774c1586a3196eb3babd29", "ff6eead2826e4113abf7ab3a8cb31b0f", "992f545bb6f1489493d89a59d046f63f", "44e2c572ab1641a29156ad4ee8884e12", "254ddfa7c23d4b6f828d515dbab38978", "a700c003887b4d2ba134fbfcf4823cb5", "32e47e93509043439580cd5f58dc7726", "845415af79634da5a64b9f368280c0e3", "a7a94847786244dd9a5cb9718957143a", "8734c1b798ff4ba0bf77dca4f3de9cbf", "ee0a78c01b63443f9e51470a1b1e79a4", "632b9d0d9ffa4d479deb70f6fafb92ab", "a9413692ae5040e6ae3c2a446dbe297c", "ca4cd1659d73446e964f9ab36d92e3a0", "42e787b78000472eab434fb795197a86", "1a7680c6279d4985bd69188dd72b11d5", "3397549a0695432990f1d3d5390941e7", "515ef7d03ef2447e9643210b029b930e", "ae52e3d810aa4bc5965559ed2ba2b78a", "08b6d9269e514d228e7e94fe0299a2c5", "7ee81979301c447bb13ff9fff5153e0f", "ea162090fc954f0198a1d63507dfff9b", "0cf9a61c88af45b6a6ef72640f93cbfd", "67728556b4c9432b877d54a081657663", "de325f4002a945b4a2a15086c2a77816", "5012438370764b4db215d545e9414c94", "6aafaca3c8824e2fa267f4a68d5d2ca3", "c4200c1f957a4179af51245a797c8921", "53b2a85381b1460d9f446390c79bfc08", "59f02fe7f9f2433bb25f5b292c213f50", "1dabf5740f4d44d68d06629f77b001e3", "0f688614251d49589f320f2b2cb55344", "c93f037dc6044d858ae1862d5b29f6f0", "00ae53beaa9341f4826b1bdc0a6f88e0", "4b7021f73f6b4e5193454128ccf323d7", "6f55aa11acb14afdb2ac0a1052be1bb6", "b5f184fbcba740999b205e34e23455d6", "d9540ab5d2394b77a65f48b501acdc18", "23fd97d95fae4f42bd21906f67115f8b", "420e8d65e9584973a8004e8398cf430c"]} id="5VEbFQp55wqo" outputId="c2321d5d-bbad-4896-b41b-dbc9ed19340d"
train_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

mnist_model = MNISTModel()
trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=20)
trainer.fit(mnist_model, train_loader)

# %% [markdown] id="gjo55nA549pU"
# ### 1. LightningModule
# Each project goes into a LightningModule.
# This module houses:
# 1. Model definition (__init__)
# 2. Computations (forward)
# 3. What happens inside the training loop (training_step)
# 4. What happens inside the validation loop (validation_step)
# 5. What optimizer(s) to use (configure_optimizers)
# 6. What data to use (train_dataloader, val_dataloader, test_dataloader)

# %% id="x-34xKCI40yW"
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # called with self(x)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32
        )


# %% [markdown] id="GROo8IDX-QCx"
# ### 2. Trainer
# The trainer is where the magic happens!
# Feed ANY LightningModule to a trainer to train the model.

# %% colab={"base_uri": "https://localhost:8080/", "height": 510, "referenced_widgets": ["6b3a598e7d01407aa5850b5a6620e7f0", "e23c0d6c117246b2a0a6681008748917", "e7a06a13ea11427ea3866cec3a55b644", "620f45256d504f0188f40c61e23e1355", "05f01b8813374534a4c58ac65fe2b390", "ee3d759a4e4442288599eacfd1347c8d", "c3587b40d9f942e98b708ff0b5fc5301", "e64a723bcf474d6699d78ec05462f995", "4ffe9fb35ca44358b0177723f73a35d8", "ba3fe1aba1b04a1fbaab268dfd3d0166", "fc3f14c4e83048aa9d6fe9963f95bf7a", "171db7c8fa1e4f11aaff71b9f5879d58", "4eaea330bc8e414fbf2f0e2b21af8b08", "118c0b8da0df4ff68a90a3d500f1d1b8", "135883097f0e428c963ae0ad320dfabd", "85741a7765a147c4a8d69872850cb072", "3d2e43ae9f924fbd8463ce72b44200f3", "7c03e0a2247442ad9c11569e443de4bb", "3a521cb700f247fd8cd345b158697f2f", "8a40fcad728841c7ab2fd15d2c40ee5f", "bc4dfb14c9d14499ae72d2a30fc6bc2d", "866f880d850a4e689a1c99723f0366db", "e7a864f4dc0f485eb045b778e981fc01", "56ef38eff92143bcaf68b22c8dae7f98", "a252ac32033b4e39b87a6c91bd21b5ae", "40ce71ff339849748486ebbc73474cbe", "f1db9d62eee44e61bb8ac26c16b3b601", "69fa0a853cf84b9482043e12881c849f", "d8db4d3709f34c869dedbc066e60501e", "9b6445338a69425889a8901c192d5144", "51b1111f5fe24042b38af809285e1b16", "294d8142a4aa48aa8261b0b8155ef97f", "a2990a67f92c4047b95876aae91e3de0", "44ca9ee5c356458680a5d20c6a891c91", "7e8cf26303ed4975b239fd43184a1dc6", "d7406a8b15f9439fba19ec4dab086c61", "fdf5c4a49602423184f6d94cd814177e", "5d3c506d3f4444d8a6b7024cd11de2cd", "42ede89dbd194eb6a603ccd7d4b96aae", "ea13174e5b894e93b3c59d7e599de5a9", "ee908316d227495381e8cf7dcf5526f1", "a4b49709f7464ce491324e8aa636c152", "8eb2086a01cf41429a5f4adff5f2359b", "bbef89e4fd9d4cf8ae4c8fcab9bc665d", "d01088cc378044cba4879032d74a852e", "352d7dae131b407cb6e0238315c1b1a0", "91a6de2063cc48b28021ef29feab7f69", "39422514a4a04a9ba290285dc586ea9f", "b73a326ada4d4a859e3c2c39abf5530d", "cd942318db094680821f0d9902941977", "29650c4a829b44ed9e1526b1dc5d2b83", "df6521155d05459882601ba8c84f3dce", "384a36423d154f2abcddb5094afeced3", "a99b7813bf88496c875a818afe3b170a", "f4a052d2223a4d4fa95ed52f94ad465d", "7ea8ad4e10bb465aa2b6708655a2793f", "15bb223836764207a5ac15616a41ddb7", "46f7df7035d44bd099f60ad23f836f8a", "296453e43f7344de8a9b5c6bc970ab1e", "cd86997da08649d7999ade2d0e7cea96", "7d15fc81537a449cb6b6afd7ccc65dac", "bef041a9f5a942f68b4a8488a371d3da", "e10c94b1fdf84a9186ab7d87fd83f87f", "19c7460c565d494abbb8b9731a34294d"]} id="HOk9c4_35FKg" outputId="a07e65a7-7452-478d-f80e-179272b26b8a"
mnist_model = MNISTModel()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(mnist_model)

# %% [markdown] id="IV77NP_Ywkzb"
# ### Testing

# %% colab={"base_uri": "https://localhost:8080/", "height": 134, "referenced_widgets": ["bcf69c2a0b694e0498beadb6f4509395", "2e20c741cf8a401cb90e8e230a23026a", "a7bcd18049d8493b9d3d9f17d86f0429", "3c99401bde8641c19978c11c9abb906a", "cd84335fb7234f3aa54dafe045614e56", "f261b8aab86b4d6e94984bf658c1b74d", "fd8ec919352046dd84057e9763bb235a", "f778d9ef70ca4f5898c423109cf82ed2"]} id="-Bnkq97qhe2x" outputId="9db00280-ef5b-4ae4-8a6d-174590ae6d0c"
trainer.test()

# %% [markdown] id="Q-qxNrXvKAlN"
# ### Plotting
#
# Plot the results

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# %% [markdown] id="xF9-ouAEGFlZ"
# By using the trainer you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# %% [markdown] id="18STRwHg-kW8"
# ### Bonus
# In fact, if you keep calling fit, it'll keep training the model where it left off!

# %% colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["6aa5e292e2094c239e7418994a31ff51", "555443a6fa564d10a3a7901cf15a79a3", "1f9c48164702427fb3aca2a26b2651e5", "d8bd5c9b233b41008109d14cffc89aaa", "61c71d4f1c2848b1813aebc0b2db5e25", "a1e2c38bb40642168cc9d44abf645a54", "d0590d65433c4478af6a0762421f9f7a", "76c916d634c644a4a0d8f12e183822fd", "a38242d3231442e2a259067d6a1355c2", "9a9ebf052d914a8881882da8d2fa9cd8", "a56deb884719491090a4146e72be3868", "7633820adf9a4757ae73b472e43031d6", "b7a073dfdeaf48fc9f3e6352b0ea2ba7", "8aab627e715a44ada2af81b74bece257", "fc262db2a53948488092a77209081319", "11db4a94a4534fc2b503aad28be631be"]} id="U2d1gc4N5IJX" outputId="f68aaf1f-dfa9-4f30-de7e-d4fdab9eb089"
trainer.fit(mnist_model)

# %% [markdown] id="P0bSmCw57aV5"
# ---
# ## GAN Example
#
# How to train a GAN!
#
# Main takeaways:
# 1. Generator and discriminator are arbitraty PyTorch modules.
# 2. training_step does both the generator and discriminator training.

# %% [markdown] id="pBhBR3QJ7mhx"
# #### A. Generator

# %% id="mesU_huG-rr6"
"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


# %% [markdown] id="bt37ycLx7uO3"
# ### B. Discriminator


# %% id="pcPCt8JG7tI-"
class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# %% [markdown] id="TyYOdg8g77P0"
# ### C. GAN


# %% id="ArrPXFM371jR"
class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        z = torch.randn(8, self.hparams.latent_dim)
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


# %% [markdown] id="-WRY6dfn8ScZ"
# ### D. Trainer

# %% [markdown] id="xsmHHcpP8ryX"
# Here we fake using argparse

# %% id="fIJl3phH8uEI"
from argparse import Namespace

args = {'batch_size': 32, 'lr': 0.0002, 'b1': 0.5, 'b2': 0.999, 'latent_dim': 100}
hparams = Namespace(**args)

# %% colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["5f119b90386c499ea9caf987fecf6c06", "6d391353197443a694f6c75147ca96df", "47df0bc1b3d14bb7b673e0591daa4e5f", "87e118f890dc42319e723331e1306787", "e068e2b1c68c48a784c19fc716c043a3", "7662324b3b924f8f9649dc409fb0d349", "afc85a52a5d04653ae9e7168b180ff98", "dbb9fd5429f5416ab6a4f78f0c72867c"]} id="h788dCGu7_Iu" outputId="bcebc504-f0fc-496b-c8d5-a0c2f3349155"
gan_model = GAN(hparams)

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(gan_model)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# %% [markdown] id="7uQVI-xv9Ddj"
# ---
# ## BERT example
# BERT + Lightning

# %% id="e2npX-Gi9uwa"
# ! pip install transformers

# %% [markdown] id="DeLyZQ_E9o1T"
# #### Data download + processing
#
# Let's grab the correct data

# %% colab={"base_uri": "https://localhost:8080/", "height": 164, "referenced_widgets": ["5484eef7b6f247d68a89f86965b0940f", "0c3473a16a5e4c46a6c7515e610bca7f", "ad849800b2124195b92f3bf9dfc7681b", "6ae5b2f9195847b5a0aa9991e14aa397", "240764252e7c4f5ca39db14fd1c724ed", "386ff59e3694480394253f1c24ff8e84", "70e48d7d8e8a411a90642926db4aada8", "1f3364ab59b541268fabcb3f9fb5c64c", "0fad6468e3c849b380e34f674e074219", "10a88a05740b45d4a6ea5873d4a7151a", "d3b107acd1b1401cabe3090724e12e86", "b3563100dd1b4a4abe14ab7193649064", "17f0e360e85f48d9a17b84c9b7f6c9f0", "29f35103a6e94af09c8ac9cdb2cca89c", "e6e15d5c14134be0b4cf86fdecfef687", "f23f02d00d424574afa29311b8d0906e", "e918a6de59b64bd590e4f1233bbc078a", "abeb0a773f3542c39ff724ae0674b74e", "892246fdf6bb476abb35ec321ddf86e8", "88c181cd21a94ec9a43df9754c1986c9", "e4098b0091124fef8ba342783a82cc6e", "498a50387a0742a88356a7ee9920bf7a", "86482894cddd4956ae2fc3d9edd8ef9a", "438d19fb8e8243ebbc658f4b1d27df99"]} id="eBP6FeY18_Ck" outputId="b2a5c5fd-88cf-4428-d196-9e1c1ddc7e30"
from transformers.data.processors.glue import MnliProcessor
import torch
from transformers import (BertModel, BertTokenizer)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)

# %% id="vMbozzxs9xq_"
import pandas as pd
import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {
    "CoLA": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",  # noqa
    "SST": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",  # noqa
    "MRPC": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",  # noqa
    "QQP": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP-clean.zip?alt=media&token=11a647cb-ecd3-49c9-9d31-79f8ca8fe277",  # noqa
    "STS": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",  # noqa
    "MNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",  # noqa
    "SNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",  # noqa
    "QNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601",  # noqa
    "RTE": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",  # noqa
    "WNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",  # noqa
    "diagnostic": [
        "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",  # noqa
        "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1",
    ],
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


# %% colab={"base_uri": "https://localhost:8080/", "height": 51} id="3CVHOXQY9yVm" outputId="f06b886b-cc32-4972-918e-f4ca5828fb2c"
download_and_extract('MNLI', '../../notebooks')

# %% id="vOR0Q1Yg-HmN"
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split

processor = MnliProcessor()

# %% [markdown] id="yuUwBKpn-TIK"
# #### Data loaders
#


# %% id="kMdQZUjO-MI7"
def generate_mnli_bert_dataloaders():
    # ----------------------
    # TRAIN/VAL DATALOADERS
    # ----------------------
    train = processor.get_train_examples('MNLI')
    features = convert_examples_to_features(
        train,
        tokenizer,
        label_list=['contradiction', 'neutral', 'entailment'],
        max_length=128,
        output_mode='classification',
        pad_on_left=False,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=0
    )
    train_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.label for f in features], dtype=torch.long)
    )

    nb_train_samples = int(0.95 * len(train_dataset))
    nb_val_samples = len(train_dataset) - nb_train_samples

    bert_mnli_train_dataset, bert_mnli_val_dataset = random_split(train_dataset, [nb_train_samples, nb_val_samples])

    # train loader
    train_sampler = RandomSampler(bert_mnli_train_dataset)
    bert_mnli_train_dataloader = DataLoader(bert_mnli_train_dataset, sampler=train_sampler, batch_size=32)

    # val loader
    val_sampler = RandomSampler(bert_mnli_val_dataset)
    bert_mnli_val_dataloader = DataLoader(bert_mnli_val_dataset, sampler=val_sampler, batch_size=32)

    # ----------------------
    # TEST DATALOADERS
    # ----------------------
    dev = processor.get_dev_examples('MNLI')
    features = convert_examples_to_features(
        dev,
        tokenizer,
        label_list=['contradiction', 'neutral', 'entailment'],
        max_length=128,
        output_mode='classification',
        pad_on_left=False,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=0
    )

    bert_mnli_test_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.label for f in features], dtype=torch.long)
    )

    # test dataset
    test_sampler = RandomSampler(bert_mnli_test_dataset)
    bert_mnli_test_dataloader = DataLoader(bert_mnli_test_dataset, sampler=test_sampler, batch_size=32)

    return bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader


# %% id="iV-baDhN-U6B"
bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader = generate_mnli_bert_dataloaders()

# %% [markdown] id="yr7eaxkF-djf"
# ### BERT Lightning module!
#
# Finally, we can create the LightningModule

# %% id="UIXLW8CO-W8w"
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class BertMNLIFinetuner(pl.LightningModule):

    def __init__(self):
        super(BertMNLIFinetuner, self).__init__()

        self.bert = bert
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return bert_mnli_train_dataloader

    def val_dataloader(self):
        return bert_mnli_val_dataloader

    def test_dataloader(self):
        return bert_mnli_test_dataloader


# %% [markdown] id="FHt8tgwa_DcM"
# ### Trainer

# %% colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["86bedd1fc6da4b8fa0deac637628729e", "f444ab7646444b9885cfec41b5a2236e", "fad0b06dc57e4b4599cf43daad7106b8", "c190999c2761453380f816372fcca608", "a5cc9e60aff641dca27f1adf6807e5b3", "0a96cc26343e4bb2ac2f5145be2fbacf", "cce9ed8de0a048679453e53b71523eea", "773fd1b84c364903bc7350630e76a825", "0e149cc766d147aba2c05f8b0f2c69d5", "191f483b5b0346a8a28cac37f29ac2dc", "24b28a7423a541c0b84ba93d70416c1a", "4820f0005e60493793e506e9f0caf5d4", "fce1fc72006f4e84a6497a493cbbfca2", "f220485e332d4c3cbfc3c45ce3b5fdf1", "bf257b8a04b44a389da2e6f4c64379d4", "7efa007fdb2d4e06b5f34c4286fe9a2f"]} id="gMRMJ-Kd-oup" outputId="790ab73c-b37d-4bcb-af5f-46b464e46f9b"
bert_finetuner = BertMNLIFinetuner()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(bert_finetuner)

# %% [markdown] id="NWvMLBDySQI5"
# ## DQN example
#
# How to train a Deep Q Network
#
# Main takeaways:
# 1. RL has the same flow as previous models we have seen, with a few additions
# 2. Handle unsupervised learning by using an IterableDataset where the dataset itself is constantly updated during training
# 3. Each training step carries has the agent taking an action in the environment and storing the experience in the IterableDataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 146} id="4ARIT37rDdIZ" outputId="37ea5092-0db7-4e73-b507-f4be9bb0ae7e"
# !pip install gym

# %% [markdown] id="nm9BKoF0Sv_O"
# ### DQN Network

# %% id="FXkKtnEhSaIV"
from torch import nn


class DQN(nn.Module):
    """
    Simple MLP network

    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x.float())


# %% [markdown] id="c9clSz7xTFZf"
# ### Memory

# %% id="zUmawp0ITE3I"
from collections import namedtuple

# Named tuple for storing experience steps gathered in training
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# %% id="Zs7h_Z0LTVoy"
from typing import Tuple


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool),
            np.array(next_states)
        )


# %% id="R5UK2VRvTgS1"
from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


# %% [markdown] id="d7sCGSURTuQK"
# ### Agent

# %% id="dS2RpSHHTvpO"
import gym
import torch


class Agent:
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ['cpu']:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


# %% [markdown] id="IAlT0-75T_Kv"
# ### DQN Lightning Module

# %% id="BS5D7s83T13H"
import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            'total_reward': torch.tensor(self.total_reward).to(device),
            'reward': torch.tensor(reward).to(device),
            'train_loss': loss
        }
        status = {
            'steps': torch.tensor(self.global_step).to(device),
            'total_reward': torch.tensor(self.total_reward).to(device)
        }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


# %% [markdown] id="JST5AN-8VFLY"
# ### Trainer


# %% id="bQEvD7gFUSaN"
def main(hparams) -> None:
    model = DQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=1, distributed_backend='dp', max_epochs=500, early_stop_callback=False, val_check_interval=100
    )

    trainer.fit(model)


# %% colab={"base_uri": "https://localhost:8080/", "height": 380, "referenced_widgets": ["e9a6bf4eda3244c6bb17216715f36525", "0922c5b2de554b4fa28dd531603f2709", "c293fc4171b0438595bc9a49fbb250cf", "819c83bf0bbd472ba417c31e957718c7", "c24384195a074989a86217b2edc411cb", "b3817e0ba30f449585f7641b4d3061bb", "8591bd2136ab4bb7831579609b43ee9c", "5a761ed145474ec7a30006bc584b26be"]} id="-iV9PQC9VOHK" outputId="2fd70097-c913-4d68-e80a-d240532edd19"
import numpy as np
import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
parser.add_argument(
    "--warm_start_size",
    type=int,
    default=1000,
    help="how many samples do we use to fill our buffer at the start of training"
)
parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
parser.add_argument("--max_episode_reward", type=int, default=200, help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000, help="max episode reward in the environment")

args, _ = parser.parse_known_args()

main(args)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
