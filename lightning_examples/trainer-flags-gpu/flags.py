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
# ---

# %% id="zaVUShmQ5n8Y"
import os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

# %% id="6tgkS8IYZwY_"
# ------------
# data
# ------------
seed_everything(1234)
PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
MAX_EPOCHS = 3

# Init DataLoader from MNIST Dataset


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        subset_idxs = list(range(len(mnist_full)))[::4]
        mnist_subset = Subset(mnist_full, subset_idxs)
        self.mnist_train, self.mnist_val = random_split(mnist_subset, [12000, 3000])
        # Assign test dataset for use in dataloader(s)
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


mnist = MNISTDataModule()

# %% [markdown] id="gEulmrbxwaYL"
# ### Simple AutoEncoder Model
#
# Were gonna define a simple Lightning model so we can play with all the settings of the Lightning Trainer.
#
# LightningModule is simply pure Pytorch reorganized into hooks, that represents all the steps in the training process.
#
# You can use LightningModule hooks to control every part of your model, but for the purpose of this video
# we will use a very simple MNIST classifier, a model that takes 28*28 grayscale images of hand written images,
# and can predict the digit between 0-9.
#
# The LightningModule can encompass a single model, like an image classifier, or a deep learning system
# composed of multiple models, like this auto encoder that contains an encoder and a decoder.
#


# %% id="x-34xKCI40yW"
class LitAutoEncoder(LightningModule):

    def __init__(self, batch_size=32, lr=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %% [markdown] id="VbxcRCrxiYly"
# You'll notice the LightningModule doesn't have epoch and batch loops, we're not calling model.train()
# and model.eval(), and no mentions of CUDA or hardware. That's because it is all automated by the Lightning Trainer.
# All the engineering boilerplate is automated by the trainer:
#
# *  Training loops
# *  Evaluation and test loops
# *  Calling model.train(), model.eval(), no_grad at the right time
# *  CUDA or to_device calls
#
# It also allows you to train your models on different hardware like GPUs and TPUs without changing your code!
#
#
# ### To use the lightning trainer simply:
#
# 1. init your LightningModule and datasets
# 2. init lightning trainer
# 3. call trainer.fit
#

# %% id="HOk9c4_35FKg"
#####################
# 1. Init Model
#####################

model = LitAutoEncoder()

#####################
# 2. Init Trainer
#####################

# these 2 flags are explained in the later sections...but for short explanation:
# - progress_bar_refresh_rate: limits refresh rate of tqdm progress bar so Colab doesn't freak out
# - max_epochs: only run 2 epochs instead of default of 1000
trainer = Trainer(max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=20)

#####################
# 3. Train
#####################
trainer.fit(model, datamodule=mnist)

# %% [markdown] id="3meDako-Qa_6"
# Our model is training just like that, using the Lightning defaults. The beauty of Lightning is that everything
# is easily configurable.
# In our next videos were going to show you all the ways you can control your Trainer to do things like controlling
# your training, validation and test loops, running on GPUs and TPUs, checkpointing, early stopping, and a lot more.
#

# %% [markdown] id="z_Wry2MckQkI"
# # Training loop and eval loop Flags

# %% [markdown] id="0MkI1xB2vsLj"
#
# To really scale up your networks, you can use accelerators like GPUs. GPUs or Graphical Processing Units,
# parallelize matrix multiplications which enable speed ups of at least 100x over training on CPUs.
#
# Let's say you have a machine with 8 GPUs on it. You can set this flag to 1, 4, or 8 GPUs and lightning
# will automatically distribute your training for you.
#
# ```
# trainer = Trainer(gpus=1)
# ```
#
# ---------
#
# Lightning makes your code hardware agnostic... This means, you can switch between CPUs, GPUs without code changes.
#
# However, it requires forming good PyTorch habits:
#
# 1. First, remove the .cuda() or .to() calls in your code.
# 2. Second, when you initialize a new tensor, set the device=self.device in the call since every lightningModule
# knows what gpu index or TPU core it is on.
#
# You can also use type_as and or you can register the tensor as a buffer in your module’s __init__ method
# with register_buffer().
#
# ```
# # before lightning
# def forward(self, x):
#     z = torch.Tensor(2, 3)
#     z = z.cuda(0)
#
# # with lightning
# def forward(self, x):
#     z = torch.Tensor(2, 3)
#     z = z.type_as(x, device=self.device)
# ```
#
#
# ```
# class LitModel(LightningModule):
#
#     def __init__(self):
#         ...
#         self.register_buffer("sigma", torch.eye(3))
#         # you can now access self.sigma anywhere in your module
# ```

# %% [markdown] id="hw6jJhhjvlSL"
# Lightning Trainer automates all the engineering boilerplate like iterating over epochs and batches,
# training eval and test loops, CUDA and to(device) calls, calling model.train and model.eval.
#
# You still have full control over the loops, by using the following trainer flags:

# %% [markdown] id="pT5-ETH9eUg6"
# ## Calling validation steps
# Sometimes, training an epoch may be pretty fast, like minutes per epoch.
# In this case, you might not need to validate on every epoch. Instead, you can actually validate after a few epochs.
#
# Use `check_val_every_n_epoch` flag to control the frequency of validation step:

# %% id="Z-EMVvKheu3D"
# run val loop every 10 training epochs
trainer = Trainer(max_epochs=MAX_EPOCHS, check_val_every_n_epoch=10)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="UOzZr9S2UcSO"
# ## val_check_interval
#
# In some cases where your epoch is very long, you might want to check validation within an epoch.
#
# You can also run validation step within your training epochs, by setting `val_check_interval` flag.
#
# Set `val_check_interval` to a float between [0.0 to 1.0] to check your validation set within a training epoch.
# For example, setting it to 0.25 will check your validation set 4 times during a training epoch.
#
# Default is set to 1.0

# %% id="9kbUbvrUVLrT"
# check validation set 4 times during a training epoch
trainer = Trainer(max_epochs=MAX_EPOCHS, val_check_interval=0.25)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="Onm1gBsKVaw4"
# When you have iterable data sets, or when streaming data for production use cases,
# it is useful to check the validation set every number of steps.
# Set val_check_interval to an int:

# %% id="psn6DVb5Vi85"
# check validation set every 1000 training batches
# use this when using iterableDataset and your dataset has no length
# (ie: production cases with streaming data)
trainer = Trainer(max_epochs=MAX_EPOCHS, val_check_interval=1000)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="QkoYonrWkb7-"
# ## num_sanity_val_steps
#
# You may have run into an issue, where you have a bug in your validation loop,
# but won't catch it until your training loop ends.
#
# and if your training loop takes hours or days, you will waste valuable compute.
#
# Instead, lightning automatically runs through 2 steps of validation in the beginning to catch these
# kinds of bugs up front.
#
#
# The `num_sanity_val_steps` flag can help you run n batches of validation before starting the training routine.
#
# You can set it to 0 to turn it off

# %% id="zOcT-ugSkiKW"
# turn it off
trainer = Trainer(max_epochs=MAX_EPOCHS, num_sanity_val_steps=0)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="zS0ob1ZmTw56"
# Set it to -1 to check all validation data before training

# %% id="rzqvjA4UT263"
# check all validation data
trainer = Trainer(max_epochs=MAX_EPOCHS, num_sanity_val_steps=-1)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="uMB41wq4T3Z2"
# Or use any arbitrary number of validation steps

# %% id="lGP78aQzT7VS"
trainer = Trainer(max_epochs=MAX_EPOCHS, num_sanity_val_steps=10)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="H-xaYRtd1rb-"
# ## Limit train, validation, and test batches
#
# You can set limits on how much of training, validation and test dataset you want your model to check.
# This is useful if you have really large validation or tests sets, for debugging or testing something
# that happens at the end of an epoch.
#
# Set the flag to int to specify the number of batches to run
#
#

# %% id="XiK5cFKL1rcA"
# run for only 10 batches
trainer = Trainer(max_epochs=MAX_EPOCHS, limit_test_batches=10)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="Y4LK0g65RrBm"
# For example, some metrics need to be computed on the entire validation results, such as AUC ROC.

# %% id="8MmeRs2DR3dD"
trainer = Trainer(max_epochs=MAX_EPOCHS, limit_val_batches=10)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="xmigcNa1A2Vy"
# You can use a float to limit the batches be percentage of the set on every epoch

# %% id="W7uGJt8nA4tv"
# run through only 25% of the test set each epoch
trainer = Trainer(max_epochs=MAX_EPOCHS, limit_test_batches=0.25)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="YRI8THtUN7_e"
# # Training on GPUs
#
#

# %% [markdown] id="R8FFkX_FwlfE"
# To run on 1 GPU set the flag to 1

# %% id="Nnzkf3KaOE27"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="cxBg47s5PB1P"
# to run on 2 or 4 GPUs, set the flag to 2 or 4.

# %% id="cSEM4ihLrohT"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=2)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="ZE6ZgwtNudro"
# You can also select which GPU devices to run on, using a list of indices like [1, 4]
#
# or a string containing a comma separated list of GPU ids like '1,2'
#

# %% id="gQkJtq0urrjq"
# list: train on GPUs 1, 4 (by bus ordering)
# trainer = Trainer(gpus='0, 1') # equivalent
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=[0, 1])

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% id="XghDPad4us74"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=list(range(2)))

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="6FVkKHpSPMTW"
# You can use all the GPUs you have available by setting `gpus=-1`

# %% id="r6cKQijYrtPe"
# trainer = Trainer(gpus='-1') - equivalent
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=-1)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="2C-fNLm3UGCV"
# Lightning uses the PCI bus_id as the index for ordering GPUs.

# %% [markdown] id="_V75s7EhOFhE"
# ### `auto_select_gpus`
#
# You can save on GPUs by running in “exclusive mode”, meaning only one process at a time can access them.
# If your not sure which GPUs you should use when running exclusive mode, Lightning can automatically
# find unoccupied GPUs for you.
#
# Simply specify the number of gpus as an integer `gpus=k`, and set the trainer flag `auto_select_gpus=True`.
# Lightning will automatically help you find k gpus that are not occupied by other processes.

# %% id="_Sd3XFsAOIwd"
# enable auto selection (will find two available gpus on system)
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=2, auto_select_gpus=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="a5JGSBMQhJNp"
# ## analyzing GPU usage
#
# ### log_gpu_memory
#
# This is useful to analyze the memory usage of your GPUs.
#
# To get the GPU memory usage for every GPU on the master node, set the flag to log_gpu_memory=all.
#
# Under the hood, lightning uses the nvidia-smi command which may slow your training down.
#
# Your logs can become overwhelmed if you log the usage from many GPUs at once.
# In this case, you can also set the flag to min_max which will log only the min and max usage across
# all the GPUs of the master node.
#
# Note that lightning is not logging the usage across all nodes for performance reasons.

# %% id="idus3ZGahOki"
# log all the GPUs (on master node only)
trainer = Trainer(max_epochs=MAX_EPOCHS, log_gpu_memory='all')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="-mevgiy_hkip"
# To avoid the performance decrease you can also set `log_gpu_memory=min_max` to only log the min and max memory on the master node.
#

# %% id="SlvLJnWyhs7J"
# log only the min and max memory on the master node
trainer = Trainer(max_epochs=MAX_EPOCHS, log_gpu_memory='min_max')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="K82FLLIJVQG3"
#
# But what if you want to train on multiple machines and not just one?

# %% [markdown] id="YViQ6PXesAue"
# # Training on multiple GPUs

# %% [markdown] id="WacbBQUivxQq"
# Lightning makes your models hardware agnostic, and you can run on GPUs with a flip of a flag.
# Lightning also supports training on multiple GPUs across many machines.
#
# You can do this by setting the num_nodes flag.
#
# The world size, or the total number of GPUs you are using, will be gpus*num_nodes.
#
# If i set gpus=8 and num_nodes=32 then I will be training on 256 GPUs.

# %% [markdown] id="GgcSbDjjlSTh"
# ## Accelerators
#
# Under the hood, Lightning uses distributed data parallel (or DDP) by default to distribute training across GPUs.
#
# This Lightning implementation of DDP calls your script under the hood multiple times with the correct
# environment variables.
#
# Under the hood it's as if you had called your script like this:
#
# 1. Each GPU across each node gets its own process.
# 2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.
# 3. Each process inits the model. (Make sure to set the random seed so that each model initializes with the same weights.)
# 4. Each process performs a full forward and backward pass in parallel.
# 5. The gradients are synced and averaged across all processes.
# 6. Each process updates its optimizer.
#
# If you request multiple GPUs or nodes without setting a mode, DDP will be automatically used.
#

# %% id="n_Brr7F5wdtj"
# ddp = DistributedDataParallel
# trainer = Trainer(gpus=2, num_nodes=2) equivalent

# %% [markdown] id="edxHyttC5J3e"
# DDP is the fastest and recommended way to distribute your training, but you can pass in other backends
# to `accelerator` trainer flag, when DDP is not supported.
#
# DDP isn't available in
# * Jupyter Notebook, Google COLAB, Kaggle, etc.
# * If You have a nested script without a root package
# * or if Your script needs to invoke .fit or .test multiple times

# %% [markdown] id="ZDh96mavxHxf"
# ### DDP_SPAWN
#
# In these cases, you can use `ddp_spawn` instead. `ddp_spawn` is exactly like DDP except that it uses
# `.spawn()` to start the training processes.

# %% id="JM5TKtgLxo37"
# trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp_spawn')
# trainer.fit(model, datamodule=mnist)

# %% [markdown] id="sebhVE3qrhKK"
# We STRONGLY discourage this use because it has limitations (due to Python and PyTorch):
#
# * Since .spawn() trains the model in subprocesses, the model on the main process does not get updated.
#
# * Dataloader(num_workers=N), where N is large, bottlenecks training with DDP… ie: it will be VERY slow
# or won’t work at all. This is a PyTorch limitation.
#
# * Forces everything to be picklable.
#
# DDP is MUCH faster than DDP_spawn. To be able to use DDP we recommend you:
#
# 1. Install a top-level module for your project using setup.py
#
# ```
# # setup.py
# #!/usr/bin/env python
#
# from setuptools import setup, find_packages
#
# setup(name='src',
#       version='0.0.1',
#       description='Describe Your Cool Project',
#       author='',
#       author_email='',
#       url='https://github.com/YourSeed',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
#       install_requires=[
#             'pytorch-lightning'
#       ],
#       packages=find_packages()
#       )
#
# ```
#
# 2. Setup your project like so:
#
# ```
# /project
#     /src
#         some_file.py
#         /or_a_folder
#     setup.py
# ```
# 3. Install as a root-level package
# ```
# cd /project
# pip install -e .
# ```
# 4. You can then call your scripts anywhere
# ```
# cd /project/src
#
# python some_file.py --accelerator 'ddp' --gpus 8
# ```

# %% [markdown] id="cmB3I_oyw7a8"
# ### DP
#
# If you're using windows, DDP is not supported. You can use `dp` for DataParallel instead:
# DataParallel uses multithreading, instead of multiprocessing. It splits a batch across k GPUs.
# That is, if you have a batch of 32 and use DP with 2 gpus, each GPU will process 16 samples,
# after which the root node will aggregate the results.
#
# DP use is discouraged by PyTorch and Lightning. Use DDP which is more stable and at least 3x faster.
#

# %% id="OO-J0ISvlVCg"
# dp = DataParallel
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=2, accelerator='dp')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="Y7E2eHZKwUn9"
# ### DDP2
#
# In certain cases, it’s advantageous to use ***all*** batches on the same machine, instead of a subset.
# For instance, in self-supervised learning, a common performance boost comes from increasing
# the number of negative samples.
#
# In this case, we can use DDP2 which behaves like DP in a machine and DDP across nodes. DDP2 does the following:
#
# * Copies a subset of the data to each node.
# * Inits a model on each node.
# * Runs a forward and backward pass using DP.
# * Syncs gradients across nodes.
# * Applies the optimizer updates.
#
#
#

# %% id="Y4xweqL3xHER"
# ddp2 = DistributedDataParallel + dp
# trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp2')
# trainer.fit(model, datamodule=mnist)

# %% [markdown] id="lhKNCnveeeq5"
# - The second mode is ddp_spawn. This works like ddp, but instead of calling your script multiple times,
# lightning will use multiprocessing spawn to start a subprocess per GPU.
#
# However, you should be careful of mixing this mode with num_workers > 0 in your dataloaders
# because it will bottleneck your training.
# This is a current known limitation of PyTorch which is why we recommend using our ddp implementation instead.
#

# %% [markdown] id="HUf9ANyQkFFO"
#
# ### mocking ddp
#
# Testing or debugging DDP can be hard, so we have a distributed backend that simulates ddp on cpus to make it easier.
# Set `num_processes` to a number greater than 1 when using accelerator="ddp_cpu" to mimic distributed training
# on a machine without GPUs. Note that while this is useful for debugging, it will not provide any speedup,
# since single-process Torch already makes efficient use of multiple CPUs.

# %% id="ZSal5Da9kHOf"
# Simulate DDP for debugging on your GPU-less laptop
trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="ddp_cpu", num_processes=2)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="ncPvbUVQqKOh"
# # Advanced distributed training
#

# %% [markdown] id="4MP7bEgnv7qK"
#
# Lightning supports distributed training across multiple GPUs and TPUs out of the box by setting trainer flags,
# but it also allows you to control the way sampling is done if you need to.

# %% [markdown] id="wdHiTfAMepKH"
# ## replace_sampler_ddp
# In PyTorch, you must use torch.nn.DistributedSampler for multi-node or GPU training.
# The sampler makes sure each GPU sees the appropriate part of your data.
#
# ```
# # without lightning
# def train_dataloader(self):
#     dataset = MNIST(...)
#     sampler = None
#
#     if self.on_tpu:
#         sampler = DistributedSampler(dataset)
#
#     return DataLoader(dataset, sampler=sampler)
# ```
# Lightning adds the correct samplers when needed, so no need to explicitly add samplers.
# By default it will add `shuffle=True` for train sampler and `shuffle=False` for val/test sampler.
#
# If you want to customize this behaviour, you can set `replace_sampler_ddp=False` and add your own distributed sampler.
#
# (note: For iterable datasets, we don’t do this automatically.)
#

# %% id="ZfmcB_e_7HbE"
sampler = torch.utils.data.distributed.DistributedSampler(mnist.mnist_train, shuffle=False)
train_dataloader = DataLoader(mnist.mnist_train, batch_size=32, sampler=sampler)
sampler = torch.utils.data.distributed.DistributedSampler(mnist.mnist_val, shuffle=False)
val_dataloader = DataLoader(mnist.mnist_val, batch_size=32, sampler=sampler)

# %% [markdown] id="-IOhk1n0lL3_"
# ## prepare_data_per_node
#
# When doing multi NODE training, if your nodes share the same file system,
# then you don't want to download data more than once to avoid possible collisions.
#
# Lightning automatically calls the prepare_data hook on the root GPU of the master node (ie: only a single GPU).
#
# In some cases where your nodes don't share the same file system, you need to download the data on each node.
# In this case you can set this flag to true and lightning will download the data on the root GPU of each node.
#
# This flag is defaulted to True.

# %% id="WFBMUR48lM04"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=2, prepare_data_per_node=False)

trainer.fit(LitAutoEncoder(), train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

# %% [markdown] id="FKBwXqo4q-Vp"
# ## sync_batchnorm
#
# Batch norm is computed per GPU/TPU. This flag enables synchronization between batchnorm layers across all GPUs.
# It is recommended if you have small batch sizes.
#

# %% id="GhaCLTEZrAQi"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=2, sync_batchnorm=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="XuFA7VTFMY9-"
# # Debugging flags
#
# Lightning offers a couple of flags to make debugging your models easier:
#

# %% [markdown] id="AKoS3fdml4Jx"
# ## Fast Dev Run
#
# To help you save time debugging, your first run should use the fast_dev_run flag.
#
# This won't generate logs or save checkpoints but will touch every line of your code to make sure
# that it is working as intended.
#
# Think about this flag like a compiler. You make changes to your code,
# and run Trainer with this flag to verify that your changes are bug free.
#

# %% id="L5vuG7GSmhzK"
trainer = Trainer(max_epochs=MAX_EPOCHS, fast_dev_run=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="HRP1qQR5nT4p"
# ## overfit_batches
#
# Uses this much data of the training set. If nonzero, will use the same training set for validation and testing.
# If the training dataloaders have shuffle=True, Lightning will automatically disable it.
#
# Useful for quickly debugging or trying to overfit on purpose.

# %% id="NTM-dqGMnXms"
# use only 1% of the train set (and use the train set for val and test)
trainer = Trainer(max_epochs=MAX_EPOCHS, overfit_batches=0.01)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% id="c0LV0gC3nl1X"
# overfit on 10 of the same batches
trainer = Trainer(max_epochs=MAX_EPOCHS, overfit_batches=10)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="lt3UHU6WgtS_"
# Or a float to represent percentage of data to run

# %% id="K3yUqADhgnkf"
# run through only 25% of the test set each epoch
trainer = Trainer(max_epochs=MAX_EPOCHS, limit_test_batches=0.25)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="ODN66NeVg_2o"
# In the case of multiple test dataloaders, the limit applies to each dataloader individually.
#

# %% [markdown] id="8aQx5SLeMz1R"
# # accumulate_grad_batches
#

# %% [markdown] id="g8GczZXFwKC7"
# The batch size controls the accuracy of the estimate of the gradients. Small batch size use less memory,
# but decrease accuracy. When training large models, such as NLP transformers, it is useful to accumulate
# gradients before calling backwards(). It allows for bigger batch sizes than what can actually
# fit on a GPU/TPU in a single step.
#
# Use accumulate_grad_batches to accumulate gradients every k batches or as set up in the dict.
# Trainer also calls optimizer.step() for the last indivisible step number.
#
# For example, set accumulate_grad_batches to 4 to accumulate every 4 batches.
# In this case the effective batch size is batch_size*4, so if your batch size is 32, effectively it will be 128.

# %% id="2jB6-Z_yPhhf"
# accumulate every 4 batches (effective batch size is batch*4)
trainer = Trainer(max_epochs=MAX_EPOCHS, accumulate_grad_batches=4)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="_Yi-bdTOgINC"
# You can also pass a dictionary to specify different accumulation per epoch. We can set it to `{5: 3, 10: 20}`
# to have no accumulation for epochs 1 to 4, accumulate 3 batches for epoch 5 to 10, and 20 batches after that.

# %% id="X3xsoZ3YPgBv"
# no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
trainer = Trainer(max_epochs=MAX_EPOCHS, accumulate_grad_batches={5: 3, 10: 20})

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="myzH8mV4M1_9"
# # 16 bit precision
#
#

# %% [markdown] id="v9EaFAonwOk6"
# Most deep learning frameworks like PyTorch, train with 32-bit floating point arithmetic.
#
# But many models can still achieve full accuracy using half the precision.
#
# In 2017, NVIDIA researchers successfully used a combination of 32 and 16 bit precision
# (also known as mixed precision) and achieved the same accuracy as 32 bit precision training.
#
# The main two advantages are:
#
# - a reduction in memory requirements which enables larger batch sizes and models.
# - and a speed up in compute. On ampere, turing and volta architectures 16 bit precision models can train at least 3 times faster.
#
# As of PyTorch 1.6, NVIDIA and Facebook moved mixed precision functionality into PyTorch core as the AMP package,
# torch.cuda.amp.
#
# This package supersedes the apex package developed by NVIDIA.

# %% [markdown] id="TjNypZPHnxvJ"
# ## precision
#
# Use precision flag to switch between full precision (32) to half precision (16). Can be used on CPU, GPU or TPUs.
#
# When using PyTorch 1.6+ Lightning uses the native amp implementation to support 16-bit.
#
# If used on TPU will use torch.bfloat16 but tensor printing will still show torch.float32

# %% id="kBZKMVx1nw-D"
# 16-bit precision
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1, precision=16)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="VJGj3Jh7oQXU"
# In earlier version of Lightning, we use NVIDIA Apex for 16-bit precision.
# Apex was the first library to attempt 16-bit and the automatic mixed precision library (amp),
# has since been merged into core PyTorch as of 1.6.
#
# If you insist in using Apex, you can set the amp_backend flag to 'apex' and install Apex on your own.

# %% id="BDV1trAUPc9h"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1, precision=16, amp_backend='apex')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="HK5c_aVfNV4e"
# ## amp_level
# Apex includes 4 optimization levels:
# O0 (FP32 training)
# O1 (Conservative Mixed Precision): only some whitelist ops are done in FP16.
# O2 (Fast Mixed Precision): this is the standard mixed precision training.
# It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights.
# O3 (FP16 training): full FP16.
# Passing keep_batchnorm_fp32=True can speed things up as cudnn batchnorm is faster anyway.
#

# %% id="FshMFPowNbWt"
# default used by the Trainer
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1, precision=16, amp_backend='apex', amp_level='O2')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="y8KEr1YvNgkC"
# # `auto_scale_batch_size`
#

# %% [markdown] id="7F1pKFIuwSFl"
# Lightning can help you improve your model by using auto_scale_batch_size flag,
# which tries to find the largest batch size that fits into memory, before you start your training.
# Larger batch size often yields better estimates of gradients, but may also result in longer training time.
#
# Set it to True to initially run a batch size finder trying to find the largest batch size that fits into memory.
# The result will be stored in self.batch_size in the LightningModule.
#

# %% id="9_jE-iyyheIv"
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_scale_batch_size=True)

trainer.tune(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="yaHsJvwFhNJt"
# You can set the value to `power`. `power` scaling starts from a batch size of 1
# and keeps doubling the batch size until an out-of-memory (OOM) error is encountered.
#

# %% id="Qx0FbQrphgw1"
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_scale_batch_size='power')

trainer.tune(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="8bwgVF9zhZ75"
# You can also set it to `binsearch`, that continues to finetune the batch size by performing a binary search.
#

# %% id="QObXNs3yNrg9"
# run batch size scaling, result overrides hparams.batch_size
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_scale_batch_size='binsearch')

trainer.tune(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="5OWdhSsZjqW7"
# This feature expects that a batch_size field in the hparams of your model, i.e.,
# model.hparams.batch_size should exist and will be overridden by the results of this algorithm.
#
# Additionally, your train_dataloader() method should depend on this field for this feature to work.
#
# The algorithm in short works by:
# 1. Dumping the current state of the model and trainer
#
# 2. Iteratively until convergence or maximum number of tries max_trials (default 25) has been reached:
#   * Call fit() method of trainer. This evaluates steps_per_trial (default 3) number of training steps.
#   Each training step can trigger an OOM error if the tensors (training batch, weights, gradients etc.)
#   allocated during the steps have a too large memory footprint.
#   * If an OOM error is encountered, decrease the batch size
#   * Else increase it.
# * How much the batch size is increased/decreased is determined by the chosen strategy.
#
# 3. The found batch size is saved to model.hparams.batch_size
#
# 4. Restore the initial state of model and trainer
#
#

# %% [markdown] id="q4CvxfZmOWBd"
# # `auto_lr_find`
#
#
#
#

# %% [markdown] id="j85e8usNwdBV"
# Selecting a good learning rate for your deep learning training is essential for both better performance
# and faster convergence.
#
# Even optimizers such as Adam that are self-adjusting the learning rate can benefit from more optimal choices.
#
# To reduce the amount of guesswork concerning choosing a good initial learning rate, you can use Lightning
# auto learning rate finder.
#
# The learning rate finder does a small run where the learning rate is increased after each processed batch
# and the corresponding loss is logged. The result of this is a lr vs. loss plot that can be used as guidance
# for choosing an optimal initial lr.
#
#
# warning: For the moment, this feature only works with models having a single optimizer.
# LR support for DDP is not implemented yet, it is coming soon.
#
#
# ***auto_lr_find=***
#
# In the most basic use case, this feature can be enabled during trainer construction with Trainer(auto_lr_find=True).
# When .fit(model) is called, the LR finder will automatically run before any training is done.
# The lr that is found and used will be written to the console and logged together with all other
# hyperparameters of the model.

# %% id="iuhve9RBOfFh"
# default used by the Trainer (no learning rate finder)
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_lr_find=False)

# %% [markdown] id="BL-gjXNCPDXk"
# This flag sets your learning rate which can be accessed via self.lr or self.learning_rate.
#


# %% id="wEb-vIMmPJQf"
class LitModel(LightningModule):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr or self.learning_rate))


# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_lr_find=True)

trainer.tune(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="RweqvpnVPPSh"
# To use an arbitrary value set it as auto_lr_find
#

# %% id="4LKI39IfPLJv"
trainer = Trainer(max_epochs=MAX_EPOCHS, auto_lr_find='0.01')

trainer.tune(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="9VAhPRKbPX-m"
# Under the hood, when you call tune it runs the learning rate finder.
#
# If you want to inspect the results of the learning rate finder before doing any actual training or just play around
# with the parameters of the algorithm, this can be done by invoking the lr_find method of the trainer.
# A typical example of this would look like
#
#
# ```
# trainer = Trainer(auto_lr_find=True)
#
# # Run learning rate finder
# lr_finder = trainer.lr_find(model)
#
# # Results can be found in
# lr_finder.results
#
# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()
#
# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
#
# # update hparams of the model
# model.hparams.lr = new_lr
#
# # Fit model
# trainer.fit(model)
# ```
#
# The figure produced by lr_finder.plot() should look something like the figure below.
# It is recommended to not pick the learning rate that achieves the lowest loss,
# but instead something in the middle of the sharpest downward slope (red point).
# This is the point returned py lr_finder.suggestion().
#
# ![image.png](auto-lr-find.png)

# %% [markdown] id="tn1RV-jfOjt1"
# # `benchmark`
#
#

# %% [markdown] id="rsmTl5zfwjM3"
# You can try to speed your system by setting `benchmark=True`, which enables cudnn.benchmark.
# This flag is likely to increase the speed of your system if your input sizes don’t change.
# This flag makes cudnn auto-tuner look for the optimal set of algorithms for the given hardware configuration.
# This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears,
# possibly leading to worse runtime performances.

# %% id="dWr-OCBgQCeb"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1, benchmark=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="qwAvSKYGa24K"
# # `deterministic`
#
#

# %% [markdown] id="tl5mfmafwmat"
# PyTorch does not guarantee reproducible results, even when using identical seeds.
# To guarentee reproducible results, you can remove most of the randomness from your process
# by setting the `deterministic` flag to True.
#
# Note that it might make your system slower.

# %% id="Mhv5LZ3HbNCK"
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=1, deterministic=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="u_5eJSvTf60f"
# # Exploding and vanishing gradients

# %% [markdown] id="B6drjh4pq6Jv"
# ## track_grad_norm
#
# You can debug your grad norm to identify exploding or vanishing gradients using the `track_grad_norm` flag.
#
# Set value to 2 to track the 2-norm. or p to any p-norm.

# %% id="2taHUir8rflR"
# track the 2-norm
trainer = Trainer(max_epochs=MAX_EPOCHS, track_grad_norm=2)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="3vHKxmruk62f"
# May be set to ‘inf’ infinity-norm.

# %% id="g7TbD6SxlAjP"
trainer = Trainer(max_epochs=MAX_EPOCHS, track_grad_norm='inf')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="TcMlRe7ywpe6"
# ## Gradient clipping
#
# Exploding gradients refer to the problem that the gradients get too large and overflow in training,
# making the model unstable. Gradient clipping will ‘clip’ the gradients or cap them to a Threshold
# value to prevent the gradients from getting too large. To avoid this, we can set `gradient_clip_val`
# (default is set to 0.0).
#
# [when to use it, what are relevant values]

# %% id="jF9JwmbOgOWF"
trainer = Trainer(max_epochs=MAX_EPOCHS, gradient_clip_val=0.1)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="ggb4MkkQrr1h"
# # truncated_bptt_steps
#
#

# %% [markdown] id="s1Iu6PyAw9_r"
# If you have a large recurrent model, you can use truncated_bptt_steps flag to split up the backprop over
# portions of the sequence. This flag will automatically truncate your batches and the trainer will apply
# Truncated Backprop to it.
#
# Make sure your batches have a sequence dimension.
#
# Lightning takes care of splitting your batch along the time-dimension.
# ```
# # we use the second as the time dimension
# # (batch, time, ...)
# sub_batch = batch[0, 0:t, ...]
# Using this feature requires updating your LightningModule’s pytorch_lightning.core.LightningModule.training_step()
# to include a hiddens arg with the hidden
#
# # Truncated back-propagation through time
# def training_step(self, batch, batch_idx, hiddens):
#     # hiddens are the hiddens from the previous truncated backprop step
#     out, hiddens = self.lstm(data, hiddens)
#
#     return {
#         "loss": ...,
#         "hiddens": hiddens  # remember to detach() this
#     }
# ```

# %% id="WiTF1VMtruMU"
# backprop every 5 steps in a batch
trainer = Trainer(max_epochs=MAX_EPOCHS, truncated_bptt_steps=5)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="8XI_kEWkS-nT"
# To modify how the batch is split, override pytorch_lightning.core.LightningModule.tbptt_split_batch():
#
# ```
# class LitMNIST(LightningModule):
#     def tbptt_split_batch(self, batch, split_size):
#         # do your own splitting on the batch
#         return splits
# ```
#

# %% [markdown] id="oLbEmbmupwQ8"
# # reload_dataloaders_every_epoch
#

# %% [markdown] id="CLdNGVv9xD_L"
# Set to True to reload dataloaders every epoch (instead of loading just once in the beginning of training).
#
# ```
# # if False (default)
# train_loader = model.train_dataloader()
# for epoch in epochs:
#     for batch in train_loader:
#         ...
#
# # if True
# for epoch in epochs:
#     train_loader = model.train_dataloader()
#     for batch in train_loader:
#
# ```

# %% id="10AXthXxp311"
trainer = Trainer(max_epochs=MAX_EPOCHS, reload_dataloaders_every_epoch=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="f513EYl0bmmL"
# # Callbacks
#

# %% [markdown] id="2pt7iGh4xNs5"
#
# Lightning Callbacks are self-contained programs that can be reused across projects.
# Callbacks should capture NON-ESSENTIAL logic that is NOT required for your LightningModule to run.
# Lightning includes some a few built-in callbacks that can be used with flags like early stopping
# and Model Checkpointing, but you can also create your own callbacks to add any functionality to your models.
#
# The callback API includes hooks that allow you to add logic at every point of your training:
# setup, teardown, on_epoch_start, on_epoch_end, on_batch_start, on_batch_end, on_init_start, on_keyboard_interrupt etc.
#
#

# %% [markdown] id="1t84gvDNsUuh"
# ## callbacks
#
# Use **callbacks=** to pass a list of user defined callbacks. These callbacks DO NOT replace the built-in callbacks
# (loggers or EarlyStopping).
#
# In this example, we create a dummy callback that prints a message when training starts and ends, using on_train_start
# and on_train_end hooks.

# %% id="oIXZYabub3f0"
from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


# a list of callbacks
callbacks = [PrintCallback()]
trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=callbacks)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="cNF74CLYfJJu"
# # Model checkpointing
#
#

# %% [markdown] id="2blgquBrxLtS"
# Checkpoints capture the exact value of all parameters used by a model.
#
# Checkpointing your training allows you to resume a training process in case it was interrupted,
# fine-tune a model or use a pre-trained model for inference without having to retrain the model.
#
# Lightning automates saving and loading checkpoints so you restore a training session,
# saving all the required parameters including:
# * 16-bit scaling factor (apex)
# * Current epoch
# * Global step
# * Model state_dict
# * State of all optimizers
# * State of all learningRate schedulers
# * State of all callbacks
# * The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)
#
# By default Lightning will save a checkpoint in the working directory, which will be updated every epoch.
#
# ### Automatic saving
# By default Lightning will save a checkpoint in the end of the first epoch in the working directory,
# which will be updated every epoch.

# %% id="XGu0JULrg9l7"
# default used by the Trainer
trainer = Trainer(max_epochs=MAX_EPOCHS, default_root_dir=os.getcwd())

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="3s9OjkGuhq1W"
# To change the checkpoint path pass in **default_root_dir=**

# %% id="DgdxkrIQhvfw"
trainer = Trainer(max_epochs=MAX_EPOCHS, default_root_dir='.')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="Qyvj_bkWrJiE"
#
# You can also have Lightning update your checkpoint based on a specific metric that you are logging (using self.log),
# by passing the key to `monitor=`. For example, if we want to save checkpoint based on the validation loss,
# logged as `val_loss`, you can pass:
#
#
# ```
# checkpoint_callback = ModelCheckpoint(
#     filepath=os.getcwd(),
#     save_top_k=1,
#     verbose=True,
#     monitor='val_loss',
#     mode='min',
#     prefix=''
# )
# ```
#

# %% id="YzYMivw1rO1O"
from pytorch_lightning.callbacks import ModelCheckpoint

trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[ModelCheckpoint(monitor='val_loss')])

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="5hYs_FV8iDMn"
# You can modify the behavior of checkpointing by creating your own callback, and passing it to the trainer.
# You can control
# * filepath- where logs are saved
# * save_top_k- save k top models
# * verbose
# * monitor- the metric to monitor
# * mode
# * prefix
#
#

# %% id="Tb1K2VYDiNTu"
from pytorch_lightning.callbacks import ModelCheckpoint

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=3,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='',
)

trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[checkpoint_callback])

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="YKhZ6xRojJcl"
# You can disable checkpointing it by passing
#
#

# %% id="Yt8zd2ZFjOXX"
trainer = Trainer(max_epochs=MAX_EPOCHS, checkpoint_callback=False)

# %% [markdown] id="HcLy8asCjrj9"
# ### Manual saving
#
# You can manually save checkpoints and restore your model from the checkpointed state.
#

# %% id="kZSkMJf0jR4x"
trainer.fit(model)
trainer.save_checkpoint("example.ckpt")
new_model = LitAutoEncoder.load_from_checkpoint(checkpoint_path="example.ckpt")

# %% [markdown] id="X2d9cjVPj7CP"
# ### Checkpoint Loading
# To load a model along with its weights, biases and module_arguments use following method:
#

# %% id="BpAFfg5zkFmH"
model = LitAutoEncoder.load_from_checkpoint("example.ckpt")

print(model.learning_rate)

# prints the learning_rate you used in this checkpoint

# %% [markdown] id="jTQ3mxSJkhFN"
# But if you don’t want to use the values saved in the checkpoint, pass in your own here


# %% id="IoMcOh9-kfUP"
class LitAutoEncoder(LightningModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.Linear(self.hparams.in_dim, self.hparams.out_dim)


# %% [markdown] id="ITPVY8mNknut"
# you can restore the model like this
#
#

# %% id="H7XeRJzVkuY8"
# if you train and save the model like this it will use these values when loading
# the weights. But you can overwrite this
LitAutoEncoder(in_dim=32, out_dim=10)

# uses in_dim=32, out_dim=10
model = LitAutoEncoder.load_from_checkpoint("example.ckpt")

# %% id="14WwGpnVk0a4"
# uses in_dim=128, out_dim=10
model = LitAutoEncoder.load_from_checkpoint("example.ckpt", in_dim=128, out_dim=10)

# %% [markdown] id="bY5s6wP_k1CU"
#
#
# ## Restoring Training State (resume_from_checkpoint)
# If your training was cut short for some reason, you can resume exactly from where you left off using
# the `resume_from_checkpoint` flag, which will automatically restore model, epoch, step, LR schedulers, apex, etc...

# %% id="9zfhHtyrk3rO"
model = LitAutoEncoder()
trainer = Trainer(max_epochs=MAX_EPOCHS, resume_from_checkpoint="example.ckpt")

# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model)

# %% [markdown] id="xkKdvALFsmT2"
# ## weights_save_path
# You can specify a directory for saving weights file using `weights_save_path`.
#
# (If you are using a custom checkpoint callback, the checkpoint callback will override this flag).

# %% id="9OwHHFcCsrgT"
# save to your custom path
trainer = Trainer(max_epochs=MAX_EPOCHS, weights_save_path='.')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% id="PbNtlJ9Wsscf"
# if checkpoint callback used, then overrides the weights path
# **NOTE: this saves weights to some/path NOT my/path
checkpoint = ModelCheckpoint(filepath='.')
trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[checkpoint], weights_save_path='.')
trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="uDdxCuyHdWQt"
# # Early stopping
#

# %% [markdown] id="fqAy3ihRxTfR"
# The EarlyStopping callback can be used to monitor a validation metric and stop the training when no improvement
# is observed, to help you avoid overfitting.
#
# To enable Early Stopping you can init the EarlyStopping callback, and pass it to `callbacks=` trainer flag.
# The callback will look for a logged metric to early stop on.
#
#

# %% id="lFx976CheH93"
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[EarlyStopping('val_loss')])

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="MwpJfTvjeOwF"
# You can customize the callback using the following params:
#

# %% id="V6I9h6HteK2U"
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.00, patience=3, verbose=False, mode='max')
trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stop_callback])

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="7TAIerPYe_Q1"
# The EarlyStopping callback runs at the end of every validation check, which, under the default configuration,
# happens after every training epoch. However, the frequency of validation can be modified by setting various
# parameters on the Trainer, for example check_val_every_n_epoch and val_check_interval.
# It must be noted that the patience parameter counts the number of validation checks with no improvement,
# and not the number of training epochs.
# Therefore, with parameters check_val_every_n_epoch=10 and patience=3,
# the trainer will perform at least 40 training epochs before being stopped.

# %% [markdown] id="VoKrX2ENh9Fg"
# # Logging

# %% [markdown] id="-CQTPKd7iKLm"
# Lightning has built in integration with various loggers such as TensorBoard, wandb, commet, etc.
#
#
# You can pass any metrics you want to log during training to `self.log`, such as loss or accuracy.
# Similarly, pass in to self.log any metric you want to log during validation step.
#
# These values will be passed in to the logger of your choise. simply pass in any supported
# logger to logger trainer flag.
#
#
# Use the as`logger=` trainer flag to pass in a Logger, or iterable collection of Loggers, for experiment tracking.
#

# %% id="ty5VPS3AiS8L"
from pytorch_lightning.loggers import TensorBoardLogger

# default logger used by trainer
logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name='lightning_logs')
trainer = Trainer(max_epochs=MAX_EPOCHS, logger=logger)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="jc5oWNpoiuuc"
# Lightning supports the use of multiple loggers, just pass a list to the Trainer.
#
#

# %% id="BlYwMRRyivp_"
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger

logger1 = TensorBoardLogger('tb_logs', name='my_model')
logger2 = TestTubeLogger('tb_logs', name='my_model')
trainer = Trainer(max_epochs=MAX_EPOCHS, logger=[logger1, logger2])

# %% [markdown] id="a7EyspQPh7iQ"
# ## flush_logs_every_n_steps
#
# Use this flag to determine when logging to disc should happen.

# %% id="Em_XvsmyiBbk"
trainer = Trainer(max_epochs=MAX_EPOCHS, flush_logs_every_n_steps=100)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="_vDeKE98qsl1"
# ## log_every_n_steps
# How often to add logging rows (does not write to disk)
#
#

# %% id="HkqD7D_0w1Tt"
trainer = Trainer(max_epochs=MAX_EPOCHS, log_every_n_steps=1000)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="9uw0gfe422CT"
# # info logging

# %% [markdown] id="dQXpt0aatDGo"
# ### default_root_dir
#
# ---
#
# Default path for logs and weights when no logger or pytorch_lightning.callbacks.ModelCheckpoint callback passed.
# On certain clusters you might want to separate where logs and checkpoints are stored. If you don’t then use this
# argument for convenience. Paths can be local paths or remote paths such as s3://bucket/path or ‘hdfs://path/’.
# Credentials will need to be set up to use remote filepaths.

# %% [markdown] id="CMmID2Bts5W3"
# ## weights_summary
# Prints a summary of the weights when training begins. Default is set to `top`- print summary of top level modules.
#
# Options: ‘full’, ‘top’, None.

# %% id="KTl6EdwDs6j2"

# print full summary of all modules and submodules
trainer = Trainer(max_epochs=MAX_EPOCHS, weights_summary='full')

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% id="R57cSLl9w9ma"
# don't print a summary
trainer = Trainer(max_epochs=MAX_EPOCHS, weights_summary=None)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="bSc2hU5AotAP"
# # progress bar

# %% [markdown] id="GgvbyDsBxcH6"
# ## process_position
#
# Orders the progress bar. Useful when running multiple trainers on the same node.
#
# (This argument is ignored if a custom callback is passed to callbacks)
#
#

# %% id="6ekz8Es8owDn"
# default used by the Trainer
trainer = Trainer(max_epochs=MAX_EPOCHS, process_position=0)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="itivQFgEphBU"
# ## progress_bar_refresh_rate
#
# How often to refresh the progress bar (in steps). In notebooks, faster refresh rates (lower number)
# is known to crash them because of their screen refresh rates, so raise it to 50 or more.

# %% id="GKe6eVxmplL5"
# default used by the Trainer
trainer = Trainer(max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=1)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% id="8rDHJOJbxNtf"
# disable progress bar
trainer = Trainer(max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=0)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="NCNvYLwjpWne"
# # profiler

# %% id="pRknrG_zpY6M"
# to profile standard training events
trainer = Trainer(max_epochs=MAX_EPOCHS, profiler=True)

trainer.fit(LitAutoEncoder(), datamodule=mnist)

# %% [markdown] id="Ji6aWpU73kMM"
# You can also use Lightning AdvancedProfiler if you want more detailed information about time spent in each function
# call recorded during a given action. The output is quite verbose and you should only use this if you want very
# detailed reports.
#
#

# %% id="layG55pt316C"
from pytorch_lightning.profiler import AdvancedProfiler

trainer = Trainer(max_epochs=MAX_EPOCHS, profiler=AdvancedProfiler())

trainer.fit(LitAutoEncoder(), datamodule=mnist)
