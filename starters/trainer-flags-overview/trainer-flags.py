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

# %% [markdown]
# <a href="https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/05-trainer-flags-overview.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="goRmGIRI5cfC"
# # Introduction to Lightning Flags ⚡🚩
#
# In this notebook, we'll go over the flags available in the `Trainer` object. Note that not everything will work in the Colab environment (multi-gpu, etc). This notebook accompanies the Trainer videos we'll be putting out.
#
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

# %% [markdown] id="jKj5lgdr5j48"
# ---
# ### Setup
# First thing first, we need to install Lightning. Simply ```pip install pytorch-lightning```

# %% id="UGjilEHk4vb7"
# ! pip install pytorch-lightning

# %% id="zaVUShmQ5n8Y"
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import LightningModule, Trainer, seed_everything

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

# %% id="6tgkS8IYZwY_"
# ------------
# data
# ------------
seed_everything(1234)
batch_size = 32

# Init DataLoader from MNIST Dataset

dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=batch_size)
val_loader = DataLoader(mnist_val, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

# %% [markdown] id="gEulmrbxwaYL"
# ### Simple AutoEncoder Model
#
# Were gonna define a simple Lightning model so we can play with all the settings of the Lightning Trainer.
#
# LightningModule is simply pure Pytorch reorganized into hooks, that represents all the steps in the training process.
#
# You can use LightningModule hooks to control every part of your model, but for the purpose of this video we will use a very simple MNIST classifier, a model that takes 28*28 grayscale images of hand written images, and can predict the digit between 0-9.
#
# The LightningModule can encompass a single model, like an image classifier, or a deep learning system composed of multiple models, like this auto encoder that contains an encoder and a decoder.
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
# You'll notice the LightningModule doesn't have epoch and batch loops, we're not calling model.train() and model.eval(), and no mentions of CUDA or hardware. That's because it is all automated by the Lightning Trainer. All the engineering boilerplate is automated by the trainer:
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
#
# 2. init lightning trainer
#
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
trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=2)

#####################
# 3. Train
#####################
trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="3meDako-Qa_6"
# Our model is training just like that, using the Lightning defaults. The beauty of Lightning is that everything is easily configurable.
# In our next videos were going to show you all the ways you can control your Trainer to do things like controlling your training, validation and test loops, running on GPUs and TPUs, checkpointing, early stopping, and a lot more.
#

# %% [markdown] id="z_Wry2MckQkI"
# # Training loop and eval loop Flags

# %% [markdown] id="0MkI1xB2vsLj"
#
# To really scale up your networks, you can use accelerators like GPUs. GPUs or Graphical Processing Units, parallelize matrix multiplications which enable speed ups of at least 100x over training on CPUs.
#
# Let's say you have a machine with 8 GPUs on it. You can set this flag to 1, 4, or 8 GPUs and lightning will automatically distribute your training for you.
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
# 2. Second, when you initialize a new tensor, set the device=self.device in the call since every lightningModule knows what gpu index or TPU core it is on.
#
# You can also use type_as and or you can register the tensor as a buffer in your module’s __init__ method with register_buffer().
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
# Lightning Trainer automates all the engineering boilerplate like iterating over epochs and batches, training eval and test loops, CUDA and to(device) calls, calling model.train and model.eval.
#
# You still have full control over the loops, by using the following trainer flags:

# %% [markdown] id="pT5-ETH9eUg6"
# ## Calling validation steps
# Sometimes, training an epoch may be pretty fast, like minutes per epoch. In this case, you might not need to validate on every epoch. Instead, you can actually validate after a few epochs.
#
# Use `check_val_every_n_epoch` flag to control the frequency of validation step:

# %% id="Z-EMVvKheu3D"
# run val loop every 10 training epochs
trainer = Trainer(check_val_every_n_epoch=10)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="UOzZr9S2UcSO"
# ## val_check_interval
#
# In some cases where your epoch is very long, you might want to check validation within an epoch.
#
# You can also run validation step within your training epochs, by setting `val_check_interval` flag.
#
# Set `val_check_interval` to a float between [0.0 to 1.0] to check your validation set within a training epoch. For example, setting it to 0.25 will check your validation set 4 times during a training epoch.
#
# Default is set to 1.0

# %% id="9kbUbvrUVLrT"
# check validation set 4 times during a training epoch
trainer = Trainer(val_check_interval=0.25)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Onm1gBsKVaw4"
# When you have iterable data sets, or when streaming data for production use cases, it is useful to check the validation set every number of steps.
# Set val_check_interval to an int:

# %% id="psn6DVb5Vi85"
# check validation set every 1000 training batches
# use this when using iterableDataset and your dataset has no length
# (ie: production cases with streaming data)
trainer = Trainer(val_check_interval=1000)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="QkoYonrWkb7-"
# ## num_sanity_val_steps
#
# You may have run into an issue, where you have a bug in your validation loop, but won't catch it until your training loop ends.
#
# and if your training loop takes hours or days, you will waste valuable compute.
#
# Instead, lightning automatically runs through 2 steps of validation in the beginning to catch these kinds of bugs up front.
#
#
# The `num_sanity_val_steps` flag can help you run n batches of validation before starting the training routine.
#
# You can set it to 0 to turn it off

# %% id="zOcT-ugSkiKW"
# turn it off
trainer = Trainer(num_sanity_val_steps=0)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="zS0ob1ZmTw56"
# Set it to -1 to check all validation data before training

# %% id="rzqvjA4UT263"
# check all validation data
trainer = Trainer(num_sanity_val_steps=-1)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="uMB41wq4T3Z2"
# Or use any arbitrary number of validation steps

# %% id="lGP78aQzT7VS"
trainer = Trainer(num_sanity_val_steps=10)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="H-xaYRtd1rb-"
# ## Limit train, validation, and test batches
#
# You can set limits on how much of training, validation and test dataset you want your model to check. This is useful if you have really large validation or tests sets, for debugging or testing something that happens at the end of an epoch.
#
# Set the flag to int to specify the number of batches to run
#
#

# %% id="XiK5cFKL1rcA"
# run for only 10 batches
trainer = Trainer(limit_test_batches=10)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Y4LK0g65RrBm"
# For example, some metrics need to be computed on the entire validation results, such as AUC ROC.

# %% id="8MmeRs2DR3dD"
trainer = Trainer(limit_val_batches=10)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="xmigcNa1A2Vy"
# You can use a float to limit the batches be percentage of the set on every epoch

# %% id="W7uGJt8nA4tv"
# run through only 25% of the test set each epoch
trainer = Trainer(limit_test_batches=0.25)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="YRI8THtUN7_e"
# # Training on GPUs
#
#

# %% [markdown] id="R8FFkX_FwlfE"
# To run on 1 GPU set the flag to 1

# %% id="Nnzkf3KaOE27"
trainer = Trainer(gpus=1)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="cxBg47s5PB1P"
# to run on 2 or 4 GPUs, set the flag to 2 or 4.

# %% id="cSEM4ihLrohT"
trainer = Trainer(gpus=2)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="ZE6ZgwtNudro"
# You can also select which GPU devices to run on, using a list of indices like [1, 4]
#
# or a string containing a comma separated list of GPU ids like '1,2'
#

# %% id="gQkJtq0urrjq"
# list: train on GPUs 1, 4 (by bus ordering)
# trainer = Trainer(gpus='1, 4') # equivalent
trainer = Trainer(gpus=[1, 4])

trainer.fit(model, train_loader, val_loader)

# %% id="XghDPad4us74"
trainer = Trainer(gpus=list(range(4)))

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="6FVkKHpSPMTW"
# You can use all the GPUs you have available by setting `gpus=-1`

# %% id="r6cKQijYrtPe"
# trainer = Trainer(gpus='-1') - equivalent
trainer = Trainer(gpus=-1)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="2C-fNLm3UGCV"
# Lightning uses the PCI bus_id as the index for ordering GPUs.

# %% [markdown] id="_V75s7EhOFhE"
# ### `auto_select_gpus`
#
# You can save on GPUs by running in “exclusive mode”, meaning only one process at a time can access them. If your not sure which GPUs you should use when running exclusive mode, Lightning can automatically find unoccupied GPUs for you.
#
# Simply specify the number of gpus as an integer `gpus=k`, and set the trainer flag `auto_select_gpus=True`. Lightning will automatically help you find k gpus that are not occupied by other processes.

# %% id="_Sd3XFsAOIwd"
# enable auto selection (will find two available gpus on system)
trainer = Trainer(gpus=2, auto_select_gpus=True)

trainer.fit(model, train_loader, val_loader)

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
# Your logs can become overwhelmed if you log the usage from many GPUs at once. In this case, you can also set the flag to min_max which will log only the min and max usage across all the GPUs of the master node.
#
# Note that lightning is not logging the usage across all nodes for performance reasons.

# %% id="idus3ZGahOki"
# log all the GPUs (on master node only)
trainer = Trainer(log_gpu_memory='all')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="-mevgiy_hkip"
# To avoid the performance decrease you can also set `log_gpu_memory=min_max` to only log the min and max memory on the master node.
#

# %% id="SlvLJnWyhs7J"
# log only the min and max memory on the master node
trainer = Trainer(log_gpu_memory='min_max')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="K82FLLIJVQG3"
#
# But what if you want to train on multiple machines and not just one?

# %% [markdown] id="YViQ6PXesAue"
# # Training on multiple GPUs

# %% [markdown] id="WacbBQUivxQq"
# Lightning makes your models hardware agnostic, and you can run on GPUs with a flip of a flag. Lightning also supports training on multiple GPUs across many machines.
#
# You can do this by setting the num_nodes flag.
#
# The world size, or the total number of GPUs you are using, will be gpus*num_nodes.
#
# If i set gpus=8 and num_nodes=32 then I will be training on 256 GPUs.

# %% id="5iKckmDvr8zZ"
trainer = Trainer(gpus=8, num_nodes=32)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="GgcSbDjjlSTh"
# ## Accelerators
#
# Under the hood, Lightning uses distributed data parallel (or DDP) by default to distribute training across GPUs.
#
# This Lightning implementation of DDP calls your script under the hood multiple times with the correct environment variables.
#
# Under the hood it's as if you had called your script like this:
#
# 1. Each GPU across each node gets its own process.
# 2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.
# 3. Each process inits the model. (Make sure to set the random seed so that each model initializes with the same weights.)
# 4. Each process performs a full forward and backward pass in parallel.
# 5. The gradients are synced and averaged across all processes.
# 6. Each process updates its optimizer.
# If you request multiple GPUs or nodes without setting a mode, DDP will be automatically used.
#

# %% id="n_Brr7F5wdtj"
# ddp = DistributedDataParallel
# trainer = Trainer(gpus=2, num_nodes=2) equivalent
trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="edxHyttC5J3e"
# DDP is the fastest and recommended way to distribute your training, but you can pass in other backends to `accelerator` trainer flag, when DDP is not supported.
#
# DDP isn't available in
# * Jupyter Notebook, Google COLAB, Kaggle, etc.
# * If You have a nested script without a root package
# * or if Your script needs to invoke .fit or .test multiple times

# %% [markdown] id="ZDh96mavxHxf"
# ### DDP_SPAWN
#
# In these cases, you can use `ddp_spawn` instead. `ddp_spawn` is exactly like DDP except that it uses `.spawn()` to start the training processes.

# %% id="JM5TKtgLxo37"
trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp_spawn')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="sebhVE3qrhKK"
# We STRONGLY discourage this use because it has limitations (due to Python and PyTorch):
#
# * Since .spawn() trains the model in subprocesses, the model on the main process does not get updated.
#
# * Dataloader(num_workers=N), where N is large, bottlenecks training with DDP… ie: it will be VERY slow or won’t work at all. This is a PyTorch limitation.
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
# If you're using windows, DDP is not supported. You can use `dp` for DataParallel instead: DataParallel uses multithreading, instead of multiprocessing. It splits a batch across k GPUs. That is, if you have a batch of 32 and use DP with 2 gpus, each GPU will process 16 samples, after which the root node will aggregate the results.
#
# DP use is discouraged by PyTorch and Lightning. Use DDP which is more stable and at least 3x faster.
#

# %% id="OO-J0ISvlVCg"
# dp = DataParallel
trainer = Trainer(gpus=2, accelerator='dp')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Y7E2eHZKwUn9"
# ### DDP2
#
# In certain cases, it’s advantageous to use ***all*** batches on the same machine, instead of a subset. For instance, in self-supervised learning, a common performance boost comes from increasing the number of negative samples.
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
trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp2')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="lhKNCnveeeq5"
# - The second mode is ddp_spawn. This works like ddp, but instead of calling your script multiple times, lightning will use multiprocessing spawn to start a subprocess per GPU.
#
# However, you should be careful of mixing this mode with num_workers > 0 in your dataloaders because it will bottleneck your training. This is a current known limitation of PyTorch which is why we recommend using our ddp implementation instead.
#

# %% [markdown] id="HUf9ANyQkFFO"
#
# ### mocking ddp
#
# Testing or debugging DDP can be hard, so we have a distributed backend that simulates ddp on cpus to make it easier. Set `num_processes` to a number greater than 1 when using accelerator="ddp_cpu" to mimic distributed training on a machine without GPUs. Note that while this is useful for debugging, it will not provide any speedup, since single-process Torch already makes efficient use of multiple CPUs.

# %% id="ZSal5Da9kHOf"
# Simulate DDP for debugging on your GPU-less laptop
trainer = Trainer(accelerator="ddp_cpu", num_processes=2)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Br_btCy5lgES"
# # Training on TPUS
#

# %% [markdown] id="DXkBNITdv44d"
# Another option for accelerating your training is using TPUs.
# A TPU is a Tensor processing unit, designed specifically for deep learning. Each TPU has 8 cores where each core is optimized for 128x128 matrix multiplies. Google estimates that 8 TPU cores are about as fast as 4 V100 GPUs!
#
# A TPU pod hosts many TPUs on it. Currently, TPU pod v2 has 2048 cores! You can request a full pod from Google cloud or a “slice” which gives you some subset of those 2048 cores.
#
# At this moment, TPUs are available on Google Cloud (GCP), Google Colab and Kaggle Environments.
#
# Lightning supports training on TPUs without any code adjustments to your model. Just like when using GPUs, Lightning automatically inserts the correct samplers - no need to do this yourself!
#
# Under the hood, lightning uses the XLA framework developed jointly by the facebook and google XLA teams. And we want to recognize their efforts in advancing TPU adoption of PyTorch.
#
# ## tpu_cores
# To train on TPUs, set the tpu_cores flag.
#
# When using colab or kaggle, the allowed values are 1 or 8 cores. When using google cloud, any value above 8 is allowed.
#
# Your effective batch size is the batch size passed into a dataloader times the total number of tpu cores.

# %% id="itP9y70gmD9M"
# int: train on a single core
trainer = Trainer(tpu_cores=1)

trainer.fit(model, train_loader, val_loader)

# %% id="NJKnzPb3mKEg"
# int: train on all cores few cores
trainer = Trainer(tpu_cores=8)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="8a4exfWUmOHq"
# You can also choose which TPU core to train on, by passing a list [1-8]. This is not an officially supported use case but we are working with the XLA team to improve this user experience.
#

# %% id="S6OrjE_bmT-_"
# list: train on a single selected core
trainer = Trainer(tpu_cores=[2])

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Afqx3sFUmfWD"
# To train on more than 8 cores (ie: a POD), submit this script using the xla_dist script.
#
#
#
# ```
# python -m torch_xla.distributed.xla_dist
# --tpu=$TPU_POD_NAME
# --conda-env=torch-xla-nightly
# --env=XLA_USE_BF16=1
# -- python your_trainer_file.py
# ```
#
#

# %% [markdown] id="ncPvbUVQqKOh"
# # Advanced distributed training
#

# %% [markdown] id="4MP7bEgnv7qK"
#
# Lightning supports distributed training across multiple GPUs and TPUs out of the box by setting trainer flags, but it also allows you to control the way sampling is done if you need to.

# %% [markdown] id="wdHiTfAMepKH"
# ## replace_sampler_ddp
# In PyTorch, you must use torch.nn.DistributedSampler for multi-node or GPU training. The sampler makes sure each GPU sees the appropriate part of your data.
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
# Lightning adds the correct samplers when needed, so no need to explicitly add samplers. By default it will add `shuffle=True` for train sampler and `shuffle=False` for val/test sampler.
#
# If you want to customize this behaviour, you can set `replace_sampler_ddp=False` and add your own distributed sampler.
#
# (note: For iterable datasets, we don’t do this automatically.)
#

# %% id="ZfmcB_e_7HbE"
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

trainer = Trainer(gpus=2, num_nodes=2, replace_sampler_ddp=False)

# %% [markdown] id="-IOhk1n0lL3_"
# ## prepare_data_per_node
#
# When doing multi NODE training, if your nodes share the same file system, then you don't want to download data more than once to avoid possible collisions.
#
# Lightning automatically calls the prepare_data hook on the root GPU of the master node (ie: only a single GPU).
#
# In some cases where your nodes don't share the same file system, you need to download the data on each node. In this case you can set this flag to true and lightning will download the data on the root GPU of each node.
#
# This flag is defaulted to True.

# %% id="WFBMUR48lM04"
trainer = Trainer(gpus=2, num_nodes=2, prepare_data_per_node=False)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="FKBwXqo4q-Vp"
# ## sync_batchnorm
#
# Batch norm is computed per GPU/TPU. This flag enables synchronization between batchnorm layers across all GPUs.
# It is recommended if you have small batch sizes.
#

# %% id="GhaCLTEZrAQi"
trainer = Trainer(gpus=4, sync_batchnorm=True)

trainer.fit(model, train_loader, val_loader)

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
# This won't generate logs or save checkpoints but will touch every line of your code to make sure that it is working as intended.
#
# Think about this flag like a compiler. You make changes to your code, and run Trainer with this flag to verify that your changes are bug free.
#

# %% id="L5vuG7GSmhzK"
trainer = Trainer(fast_dev_run=True)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="HRP1qQR5nT4p"
# ## overfit_batches
#
# Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it.
#
# Useful for quickly debugging or trying to overfit on purpose.

# %% id="NTM-dqGMnXms"
# use only 1% of the train set (and use the train set for val and test)
trainer = Trainer(overfit_batches=0.01)

trainer.fit(model, train_loader, val_loader)

# %% id="c0LV0gC3nl1X"
# overfit on 10 of the same batches
trainer = Trainer(overfit_batches=10)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="lt3UHU6WgtS_"
# Or a float to represent percentage of data to run

# %% id="K3yUqADhgnkf"
# run through only 25% of the test set each epoch
trainer = Trainer(limit_test_batches=0.25)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="ODN66NeVg_2o"
# In the case of multiple test dataloaders, the limit applies to each dataloader individually.
#

# %% [markdown] id="8aQx5SLeMz1R"
# # accumulate_grad_batches
#
#
#

# %% [markdown] id="g8GczZXFwKC7"
# The batch size controls the accuracy of the estimate of the gradients. Small batch size use less memory, but decrease accuracy. When training large models, such as NLP transformers, it is useful to accumulate gradients before calling backwards(). It allows for bigger batch sizes than what can actually fit on a GPU/TPU in a single step.
#
# Use accumulate_grad_batches to accumulate gradients every k batches or as set up in the dict. Trainer also calls optimizer.step() for the last indivisible step number.
#
# For example, set accumulate_grad_batches to 4 to accumulate every 4 batches. In this case the effective batch size is batch_size*4, so if your batch size is 32, effectively it will be 128.

# %% id="2jB6-Z_yPhhf"
# accumulate every 4 batches (effective batch size is batch*4)
trainer = Trainer(accumulate_grad_batches=4)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="_Yi-bdTOgINC"
# You can also pass a dictionary to specify different accumulation per epoch. We can set it to `{5: 3, 10: 20}` to have no accumulation for epochs 1 to 4, accumulate 3 batches for epoch 5 to 10, and 20 batches after that.

# %% id="X3xsoZ3YPgBv"
# no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="myzH8mV4M1_9"
# # 16 bit precision
#
#

# %% [markdown] id="v9EaFAonwOk6"
# Most deep learning frameworks like PyTorch, train with 32-bit floating point arithmetic.
#
# But many models can still achieve full accuracy using half the precision.
#
# In 2017, NVIDIA researchers successfully used a combination of 32 and 16 bit precision (also known as mixed precision) and achieved the same accuracy as 32 bit precision training.
#
# The main two advantages are:
#
# - a reduction in memory requirements which enables larger batch sizes and models.
# - and a speed up in compute. On ampere, turing and volta architectures 16 bit precision models can train at least 3 times faster.
#
# As of PyTorch 1.6, NVIDIA and Facebook moved mixed precision functionality into PyTorch core as the AMP package, torch.cuda.amp.
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
trainer = Trainer(gpus=1, precision=16)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="VJGj3Jh7oQXU"
# In earlier version of Lightning, we use NVIDIA Apex for 16-bit precision. Apex was the first library to attempt 16-bit and the automatic mixed precision library (amp), has since been merged into core PyTorch as of 1.6.
#
# If you insist in using Apex, you can set the amp_backend flag to 'apex' and install Apex on your own.

# %% id="BDV1trAUPc9h"
trainer = Trainer(gpus=1, precision=16, amp_backend='apex')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="HK5c_aVfNV4e"
# ## amp_level
# Apex includes 4 optimization levels:
# O0 (FP32 training)
# O1 (Conservative Mixed Precision): only some whitelist ops are done in FP16.
# O2 (Fast Mixed Precision): this is the standard mixed precision training. It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights.
# O3 (FP16 training): full FP16. Passing keep_batchnorm_fp32=True can speed things up as cudnn batchnorm is faster anyway.
#

# %% id="FshMFPowNbWt"
# default used by the Trainer
trainer = Trainer(gpus=1, precision=16, amp_backend='apex', amp_level='O2')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="y8KEr1YvNgkC"
# # `auto_scale_batch_size`
#
#
#
#

# %% [markdown] id="7F1pKFIuwSFl"
# Lightning can help you improve your model by using auto_scale_batch_size flag, which tries to find the largest batch size that fits into memory, before you start your training.
# Larger batch size often yields better estimates of gradients, but may also result in longer training time.
#
# Set it to True to initially run a batch size finder trying to find the largest batch size that fits into memory. The result will be stored in self.batch_size in the LightningModule.
#

# %% id="9_jE-iyyheIv"
trainer = Trainer(auto_scale_batch_size=True)

trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# %% [markdown] id="yaHsJvwFhNJt"
# You can set the value to `power`. `power` scaling starts from a batch size of 1 and keeps doubling the batch size until an out-of-memory (OOM) error is encountered.
#

# %% id="Qx0FbQrphgw1"
trainer = Trainer(auto_scale_batch_size='power')

trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# %% [markdown] id="8bwgVF9zhZ75"
# You can also set it to `binsearch`, that continues to finetune the batch size by performing a binary search.
#

# %% id="QObXNs3yNrg9"
# run batch size scaling, result overrides hparams.batch_size
trainer = Trainer(auto_scale_batch_size='binsearch')

trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# %% [markdown] id="5OWdhSsZjqW7"
# This feature expects that a batch_size field in the hparams of your model, i.e., model.hparams.batch_size should exist and will be overridden by the results of this algorithm.
#
# Additionally, your train_dataloader() method should depend on this field for this feature to work.
#
# The algorithm in short works by:
# 1. Dumping the current state of the model and trainer
#
# 2. Iteratively until convergence or maximum number of tries max_trials (default 25) has been reached:
# * Call fit() method of trainer. This evaluates steps_per_trial (default 3) number of training steps. Each training step can trigger an OOM error if the tensors (training batch, weights, gradients etc.) allocated during the steps have a too large memory footprint.
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
# Selecting a good learning rate for your deep learning training is essential for both better performance and faster convergence.
#
# Even optimizers such as Adam that are self-adjusting the learning rate can benefit from more optimal choices.
#
# To reduce the amount of guesswork concerning choosing a good initial learning rate, you can use Lightning auto learning rate finder.
#
# The learning rate finder does a small run where the learning rate is increased after each processed batch and the corresponding loss is logged. The result of this is a lr vs. loss plot that can be used as guidance for choosing an optimal initial lr.
#
#
# warning: For the moment, this feature only works with models having a single optimizer. LR support for DDP is not implemented yet, it is coming soon.
#
#
# ***auto_lr_find=***
#
# In the most basic use case, this feature can be enabled during trainer construction with Trainer(auto_lr_find=True).
# When .fit(model) is called, the LR finder will automatically run before any training is done. The lr that is found and used will be written to the console and logged together with all other hyperparameters of the model.

# %% id="iuhve9RBOfFh"
# default used by the Trainer (no learning rate finder)
trainer = Trainer(mnist_model, auto_lr_find=False)

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
trainer = Trainer(mnist_model, auto_lr_find=True)

trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# %% [markdown] id="RweqvpnVPPSh"
# To use an arbitrary value set it as auto_lr_find
#

# %% id="4LKI39IfPLJv"
trainer = Trainer(mnist_model, auto_lr_find='my_value')

trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# %% [markdown] id="9VAhPRKbPX-m"
# Under the hood, when you call tune it runs the learning rate finder.
#
# If you want to inspect the results of the learning rate finder before doing any actual training or just play around with the parameters of the algorithm, this can be done by invoking the lr_find method of the trainer. A typical example of this would look like
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
# The figure produced by lr_finder.plot() should look something like the figure below. It is recommended to not pick the learning rate that achieves the lowest loss, but instead something in the middle of the sharpest downward slope (red point). This is the point returned py lr_finder.suggestion().
#
# ![image.png](data:image/png;base64,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/SsqKsxRkqVLl/p+rGN+j8b66x0f7rzzTklLS3PMera2INFYf89nEQA9EqH9SwAMzctq6uZfAD2Eq+OaX7X773//25wLGOyH6RWAJ598sjkE4OTbwERr/fUP/xFHHCEaqJ08RGv99TCL3v5FTxF4+umnHUsQjfVftWqVOf9LzwHT9e/Tp4+r1t93ZRMtAEbi75+eL62nyOj219ugfPTRR74kjvo9GvU/ffp02W233by3wVKLRAnAkdj+uoH1IrjDDz/cnAN69NFHt+hPHVUEDlsYAmAMN0g0/gDEcPGtP4r1978IKFJ/AK03TIxmwPZn+/teBEf9R2YHQIy+vtYf4/bvvzVgFGZAAIwCamuzbP4FiMQh4NY+y4njWX//AMD2tz8Fwol13toyUf/Uv28A5vvvru9/a38X4jmeABhD/eYdgH60ngT9+OOPe5dCz+PS3djBXgTibZgAv7D+/h0g25/65/vP3z/+/ruj/3NiF00AjPJW0RPT9U7t+qMBSB/bpr8XFhaaT9bbwOj9/z7//HP56aef5OGHHza3gXHSrQxsiFh/tj/1z/efv3/8/Xdj/2fTd8aiLQEwysp6dZIWfvOfe++91/vJ77//vnmah94PUPcIzJkzx/teov/C+rP9m9e+vqb++f57/rbx94+///o0q2Ts/zw17tR/CYBO3TIsFwIIIIAAAgggECUBAmCUYJktAggggAACCCDgVAECoFO3DMuFAAIIIIAAAghESYAAGCVYZosAAggggAACCDhVgADo1C3DciGAAAIIIIAAAlESIABGCZbZIoAAAggggAACThUgADp1y7BcCCCAAAIIIIBAlAQIgFGCZbYIIIAAAggggIBTBQiATt0yLBcCCCCAAAIIIBAlAQJglGCZLQIIJIZAx44d5d13302MhWUpEUAAgQgJEAAjBMlsEECgdQF99NsNN9zQ+gRxfGfz5s1SXV0dxyVo+6OdbNf2kvMuAgg4WYAA6OStw7IhkCQC8QgxO3fudLResMsXDztHw7FwCCAQEQECYEQYmQkCCLQl0F6IWbZsmVx11VWy3377yRFHHCF33323lJaWemc5btw4ufDCC+Wggw6SQw89VK655hopKCjwvr9mzRpJSUmRr7/+Wi6++GLZa6+9ZNCgQeL53DfffFP+53/+x7R99NFHxTd8NT8ErPMZOHCg3HjjjbLPPvvICSecID/88IP3s/QXfa3j9XMuvfRS+fzzz83nl5eX+03n+0Ln269fP7nuuutk3333lbS0NGloaJAHHnhAOnXqJHvvvbeceOKJ0qdPH28znUbb+f5kZmaa94uKiuS2224zJocccohcf/31og4MCCCAQDACBMBglJgGAQSsBDxBLNBMNDR16NBBunXrJitWrJDs7Gy54oorpGvXrt7Jv/nmG/n2228lPz9fFi1aZEJUly5dpLGx0UzjCYAapHS61atXy4YNG0wAPPDAA+Xvf/+7mffo0aNN+BowYIB33oEC4DHHHCNfffWV+bx//OMfsv/++8uWLVtMG533HnvsIf/6178kNzdXhg0bJkcffXRQAVDD7WeffSarVq2SwsJCE0RfeOEFmT9/vlnmIUOGmOUbPny4+ayqqiq5/fbbTTjeuHGj6E9dXZ1pd8opp5jwuHTpUvnpp5/kL3/5i5x00knmfe/K8QsCCCDQigABsBUYRiOAQOQE2gqAL730klx55ZV+H1ZcXGwC1cqVK/3Ge17o3kHdK6Z7DnXwBEDfvWc6Xj9XA57uafMMutfsjjvu8Lw07/teBKLzfe6557zvb9++3XyW7oXU4dlnn5XTTz/d+77+0qNHj6AC4FNPPeXXLtCLxx57TG655RbvW4HsvvzySxP2mpqavNNpMNQ9lhkZGd5x/IIAAgi0JkAAbE2G8QggEDGBQCHGM/Nbb73V7FHTw7++PxrE0tPTzWR5eXly5513ynHHHScHHHCAmU7fHzt2rHnfEwBnzJjhma35Vz/36quv9hune/R89y4G2gM4YsQIvza6F/GLL74w4/TQ8P333+/3vh4S1uVp7xCw7uFrPnzwwQdy9tlny+GHH27WS/cunnvuud7JAtnp3sfdd9/dz0vtdtttN3OY2duYXxBAAIFWBAiArcAwGgEEIicQKMR45q7n/t18883mcKse4vX90b1vOuihTd1LOGnSJHO4MycnxwSu77//3rzvCYB6eNh3CPS5Tz75pFxyySXeyQIFQM98PRPpuYd6TqEONgGw+Xz18LGe+/fhhx+aQ9+67g8//LCceeaZno/2nsfoHSFiDmmfd955flYet4qKCt9J+R0BBBAIKEAADMjCSAQQiKRAoCDmmX/37t1NwKuvr/eM8vu3rKzMhL2srCzv+OnTp8ctAOohYD3/0HfQQ8bB7AFsHgAff/xx+eMf/+g7K7nsssv8AuBDDz0k1157rd80eg6jXvixbds2v/G8QAABBIIVIAAGK8V0CCAQtoAGQL1aVvfQ+f7olazr1683F4HooeB58+aZq3vHjx8v9913nzl3Ty/0OOyww8yVwbqXa/LkyeYQqQYuT6CK5R5Az0Ug//nPf0TPUdQLNvSiEV2etva++S6vB7Jv376ih5d1fXVeGiT1te8ewN69e8uxxx5rLjjRcx/1Cma9b2Hnzp2NqQZjXSa9OviJJ54QPX+SAQEEEGhPgADYnhDvI4CAtYAGQA1AzX8efPBBM289x++mm26Sgw8+2FzIcPLJJ4teMOG5yGHixImiV73qbVfOOOMMmTp1qplXPAKgLnDz28D079/fLE9NTU2rVoECYG1trQm6eohZ1/2RRx6R1NRUvwCoN6rWq6L1SmSdh+c2MHpF8D333GPOHVSX448/XnRvIXsFW90EvIEAAj4CBEAfDH5FAAEEwhF4+eWXzV7AcNrSBgEEEIiHAAEwHup8JgIIJLSAXrShh6v1fn6DBw82N2PWW8EwIIAAAokiQABMlC3FciKAgGME9PD0kUceaQ5J67l4vXr1ktYuYnHMQrMgCCCAgI8AAdAHg18RQAABBBBAAAE3CBAA3bCVWUcEEEAAAQQQQMBHgADog8GvCCCAAAIIIICAGwQIgG7YyqwjAggggAACCCDgI0AA9MHgVwQQQAABBBBAwA0C/w+ELQeExqjNywAAAABJRU5ErkJggg==)

# %% [markdown] id="tn1RV-jfOjt1"
# # `benchmark`
#
#

# %% [markdown] id="rsmTl5zfwjM3"
# You can try to speed your system by setting `benchmark=True`, which enables cudnn.benchmark. This flag is likely to increase the speed of your system if your input sizes don’t change. This flag makes cudnn auto-tuner look for the optimal set of algorithms for the given hardware configuration. This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.

# %% id="dWr-OCBgQCeb"
trainer = Trainer(gpus=1, benchmark=True)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="qwAvSKYGa24K"
# # `deterministic`
#
#

# %% [markdown] id="tl5mfmafwmat"
# PyTorch does not guarantee reproducible results, even when using identical seeds. To guarentee reproducible results, you can remove most of the randomness from your process by setting the `deterministic` flag to True.
#
# Note that it might make your system slower.

# %% id="Mhv5LZ3HbNCK"
trainer = Trainer(gpus=1, deterministic=True)

trainer.fit(model, train_loader, val_loader)

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
trainer = Trainer(track_grad_norm=2)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="3vHKxmruk62f"
# May be set to ‘inf’ infinity-norm.

# %% id="g7TbD6SxlAjP"
trainer = Trainer(track_grad_norm='inf')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="TcMlRe7ywpe6"
# ## Gradient clipping
#
#
# Exploding gradients refer to the problem that the gradients get too large and overflow in training, making the model unstable. Gradient clipping will ‘clip’ the gradients or cap them to a Threshold value to prevent the gradients from getting too large. To avoid this, we can set `gradient_clip_val` (default is set to 0.0).
#
# [when to use it, what are relevant values]

# %% id="jF9JwmbOgOWF"
trainer = Trainer(gradient_clip_val=0.1)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="ggb4MkkQrr1h"
# # truncated_bptt_steps
#
#

# %% [markdown] id="s1Iu6PyAw9_r"
# If you have a large recurrent model, you can use truncated_bptt_steps flag to split up the backprop over portions of the sequence. This flag will automatically truncate your batches and the trainer will apply Truncated Backprop to it.
#
# Make sure your batches have a sequence dimension.
#
# Lightning takes care of splitting your batch along the time-dimension.
# ```
# # we use the second as the time dimension
# # (batch, time, ...)
# sub_batch = batch[0, 0:t, ...]
# Using this feature requires updating your LightningModule’s pytorch_lightning.core.LightningModule.training_step() to include a hiddens arg with the hidden
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
trainer = Trainer(truncated_bptt_steps=5)

trainer.fit(model, train_loader, val_loader)

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
trainer = Trainer(reload_dataloaders_every_epoch=True)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="f513EYl0bmmL"
# # Callbacks
#

# %% [markdown] id="2pt7iGh4xNs5"
#
# Lightning Callbacks are self-contained programs that can be reused across projects.
# Callbacks should capture NON-ESSENTIAL logic that is NOT required for your LightningModule to run. Lightning includes some a few built-in callbacks that can be used with flags like early stopping and Model Checkpointing, but you can also create your own callbacks to add any functionality to your models.
#
# The callback API includes hooks that allow you to add logic at every point of your training:
# setup, teardown, on_epoch_start, on_epoch_end, on_batch_start, on_batch_end, on_init_start, on_keyboard_interrupt etc.
#
#

# %% [markdown] id="1t84gvDNsUuh"
# ## callbacks
#
# Use **callbacks=** to pass a list of user defined callbacks. These callbacks DO NOT replace the built-in callbacks (loggers or EarlyStopping).
#
# In this example, we create a dummy callback that prints a message when training starts and ends, using on_train_start and on_train_end hooks.

# %% id="oIXZYabub3f0"
from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


# a list of callbacks
callbacks = [PrintCallback()]
trainer = Trainer(callbacks=callbacks)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="cNF74CLYfJJu"
# # Model checkpointing
#
#

# %% [markdown] id="2blgquBrxLtS"
# Checkpoints capture the exact value of all parameters used by a model.
#
# Checkpointing your training allows you to resume a training process in case it was interrupted, fine-tune a model or use a pre-trained model for inference without having to retrain the model.
#
# Lightning automates saving and loading checkpoints so you restore a training session, saving all the required parameters including:
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
# By default Lightning will save a checkpoint in the end of the first epoch in the working directory, which will be updated every epoch.

# %% id="XGu0JULrg9l7"
# default used by the Trainer
trainer = Trainer(default_root_dir=os.getcwd())

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="3s9OjkGuhq1W"
# To change the checkpoint path pass in **default_root_dir=**

# %% id="DgdxkrIQhvfw"
trainer = Trainer(default_root_dir='/your/path/to/save/checkpoints')

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Qyvj_bkWrJiE"
#
# You can also have Lightning update your checkpoint based on a specific metric that you are logging (using self.log), by passing the key to `monitor=`. For example, if we want to save checkpoint based on the validation loss, logged as `val_loss`, you can pass:
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

trainer = Trainer(callbacks=[ModelCheckpoint(monitor='val_loss')])

trainer.fit(model, train_loader, val_loader)

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

trainer = Trainer(callbacks=[checkpoint_callback])

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="YKhZ6xRojJcl"
# You can disable checkpointing it by passing
#
#

# %% id="Yt8zd2ZFjOXX"
trainer = Trainer(checkpoint_callback=False)

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
#

# %% id="BpAFfg5zkFmH"
model = LitAutoEncoder.load_from_checkpoint(PATH)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model.eval()
y_hat = model(x)

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
model = LitAutoEncoder.load_from_checkpoint(PATH)

# %% id="14WwGpnVk0a4"
# uses in_dim=128, out_dim=10
model = LitAutoEncoder.load_from_checkpoint(PATH, in_dim=128, out_dim=10)

# %% [markdown] id="bY5s6wP_k1CU"
#
#
# ## Restoring Training State (resume_from_checkpoint)
# If your training was cut short for some reason, you can resume exactly from where you left off using the `resume_from_checkpoint` flag, which will automatically restore model, epoch, step, LR schedulers, apex, etc...

# %% id="9zfhHtyrk3rO"
model = LitAutoEncoder()
trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model)

# %% [markdown] id="xkKdvALFsmT2"
# ## weights_save_path
# You can specify a directory for saving weights file using `weights_save_path`.
#
# (If you are using a custom checkpoint callback, the checkpoint callback will override this flag).

# %% id="9OwHHFcCsrgT"
# save to your custom path
trainer = Trainer(weights_save_path='my/path')

trainer.fit(model, train_loader, val_loader)

# %% id="PbNtlJ9Wsscf"
# if checkpoint callback used, then overrides the weights path
# **NOTE: this saves weights to some/path NOT my/path
checkpoint = ModelCheckpoint(filepath='some/path')
trainer = Trainer(callbacks=[checkpoint], weights_save_path='my/path')
trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="uDdxCuyHdWQt"
# # Early stopping
#

# %% [markdown] id="fqAy3ihRxTfR"
# The EarlyStopping callback can be used to monitor a validation metric and stop the training when no improvement is observed, to help you avoid overfitting.
#
# To enable Early Stopping you can init the EarlyStopping callback, and pass it to `callbacks=` trainer flag. The callback will look for a logged metric to early stop on.
#
#

# %% id="lFx976CheH93"
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

trainer = Trainer(callbacks=[EarlyStopping('val_loss')])

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="MwpJfTvjeOwF"
# You can customize the callback using the following params:
#

# %% id="V6I9h6HteK2U"
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.00, patience=3, verbose=False, mode='max')
trainer = Trainer(callbacks=[early_stop_callback])

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="7TAIerPYe_Q1"
# The EarlyStopping callback runs at the end of every validation check, which, under the default configuration, happens after every training epoch. However, the frequency of validation can be modified by setting various parameters on the Trainer, for example check_val_every_n_epoch and val_check_interval. It must be noted that the patience parameter counts the number of validation checks with no improvement, and not the number of training epochs. Therefore, with parameters check_val_every_n_epoch=10 and patience=3, the trainer will perform at least 40 training epochs before being stopped.

# %% [markdown] id="VoKrX2ENh9Fg"
# # Logging

# %% [markdown] id="-CQTPKd7iKLm"
# Lightning has built in integration with various loggers such as TensorBoard, wandb, commet, etc.
#
#
# You can pass any metrics you want to log during training to `self.log`, such as loss or accuracy. Similarly, pass in to self.log any metric you want to log during validation step.
#
# These values will be passed in to the logger of your choise. simply pass in any supported logger to logger trainer flag.
#
#
#
# Use the as`logger=` trainer flag to pass in a Logger, or iterable collection of Loggers, for experiment tracking.
#
#
#
#

# %% id="ty5VPS3AiS8L"
from pytorch_lightning.loggers import TensorBoardLogger

# default logger used by trainer
logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name='lightning_logs')
trainer = Trainer(logger=logger)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="jc5oWNpoiuuc"
# Lightning supports the use of multiple loggers, just pass a list to the Trainer.
#
#

# %% id="BlYwMRRyivp_"
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger

logger1 = TensorBoardLogger('tb_logs', name='my_model')
logger2 = TestTubeLogger('tb_logs', name='my_model')
trainer = Trainer(logger=[logger1, logger2])

# %% [markdown] id="a7EyspQPh7iQ"
# ## flush_logs_every_n_steps
#
# Use this flag to determine when logging to disc should happen.

# %% id="Em_XvsmyiBbk"
trainer = Trainer(flush_logs_every_n_steps=100)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="_vDeKE98qsl1"
# ## log_every_n_steps
# How often to add logging rows (does not write to disk)
#
#

# %% id="HkqD7D_0w1Tt"
trainer = Trainer(log_every_n_steps=1000)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="9uw0gfe422CT"
# # info logging

# %% [markdown] id="dQXpt0aatDGo"
# ### default_root_dir
#
# ---
#
#
#
# Default path for logs and weights when no logger or pytorch_lightning.callbacks.ModelCheckpoint callback passed. On certain clusters you might want to separate where logs and checkpoints are stored. If you don’t then use this argument for convenience. Paths can be local paths or remote paths such as s3://bucket/path or ‘hdfs://path/’. Credentials will need to be set up to use remote filepaths.

# %% [markdown] id="CMmID2Bts5W3"
# ## weights_summary
# Prints a summary of the weights when training begins. Default is set to `top`- print summary of top level modules.
#
# Options: ‘full’, ‘top’, None.

# %% id="KTl6EdwDs6j2"

# print full summary of all modules and submodules
trainer = Trainer(weights_summary='full')

trainer.fit(model, train_loader, val_loader)

# %% id="R57cSLl9w9ma"
# don't print a summary
trainer = Trainer(weights_summary=None)

trainer.fit(model, train_loader, val_loader)

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
trainer = Trainer(process_position=0)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="itivQFgEphBU"
# ## progress_bar_refresh_rate
#
# How often to refresh the progress bar (in steps). In notebooks, faster refresh rates (lower number) is known to crash them because of their screen refresh rates, so raise it to 50 or more.

# %% id="GKe6eVxmplL5"
# default used by the Trainer
trainer = Trainer(progress_bar_refresh_rate=1)

trainer.fit(model, train_loader, val_loader)

# %% id="8rDHJOJbxNtf"
# disable progress bar
trainer = Trainer(progress_bar_refresh_rate=0)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="NCNvYLwjpWne"
# # profiler

# %% id="pRknrG_zpY6M"
# to profile standard training events
trainer = Trainer(profiler=True)

trainer.fit(model, train_loader, val_loader)

# %% [markdown] id="Ji6aWpU73kMM"
# You can also use Lightning AdvancedProfiler if you want more detailed information about time spent in each function call recorded during a given action. The output is quite verbose and you should only use this if you want very detailed reports.
#
#

# %% id="layG55pt316C"
from pytorch_lightning.profiler import AdvancedProfiler

trainer = Trainer(profiler=AdvancedProfiler())

trainer.fit(model, train_loader, val_loader)
