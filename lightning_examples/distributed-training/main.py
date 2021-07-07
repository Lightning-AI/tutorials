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

# %%
import prepare_notebook

# %% [markdown]
# # Distributed Training with PyTorch Lightning
#
# This tutorial covers several accelerator choices for multi-GPU distributed training and will walk you through the differences between these, when to use one over the other, and best practices in writing accelerator-agnostic code.
#
#
# ## Table of Contents
#
#
# 1. [Multi-GPU Acceleration in Lightning](#multi-gpu)
# 2. [Prerequisites](#prerequisites)
#
# 3. [DDP: Distributed Data-Parallel](#ddp)
# 4. [DDP-spawn: Distributed Data-Parallel Spawn](#ddp_spawn)
# 5. [DP: Data-Parallel](#dp)
# 6. [SDP: Sharded Data-Parallel](#sdp)
# 7. [FSDP: Fully Sharded Data-Parallel](#fsdp)
# 8. [Distributed Inference](#inference)
# 9. [Best practices](#best-practice)

# %% [markdown]
# <a id='multi-gpu'></a>
#
# ## Multi-GPU Acceleration in Lightning
#
# Lightning supports a variety of different accelerators and plugins for multi-GPU/distributed training. The **Accelerator** determines the hardware type we are running on. This can be a CPU, GPU, TPU or IPU. Part of the accelerator is also a **Plugin** (also referred to as "training type plugin", "backend" or "distributed backend" sometimes) that determines how model and data are split across multiple devices and it defines the communication and synchronization between devices and processes.
#
# This tutorial will focus on the **GPU accelerator** because it is compatible with a large selection of different plugins.
# Below we list all of the major choices that Lightning offers, each with recommendations when to use and when not to use, an example code, and important details to consider for writing device-agnostic and performant code.
#
#
#
#
#

# %% [markdown]
# <a id='prerequisites'></a>
#
# ## Prerequisites
#
# In order to run multi-GPU experiments, you will need
#
# - A server or desktop machine with GPU devices
# - PyTorch installed with GPU support
#
#
# Throughout the next sections, we will re-use the following templates for the model and data module.
#
# **IMPORTANT NOTE:** This notebook is not meant to be executed in full. Some cells will produce an output but most of the backends presented here will NOT run in a Jupyter environment.

# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = "./", batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # only downloads the data once
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
    
    
class TutorialModule(LightningModule):

    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)
        self.val_accuracy = Accuracy(num_classes=10)
        self.test_accuracy = Accuracy(num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, "y_hat": y_hat.detach()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        self.log("val_acc", self.val_accuracy(pred, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        self.log("test_acc", self.test_accuracy(pred, y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# %% [markdown]
# We perform a quick test run to check that the template code works. It should achieve a test accuracy of ~92% after one epoch of training.

# %%
seed_everything(1)

model = TutorialModule()
datamodule = MNISTDataModule()
trainer = Trainer(max_epochs=1)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

# %% [markdown]
# <a id='ddp'></a>
#
# ## DDP: Distributed Data-Parallel
#
# **Use when:** 
# - you want to scale your training to as many GPUs as you want;
# - you want to perform multi-node training.
#
# **Do not use when:**
# - you are running inside a Jupyter noteboook.
#
# **How to activate:**

# %%
trainer = Trainer(
    gpus=2, 
    accelerator="ddp",
)


# %% [markdown]
# The Distributed Data-Parallel (DDP) plugin in Lightning is orchestrating training in several processes. There are as many processes as devices are involved and these processes can either be launched directly by Lightning (default) or an external launch utilitiy like ``torch.distributed.launch``. The DDP plugin is the recommended choice by Lightning because it scales linearly (with a constant overhead) in the number of GPUs and has very few limitations for the average use case.
#
# **IMPORTANT:** DDP only works in script-mode, i.e., you need to be able to launch your program like so:
# ```bash
# python train.py [ARGS]
# ```
# It will NOT work in Jupyter notebooks, Google Colab, Kaggle, etc.
#
# Under the hood, Lightning calls the script (itself) several times to launch more processes, like so:
#
# ```bash
# # this is what the user launches
# python train.py --gpus 4
#
# # lightning launches the same program an additional 3 times:
# LOCAL_RANK=1 python train.py --gpus 4
# LOCAL_RANK=2 python train.py --gpus 4
# LOCAL_RANK=3 python train.py --gpus 4
# ```
#
# The local rank is what uniquely identifies each process. These processes will run independently in parallel and synchronize at certain points of their execution (more about that later). There are two important aspects crucial to the understanding of DDP; the model and the data.
#
# **Data:** The data gets partinioned into N subsets where N is the number of GPUs/processes. Each process only has access to its assigned subset and this is why the plugin is called data-parallel. The splitting of the data into each process is automatically taken care of by Lightning and the PyTorch distributed sampler.
#
# **Model:** The model, at any point in time, has the same parameter values across all GPUs. The difference between the processes are the gradients, because they get computed from data that is different in each process. Before a model is updated, the gradients are synchronized (averaged) so that after the optimizer update the model weights are all the same again across the processes.

# %% [markdown]
# **Example 1:** No code changes are required to run with DDP. Simply set the Trainer argument for the accelerator:

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# model = TutorialModule()
# datamodule = MNISTDataModule()
#
# trainer = Trainer(
#     gpus=4,
#     accelerator="ddp",
# )
# trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# **Example 2:** If there is data to download, extract and preprocess before training, it is important to do this only once and only in one process. Otherwise, each process will write to the same files at the same time. In Lightning, we split this logic into two separate hooks: ``prepare_data()`` and ``setup()``. 

# %%
class PreprocessExampleDataModule(MNISTDataModule):

    def prepare_data(self):
        # runs only once and only in the process 0
        # this hook is also available in the LightningModule
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # runs in each process
        # this hook is also available in the LightningModule
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)


# %% [markdown]
# **Example 3**: With DDP it is also possible to run multi-node training.

# %%
model = TutorialModule()
datamodule = MNISTDataModule()

trainer = Trainer(
    gpus=4,
    num_nodes=2,
    accelerator="ddp",
    # set to False if you are preparing data on a shared filesystem
    # default is True
    prepare_data_per_node=False,
)


# %% [markdown]
# Notice the ``prepare_data_per_node`` Trainer argument. The setting of this boolean depends on what we are doing in the ``prepare_data`` hook: In our example here we are downloading data, and we don't want to do that on every node if the filesystem is a shared across the servers.
#
# In the normal case, no other changes to the script are required to make multi-node training possible. However, the way the script gets launched depends on your cluster. Instructions how do so can be found in the [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html).

# %% [markdown]
# **Example 4:** As mentioned before, the dataset gets partitioned and evenly distributed across the processes. This is possible with a [distributed sampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler) which Lightning automatically adds for us. There are two important caveats we need to be aware of!
#
# 1. If the dataset is not evenly divisible by the number of GPUs, then the distributed sampler will append enough "fake" samples such that all GPUs see the same number of samples. These samples are fake in the sense that they are copies of existing data and thus data distribution is slightly biased. It is necessary due to the way PyTorch synchronizes the processes and cannot be avoided.
#
# 2. The `training_epoch_end`, `validation_epoch_end` and `test_epoch_end` hooks will receive ONLY the outputs of the step method in the respective process, NOT all outputs from all processes/GPUs. 

# %%
class DDPDataDemoModule(TutorialModule):

    def training_epoch_end(self, outputs):
        process_id = self.global_rank
        print(f"{process_id=} saw {len(outputs)} samples total")

    def on_train_end(self):
        print(f"training set contains {len(self.trainer.datamodule.mnist_train)} samples")



# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
# model = DDPDataDemoModule()
# datamodule = MNISTDataModule(batch_size=1)
#
# trainer = Trainer(
#     gpus=4,
#     accelerator="ddp",
#     max_epochs=1,
# )
# trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# In the terminal output we see the following:
#     
# ```
# process_id=0 saw 13750 samples total
# process_id=1 saw 13750 samples total
# process_id=2 saw 13750 samples total
# process_id=3 saw 13750 samples total
#
# training set contains 55000 samples
# ```
# As we can see, each GPU gets 55000 / 4 = 137500 data samples. However, what happens if we set the number of GPUs to 3, which does not divide 55000?

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# trainer = Trainer(
#     # NOTICE: does not divide dataset size evenly
#     gpus=3,
#     accelerator="ddp",
#     max_epochs=1,
# )
# trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# ```
# process_id=0 saw 18334 samples total
# process_id=1 saw 18334 samples total
# process_id=2 saw 18334 samples total
# training set contains 55000 samples
# ```
#
# Notice that we saw 18334 samples in each process, and 18334 * 3 = 55002, while 55000 % 3 = 1. This means the distributed sampler produced **one extra sample** in each process.

# %% [markdown]
# **Example 5:** As we have seen in the previous example, the ``*_epoch_end`` hooks collect only the outputs for the current process. What if we want all outputs? It can be achieved by calling the ``LightningModule.all_gather`` method:

# %%
class DDPAllGatherDemoModule(TutorialModule):

    def test_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred

    def test_epoch_end(self, outputs):
        process_id = self.global_rank
        preds = torch.cat(outputs)
        print(f"{process_id=} saw {len(outputs)} test_step outputs, made {len(preds)} predictions")

        # gather all predictions into all processes
        all_preds = self.all_gather(preds)
        print(f"{process_id=} all-gathered {all_preds.shape[0]} x {all_preds.shape[1]} predictions")

        # do something will all outputs
        # ...


# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# model = DDPAllGatherDemoModule()
# datamodule = MNISTDataModule()
#
# trainer = Trainer(
#     gpus=4,
#     accelerator="ddp",
#     max_epochs=1,
# )
# trainer.test(model, datamodule=datamodule)

# %% [markdown]
# The output is:
#
#
# ```
# process_id=0 saw 157 test_step outputs, made 2500 predictions
# process_id=1 saw 157 test_step outputs, made 2500 predictions
# process_id=2 saw 157 test_step outputs, made 2500 predictions
# process_id=3 saw 157 test_step outputs, made 2500 predictions
#
# process_id=0 all-gathered 4 x 2500 predictions
# process_id=1 all-gathered 4 x 2500 predictions
# process_id=2 all-gathered 4 x 2500 predictions
# process_id=3 all-gathered 4 x 2500 predictions
# ```

# %% [markdown]
# **Example 6:** Effective batch size, learning rate, number of workers. 
#
# The batching happens independently in each process, i.e., the ``batch_size`` argument set in the data loader is local to the current process. The effective batch size can be computed like so:

# %%
model = TutorialModule()
datamodule = MNISTDataModule()

trainer = Trainer(gpus=4, accelerator="ddp")

effective_batch_size = datamodule.batch_size * trainer.world_size

print(f"{trainer.world_size=}")
print(f"{datamodule.batch_size=}")
print(f"{effective_batch_size=}")

# %% [markdown]
# If, for example, we double the number of GPUs, the effective batch size automatically doubles too. By the rule of thumb, it is also advised to double the learning rate by that same factor. We can do that easily by defining a *base learning rate* for a single GPU and then multiply by the total GPUs, like so:

# %%
trainer = Trainer(gpus=4, accelerator="ddp")

base_learning_rate = 0.001
world_learning_rate = base_learning_rate * trainer.world_size

print(f"{trainer.world_size=}")
print(f"{base_learning_rate=}")
print(f"{world_learning_rate=}")

model = TutorialModule(learning_rate=world_learning_rate)

# %% [markdown]
# Also, the number of workers ``num_workers`` as well as all other settings for the dataloaders applies *per process*.
# Best practice is to tune the learning rate, batch size and num workers using a single GPU to a good initial value, and then scale up to many GPUs as shown above with no code changes required.

# %% [markdown]
# <a id='ddp_spawn'></a>
#
# ## DDP-spawn: Distributed Data-Parallel Spawn
#
#
# **Use when:** 
# - you want to get the benefits of DDP / distributed data-parallel, but
# - you want to run DDP inside a Jupyter notebook.
#
# **Do not use when:**
# - you need to run multi-node training;
# - you want the fastest multi-GPU training (e.g., use DDP instead);
# - one or several of your Python objects are not picklable.
#
# **How to activate:**

# %%
trainer = Trainer(
    gpus=4, 
    accelerator="ddp_spawn",  # this is already the default in Lightning!
)

# %% [markdown]
#
# DDP-spawn is a variation of [DDP](#ddp) and is the default accelerator when the ``gpus`` Trainer argument is used. The two behave identically when training a model, however, the spawn version launches the distributed processes differently, namely using the [``torch.multiprocessing.spawn``](https://pytorch.org/docs/stable/multiprocessing.html?highlight=spawn#torch.multiprocessing.spawn) function. A call to ``trainer.fit()`` (or test/validate/predict) with N GPUs will do the following:
#
# 1. The main forks N new processes in which the model will train. The main process will wait and DO NOTHING until all worker processes finish. This is different from [DDP](#ddp) where the main process launches N-1 subprocesses and then continues to participate for training.
#
# 2. When forking the processes, all objects in the main process get pickled and sent to the worker processes. This includes the initial weights of the model and any other objects defined by the user. These objects need to be picklable!
#
# 3. When the worker processes finish training, the execution continues in the main process where ``trainer.fit()`` (or test/validate/predict) ends. At the same time, the model weights get copied to the main process (which so far was waiting and never training anything). **IMPORTANT:** Only the model weights get copied back to the main process.
#
# 4. The worker processes die off and execution continues in the main process.
#
#
# This method of forking processes has several disadvantages.
#
# - The forking is expensive, especially when many dataloader workers (``num_workers``) are involved.
# - Every object needs to be picklable.
#
# In light of these limitations, [DDP](#ddp) offers a much better user experience overall. However, DDP-spawn and [DP](#dp) are the only accelerators that work in a Jupyter notebook (Google Colab, Kaggle, etc.). 
#
# **Examples:** All examples and instructions in the [DDP](#ddp) section apply for DDP-spawn as well! All you have to do is change ``accelerator="ddp"`` to ``accelerator="ddp_spawn"``.

# %% [markdown]
# <a id='dp'></a>
#
# ## DP: Data-Parallel
#
# **Use when:** 
# - you want to port an existing PyTorch model written with DataParallel and want to maintain 100% parity;
# - your optimization needs the full aggregated batch of outputs/losses from all GPUs;
# - you need to run multi-GPU in a Jupyter notebook cell, and cannot convert to a script;
# - none of the other backends presented here are suitable due to their hardware and runtime requirements.
#
# **Do not use when:**
# - you are looking for the most performant multi-GPU code;
# - you have custom batch structures that can not be converted to primitive containers like tuples, lists, dicts etc.;
# - you rely heavily on torchmetrics.
#
# **How to activate:**

# %%
# data-parallel with 2 GPUs
trainer = Trainer(
    gpus=2, 
    accelerator="dp",
)


# %% [markdown]
# Data-Parallel initially moves all model parameters, buffers and data tensors to the root GPU. In Lighting, this is GPU 0. The following steps take place in _every_ training step:
# 1. The model gets replicated to every device, i.e., parameters and buffers get copied from the root device to all other devices. 
# 2. The data batch that initially resides on GPU 0 gets split into N sub-batches along dimension 0 (batch dimension). Each GPU receives one of these batch splits and they are passed to the ``training_step`` hook.
# 3. The output of ``training_step`` in each device will be transferred back to the root device and averaged.
#
# The fact that the module is replicated every forward and backward pass makes this the least efficient plugin for multi-GPU training. An additional caveat is that state changes on the module during ``training_step`` are lost, and this is a common source of bugs. It is also the reason why torchmetrics is not recommended together with this plugin.

# %% [markdown]
# **Example 1:**

# %%
class DPModule(TutorialModule):
    
    # *_step() happens on the replica of the model (each GPU runs this)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # total batch size = 16, 2 GPUs -> each GPU sees batch of size 8
        # the last batch may still be smaller, the dataset may not be evenly divisible by the batch size 
        assert x.shape[0] <= 8 
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred, y

    def validation_step_end(self, outputs):
        # torchmetrics do not support sync on the replica
        # all torchmetric computations need to be performed in *_step_end()
        # which happens on the root device
        pred, y = outputs
        self.log("val_acc", self.val_accuracy(pred, y), prog_bar=True)


# %%
seed_everything(1)

model = DPModule()

datamodule = MNISTDataModule()
trainer = Trainer(
    gpus=2, 
    accelerator="dp", 
    max_epochs=1,
)

trainer.fit(model, datamodule=datamodule)


# %% [markdown]
# **Example 2: Custom reduction**

# %%
class DPModule(TutorialModule):
    
    def training_step_end(self, outputs):
        # outputs is a dict
        # it is the result of merging all dicts returned by training_step() on each device
        
        # the loss from each GPU, 2 GPUs are used here
        losses = outputs["loss"]
        assert losses.shape[0] == 2
        
        # each GPU returned 8 predictions
        y_hats = outputs["y_hat"]
        assert y_hats.shape[0] == 2 * 8
        
        probs = F.softmax(y_hats, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = torch.mean(losses)
        return {"loss": loss, "pred": preds}
    
    def training_epoch_end(self, outputs):
        # we can receive all outputs from all training steps and concatenate them
        all_predictions = torch.cat([out["pred"] for out in outputs])
        print(all_predictions)



# %%
model = DPModule()
datamodule = MNISTDataModule()

trainer = Trainer(
    gpus=2, 
    accelerator="dp", 
    max_steps=4,
    limit_val_batches=0,
)
trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# <a id='sdp'></a>
#
#
# ## SDP: Sharded Data-Parallel

# %% [markdown]
# **Use when:** 
# - memory is a concern because model parameters + optimizer + gradients do not fit on a GPU;
# - your model has >= 500 million parameters;
# - you are using very large batch sizes or inputs.
#
# **Do not use when:**
# - your model is small enough to fit on a single GPU.
#
# **How to activate:**

# %%
trainer = Trainer(
    gpus=4, 
    accelerator="ddp_sharded", 
)


# %% [markdown]
# Sharded Data Parallel (SDP) offers significant memory savings for very large models (above 500M parameters). It enables one to train models that would normally not fit onto a single GPU, or allows for an increased batch- or input size.
#
# The motivation behind sharded training comes from the observation that gradients and optimizer state are usually dominating the memory during training. This is especially significant for optimizers such as Adam where per-parameter weights and running averages are kept in memory.
#
# **Example**: Memory comparison (artificial example). Here we want to draw a simple comparison between DDP and Sharded DDP using the MNIST toy example. Note, the MNIST classifier is way too small to see practical benefits here, but for demonstration purposes it will suffice to show minor memory efficiency.
#
# First we measure the memory footprint when training with ``accelerator="ddp"``.

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
# model = TutorialModule()
# datamodule = MNISTDataModule()
#
# trainer = Trainer(
#     gpus=2,
#     accelerator="ddp",
#     max_epochs=1,
# )
# trainer.fit(model, datamodule=datamodule)
#
# ddp_max_mem = torch.cuda.max_memory_allocated(trainer.local_rank) / 1000
# print(f"GPU {trainer.local_rank} max memory using DDP: {ddp_max_mem:.2f} MB")

# %% [markdown]
# Output:
#
# ```
# GPU 0 max memory using DDP: 2859.52 MB
# GPU 1 max memory using DDP: 2859.52 MB
# ```
#
# Next, we switch to ``accelerator="ddp_sharded"``:

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
# model = TutorialModule()
# datamodule = MNISTDataModule()
#
# trainer = Trainer(
#     gpus=2,
#     accelerator="ddp_sharded",
#     max_epochs=1,
# )
# trainer.fit(model, datamodule=datamodule)
#
# sdp_max_mem = torch.cuda.max_memory_allocated(trainer.local_rank) / 1000
# print(f"GPU {trainer.local_rank} max memory using SDP: {sdp_max_mem:.2f} MB")

# %% [markdown]
# Output:
#
# ```
# GPU 0 max memory using SDP: 2433.02 MB
# GPU 1 max memory using SDP: 905.22 MB
# ```

# %% [markdown]
# <a id='fsdp'></a>
#
#
# ## FSDP: Fully Sharded Data-Parallel (COMING SOON)

# %%

# %% [markdown]
# <a id='inference'></a>
#
#
# ## Distributed Inference

# %% [markdown]
# There are mainly two ways Lightning offers distributed inference:
#
# 1. The model produces outputs in each process/GPU and at the end all predictions get *all-gathered* into each GPU. This requires that all predictions must fit into the memory of a GPU.
#
# 2. If approach 1 is not feasible due to memory limitations, Lightning offers a `BasePredictionWriter` Callback that can be extended with a save function. The callback will receive the predictions and save them to disk (or upload to a remote filesystem) with the user provided save function. Later on, the predictions can be loaded into CPU memory for further processing.
#
#

# %%
class DDPInferenceDataModule(MNISTDataModule):

    def setup(self, stage=None):
        super().setup(stage=stage)
        if stage == 'predict' or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)



# %%
class DDPInferenceModel(TutorialModule):

    def predict_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred



# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
#
# model = TutorialModule()
# datamodule = MNISTDataModule()
#
# trainer = Trainer(
#     gpus=2,
#     accelerator="ddp",
#     max_epochs=1,
# )
#
# trainer.fit(model, datamodule=datamodule)
#
# best_path = trainer.checkpoint_callback.best_model_path
# print(f"Best model: {best_path}")

# %% [markdown]
# **Example 1**: Saving predictions to GPU memory.

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
#
# model = DDPInferenceModel()
# datamodule = DDPInferenceDataModule(batch_size=1)
#
# trainer = Trainer(
#     gpus=2,
#     accelerator="ddp",
#     limit_predict_batches=4,
# )
#
# predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=best_path, return_predictions=True)
# all_predictions = torch.cat(predictions)
# all_predictions = trainer.accelerator.all_gather(all_predictions)
#
# if trainer.global_rank == 0:
#     print(f"All predictions gathered in GPU memory: {all_predictions}")

# %% [markdown]
# The output of this script is:
# ```
# All predictions gathered in GPU memory: tensor([[7, 9, 1, 0], [0, 3, 9, 1]], device='cuda:0')
# ```
#
# There are four outputs per GPU because we set `Trainer(limit_predict_batches=4)` for this example.

# %% [markdown]
# **Example 2:** Writing predictions to disk and loading them into CPU memory.

# %%
from pytorch_lightning.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions, batch_indices):
        predictions = torch.cat(predictions[0]).cpu()
        torch.save(predictions, os.path.join(self.output_dir, f"predictions-{trainer.global_rank}.pt"))

# %% magic_args="echo Skipping cell. Run this code in a script instead!" language="script"
#
# seed_everything(1)
#
# model = DDPInferenceModel()
# datamodule = DDPInferenceDataModule(batch_size=1)
#
# trainer = Trainer(
#     gpus=2,
#     accelerator="ddp",
#     limit_predict_batches=4,
#     callbacks=[CustomWriter("./predictions/", write_interval="epoch")]
# )
# trainer.predict(model, datamodule=datamodule, ckpt_path=best_path, return_predictions=False)
#
# if trainer.global_rank == 0:
#     all_predictions = []
#     for i in range(trainer.world_size):
#         all_predictions.append(torch.load(f"./predictions/predictions-{i}.pt"))
#
#     all_predictions = torch.cat(all_predictions)
#     print(f"All predictions gathered from disk: {all_predictions}")

# %% [markdown]
# The output of this script is:
#
# ```
# All predictions gathered from disk: tensor([7, 9, 1, 0, 0, 3, 9, 1])
# ```
#
# As we can see, there were four predictions made per GPU, i.e., in total we collect 8 predictions.

# %%
