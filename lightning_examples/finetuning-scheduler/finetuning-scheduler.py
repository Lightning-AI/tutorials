# %% [markdown]
# ## Scheduled Finetuning
#
# The [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) callback accelerates and enhances model experimentation with flexible finetuning schedules.
#
# Training with the [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) is simple and confers a host of benefits:
#
# - it dramatically increases finetuning flexibility
# - expedites and facilitates exploration of model tuning dynamics
# - enables marginal performance improvements of finetuned models
#
# <div style="display:inline" id="a1">
#
# Fundamentally, the [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) callback enables scheduled, multi-phase,
# finetuning of foundational models. Gradual unfreezing (i.e. thawing) can help maximize
# foundational model knowledge retention while allowing (typically upper layers of) the model to
# optimally adapt to new tasks during transfer learning [1, 2, 3](#f1)
#
#
# </div>
#
# [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) orchestrates the gradual unfreezing
# of models via a finetuning schedule that is either implicitly generated (the default) or explicitly provided by the user
# (more computationally efficient). Finetuning phase transitions are driven by
# [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping)
# criteria (a multi-phase extension of ``EarlyStopping``), user-specified epoch transitions or a composition of the two (the default mode).
# A [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) training session completes when the
# final phase of the schedule has its stopping criteria met. See
# the [early stopping documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html) for more details on that callback's configuration.
#
# ![FinetuningScheduler explicit loss animation](fts_explicit_loss_anim.gif)

# %% [markdown]
#
# ## Basic Usage
#
# <div id="basic_usage">
#
# If no finetuning schedule is provided by the user, [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) will generate a
# [default schedule](#The-Default-Finetuning-Schedule) and proceed to finetune according to the generated schedule,
# using default [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping) and [FTSCheckpoint](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSCheckpoint) callbacks with ``monitor=val_loss``.
#
# </div>
#
# ```python
# from pytorch_lightning import Trainer
# from finetuning_scheduler import FinetuningScheduler
# trainer = Trainer(callbacks=[FinetuningScheduler()])
# ```

# %% [markdown]
# ## The Default Finetuning Schedule
#
# Schedule definition is facilitated via the [gen_ft_schedule](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.SchedulingMixin.gen_ft_schedule) method which dumps a default finetuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
# desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will likely be less computationally efficient than a user-defined finetuning schedule but is useful for exploring a model's finetuning behavior and can serve as a good baseline for subsquent explicit schedule refinement.
# While the current version of [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) only supports single optimizer and (optional) lr_scheduler configurations, per-phase maximum learning rates can be set as demonstrated in the next section.

# %% [markdown]
# ## Specifying a Finetuning Schedule
#
# To specify a finetuning schedule, it's convenient to first generate the default schedule and then alter the thawed/unfrozen parameter groups associated with each finetuning phase as desired. Finetuning phases are zero-indexed and executed in ascending order.
#
# 1. First, generate the default schedule to ``Trainer.log_dir``. It will be named after your
#    ``LightningModule`` subclass with the suffix ``_ft_schedule.yaml``.
#
# ```python
#     from pytorch_lightning import Trainer
#     from finetuning_scheduler import FinetuningScheduler
#     trainer = Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])
# ```
#
# 2. Alter the schedule as desired.
#
# ![side_by_side_yaml](side_by_side_yaml.png)
#
# 3. Once the finetuning schedule has been altered as desired, pass it to
#    [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) to commence scheduled training:
#
# ```python
# from pytorch_lightning import Trainer
# from finetuning_scheduler import FinetuningScheduler
#
# trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])
# ```

# %% [markdown]
# ## Early-Stopping and Epoch-Driven Phase Transition Criteria
#
#
# By default, [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping) and epoch-driven
# transition criteria are composed. If a ``max_transition_epoch`` is specified for a given phase, the next finetuning phase will begin at that epoch unless [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping) criteria are met first.
# If [FinetuningScheduler.epoch_transitions_only](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler.params.epoch_transitions_only) is ``True``, [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping) will not be used
# and transitions will be exclusively epoch-driven.
#
#
# <div class="alert alert-info">
#
# **Tip:** Use of regex expressions can be convenient for specifying more complex schedules. Also, a per-phase base maximum lr can be specified:
#
# ![emphasized_yaml](emphasized_yaml.png)
#
# </div>
#
#
#
# The end-to-end example in this notebook ([Scheduled Finetuning For SuperGLUE](#superglue)) uses [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) in explicit mode to finetune a small foundational model on the [RTE](https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte) task of [SuperGLUE](https://super.gluebenchmark.com/).
# Please see the [official Finetuning Scheduler documentation](https://finetuning-scheduler.readthedocs.io/en/latest/index.html) if you are interested in a similar [CLI-based example](https://finetuning-scheduler.readthedocs.io/en/latest/index.html#scheduled-finetuning-superglue) using the LightningCLI.

# %% [markdown]
# ## Resuming Scheduled Finetuning Training Sessions
#
# Resumption of scheduled finetuning training is identical to the continuation of
# [other training sessions](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) with the caveat that the provided checkpoint must have been saved by a [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) session.
# [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) uses [FTSCheckpoint](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSCheckpoint) (an extension of ``ModelCheckpoint``) to maintain schedule state with special metadata.
#
#
# ```python
# from pytorch_lightning import Trainer
# from finetuning_scheduler import FinetuningScheduler
# trainer = Trainer(callbacks=[FinetuningScheduler()])
# trainer.fit(..., ckpt_path="some/path/to/my_checkpoint.ckpt")
# ```
#
# Training will resume at the depth/level of the provided checkpoint according the specified schedule. Schedules can be altered between training sessions but schedule compatibility is left to the user for maximal flexibility. If executing a user-defined schedule, typically the same schedule should be provided for the original and resumed training sessions.
#
# By default ([FinetuningScheduler.restore_best](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html?highlight=restore_best#finetuning_scheduler.fts.FinetuningScheduler.params.restore_best) is ``True``), [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) will attempt to restore the best available checkpoint before finetuning depth transitions.
#
# ```python
# trainer = Trainer(callbacks=[FinetuningScheduler(new_incarnation_mode=True)])
# trainer.fit(..., ckpt_path="some/path/to/my_kth_best_checkpoint.ckpt")
# ```
#
# To handle the edge case wherein one is resuming scheduled finetuning from a non-best checkpoint and the previous best checkpoints may not be accessible, setting [FinetuningScheduler.new_incarnation_mode](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html?highlight=new_Incarnation_mode#finetuning_scheduler.fts.FinetuningScheduler.params.new_incarnation_mode) to
# ``True`` as above will re-intialize the checkpoint state with a new best checkpoint at the resumption depth.

# %% [markdown]
# <div class="alert alert-warning">
#
# **Note:** Currently, [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) supports the following ``StrategyType``s:
#
# - ``DP``
# - ``DDP``
# - ``DDP_SPAWN``
# - ``DDP_SHARDED``
# - ``DDP_SHARDED_SPAWN``
#
# </div>

# %% [markdown]
# <div id="superglue"></div>
#
# ## Scheduled Finetuning For SuperGLUE
#
# The following example demonstrates the use of [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) to finetune a small foundational model on the [RTE](https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte) task of [SuperGLUE](https://super.gluebenchmark.com/). Iterative early-stopping will be applied according to a user-specified schedule.
#

# %%
import os
import warnings
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cli import _Registry
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging
from transformers.tokenization_utils_base import BatchEncoding

# %%
# a couple helper functions to prepare code to work with a user module registry
MOCK_REGISTRY = _Registry()


def mock_register_module(key: str, require_fqn: bool = False) -> List:
    if key.lower() == "finetuningscheduler":
        mod = import_module("finetuning_scheduler")
        MOCK_REGISTRY.register_classes(mod, pl.callbacks.Callback)
    else:
        raise MisconfigurationException(f"user module key '{key}' not found")
    registered_list = []
    # make registered class available by unqualified class name by default
    if not require_fqn:
        for n, c in MOCK_REGISTRY.items():
            globals()[f"{n}"] = c
        registered_list = ", ".join([n for n in MOCK_REGISTRY.names])
    else:
        registered_list = ", ".join([c.__module__ + "." + c.__name__ for c in MOCK_REGISTRY.classes])
    print(f"Imported and registered the following callbacks: {registered_list}")


# %%
# Load the `FinetuningScheduler` PyTorch Lightning extension module we want to use. This will import all necessary callbacks.
mock_register_module("finetuningscheduler")
# set notebook-level variables
AVAIL_GPUS = torch.cuda.device_count()
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"

transformers_logging.set_verbosity_error()
# ignore warnings related tokenizers_parallelism/DataLoader parallelism trade-off and
# expected logging behavior
for warnf in [".*does not have many workers*", ".*The number of training samples.*"]:
    warnings.filterwarnings("ignore", warnf)


# %%
class RteBoolqDataModule(pl.LightningDataModule):
    """A ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face datasets."""

    TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("question", "passage")}
    LOADER_COLUMNS = (
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    )

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = DEFAULT_TASK,
        max_seq_length: int = 128,
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        tokenizers_parallelism: bool = True,
        **dataloader_kwargs: Any,
    ):
        """Initialize this ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            model_name_or_path (str):
                Can be either:
                    - A string, the ``model id`` of a pretrained model hosted inside a model repo on huggingface.co.
                        Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                        a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a ``directory`` containing model weights saved using
                        :meth:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
            task_name (str, optional): Name of the SuperGLUE task to execute. This module supports 'rte' or 'boolq'.
                Defaults to DEFAULT_TASK which is 'rte'.
            max_seq_length (int, optional): Length to which we will pad sequences or truncate input. Defaults to 128.
            train_batch_size (int, optional): Training batch size. Defaults to 16.
            eval_batch_size (int, optional): Batch size to use for validation and testing splits. Defaults to 16.
            tokenizers_parallelism (bool, optional): Whether to use parallelism in the tokenizer. Defaults to True.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizers_parallelism = tokenizers_parallelism
        self.dataloader_kwargs = {
            "num_workers": dataloader_kwargs.get("num_workers", 0),
            "pin_memory": dataloader_kwargs.get("pin_memory", False),
        }
        self.text_fields = self.TASK_TEXT_FIELD_MAP[self.task_name]
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, local_files_only=False)

    def prepare_data(self):
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.task_name)

    def setup(self, stage):
        """Setup our dataset splits for training/validation/testing."""
        self.dataset = datasets.load_dataset("super_glue", self.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)

    def _convert_to_features(self, example_batch) -> BatchEncoding:
        """Convert raw text examples to a :class:`~transformers.tokenization_utils_base.BatchEncoding` container
        (derived from python dict) of features that includes helpful methods for translating between word/character
        space and token space.

        Args:
            example_batch ([type]): The set of examples to convert to token space.

        Returns:
            ``BatchEncoding``: A batch of encoded examples (note default tokenizer batch_size=1000)
        """
        text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            text_pairs, max_length=self.max_seq_length, padding="longest", truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        return features


# %%
class RteBoolqModule(pl.LightningModule):
    """A ``LightningModule`` that can be used to finetune a foundational model on either the RTE or BoolQ SuperGLUE
    tasks using Hugging Face implementations of a given model and the `SuperGLUE Hugging Face dataset."""

    def __init__(
        self,
        model_name_or_path: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
        model_cfg: Optional[Dict[str, Any]] = None,
        task_name: str = DEFAULT_TASK,
        experiment_tag: str = "default",
    ):
        """
        Args:
            model_name_or_path (str): Path to pretrained model or identifier from https://huggingface.co/models
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
        """
        super().__init__()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        if task_name in TASK_NUM_LABELS.keys():
            self.task_name = task_name
        else:
            self.task_name = DEFAULT_TASK
            rank_zero_warn(f"Invalid task_name '{task_name}'. Proceeding with the default task: '{DEFAULT_TASK}'")
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}"
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.init_hparams = {
            "optimizer": self.optimizer_init,
            "lr_scheduler": self.lr_scheduler_init,
            "model_config": self.model.config,
            "model_name_or_path": model_name_or_path,
            "task_name": self.task_name,
            "experiment_id": self.experiment_id,
        }
        self.save_hyperparameters(self.init_hparams)
        self.metric = datasets.load_metric("super_glue", self.task_name, experiment_id=self.experiment_id)
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.finetuningscheduler_callback = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.finetuningscheduler_callback:
            self.log("finetuning_schedule_depth", float(self.finetuningscheduler_callback.curr_depth))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metric_dict, prog_bar=True)

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.optimizer_init["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        # the phase 0 parameters will have been set to require gradients during setup
        # you can initialize the optimizer with a simple requires.grad filter as is often done,
        # but in this case we pass a list of parameter groups to ensure weight_decay is
        # not applied to the bias parameter (for completeness, in this case it won't make much
        # performance difference)
        optimizer = AdamW(params=self._init_param_groups(), **self.optimizer_init)
        scheduler = {"scheduler": CosineAnnealingWarmRestarts(optimizer, **self.lr_scheduler_init)}
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        found_fts = [c for c in self.trainer.callbacks if isinstance(c, FinetuningScheduler)]  # type: ignore # noqa
        if found_fts:
            self.finetuningscheduler_callback = found_fts[0]
        return super().configure_callbacks()


# %% [markdown]
# ### Our Training Sessions
#
# We'll be comparing three different finetuning training configurations. Every configuration in this example depends
# upon a shared set of defaults, only differing in their respective finetuning schedules.
#
# | Experiment Tag    | Training Scenario Description                                          |
# |:-----------------:| ---------------------------------------------------------------------- |
# | ``fts_explicit``  | Training with a finetuning schedule explicitly provided by the user    |
# | ``nofts_baseline``| A baseline finetuning training session (without scheduled finetuning)  |
# | ``fts_implicit``  | Training with an implicitly generated finetuning schedule (the default)|
#
# Let's begin by configuring the ``fts_explicit`` scenario. We'll subsequently run the other two scenarios for
# comparison.

# %%
# Let's create a finetuning schedule for our model and run an explicitly scheduled finetuning training scenario with it
# Please see the [FinetuningScheduler documentation](https://finetuning-scheduler.readthedocs.io/en/latest/index.html) for a full description of the schedule format


ft_schedule_yaml = """
0:
  params:
  - model.classifier.bias
  - model.classifier.weight
  - model.pooler.dense.bias
  - model.pooler.dense.weight
  - model.deberta.encoder.LayerNorm.bias
  - model.deberta.encoder.LayerNorm.weight
  - model.deberta.encoder.rel_embeddings.weight
  - model.deberta.encoder.layer.{0,11}.(output|attention|intermediate).*
1:
  params:
  - model.deberta.embeddings.LayerNorm.bias
  - model.deberta.embeddings.LayerNorm.weight
2:
  params:
  - model.deberta.embeddings.word_embeddings.weight
"""
ft_schedule_name = "RteBoolqModule_ft_schedule_deberta_base.yaml"
# Let's write the schedule to a file so we can simulate loading an explicitly defined finetuning
# schedule.
with open(ft_schedule_name, "w") as f:
    f.write(ft_schedule_yaml)

# %%
datasets.set_progress_bar_enabled(False)
pl.seed_everything(42)
dm = RteBoolqDataModule(model_name_or_path="microsoft/deberta-v3-base", tokenizers_parallelism=True)

# %% [markdown]
# ### Optimizer Configuration
#
# <div id="a2">
#
# Though other optimizers can arguably yield some marginal advantage contingent on the context,
# the Adam optimizer (and the [AdamW version](https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW) which
# implements decoupled weight decay) remains robust to hyperparameter choices and is commonly used for finetuning
# foundational language models.  See [(Sivaprasad et al., 2020)](#f2) and [(Mosbach, Andriushchenko & Klakow, 2020)](#f3) for theoretical and systematic empirical justifications of Adam and its use in finetuning
# large transformer-based language models. The values used here have some justification
# in the referenced literature but have been largely empirically determined and while a good
# starting point could be could be further tuned.
#
# </div>

# %%
optimizer_init = {"weight_decay": 1e-05, "eps": 1e-07, "lr": 1e-05}

# %% [markdown]
# ### LR Scheduler Configuration
#
# <div id="a3">
#
# The [CosineAnnealingWarmRestarts scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html?highlight=cosineannealingwarm#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) nicely fits with our iterative finetuning since it does not depend upon a global max_epoch
# value. The importance of initial warmup is reduced due to the innate warmup effect of Adam bias correction [[5]](#f3)
# and the gradual thawing we are performing. Note that commonly used LR schedulers that depend on providing
# max_iterations/epochs (e.g. the
# [CosineWarmupScheduler](https://github.com/PyTorchLightning/lightning-tutorials/blob/0c325829101d5a6ebf32ed99bbf5b09badf04a59/course_UvA-DL/05-transformers-and-MH-attention/Transformers_MHAttention.py#L688)
# used in other pytorch-lightning tutorials) also work with FinetuningScheduler. Though the LR scheduler is theoretically
# justified [(Loshchilov & Hutter, 2016)](#f4), the particular values provided here are primarily empircally driven.
#
# </div>


# %%
lr_scheduler_init = {"T_0": 1, "T_mult": 2, "eta_min": 1e-07}

# %%
# Load our lightning module...
lightning_module_kwargs = {
    "model_name_or_path": "microsoft/deberta-v3-base",
    "optimizer_init": optimizer_init,
    "lr_scheduler_init": lr_scheduler_init,
}
model = RteBoolqModule(**lightning_module_kwargs, experiment_tag="fts_explicit")

# %% [markdown]
# ### Callback Configuration
#
# The only callback required to invoke the [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) is the [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) callback itself.
# Default versions of [FTSCheckpoint](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSCheckpoint) and [FTSEarlyStopping](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts_supporters.html#finetuning_scheduler.fts_supporters.FTSEarlyStopping)
# (if not specifying ``epoch_only_transitions``) will be included ([as discussed above](#basic_usage)) if not provided
# in the callbacks list. For demonstration purposes I'm including example configurations of all three callbacks below.

# %%
# let's save our callback configurations for the explicit scenario since we'll be reusing the same
# configurations for the implicit and nofts_baseline scenarios (except the  config for the
# FinetuningScheduler callback itself of course in the case of nofts_baseline)
earlystopping_kwargs = {"monitor": "val_loss", "min_delta": 0.001, "patience": 2}
checkpoint_kwargs = {"monitor": "val_loss", "save_top_k": 1}
fts_kwargs = {"max_depth": 1}
callbacks = [
    FinetuningScheduler(ft_schedule=ft_schedule_name, **fts_kwargs),  # type: ignore # noqa
    FTSEarlyStopping(**earlystopping_kwargs),  # type: ignore # noqa
    FTSCheckpoint(**checkpoint_kwargs),  # type: ignore # noqa
]

# %%
logger = TensorBoardLogger("lightning_logs", name="fts_explicit")
# optionally start tensorboard and monitor progress graphically while viewing multi-phase finetuning specific training
# logs in the cell output below by uncommenting the next 2 lines
# # %load_ext tensorboard
# # %tensorboard --logdir lightning_logs
# disable progress bar by default to focus on multi-phase training logs. Set to True to re-enable if desired
enable_progress_bar = False

# %%


def train() -> None:
    trainer = pl.Trainer(
        enable_progress_bar=enable_progress_bar,
        max_epochs=100,
        precision=16,
        strategy="dp",
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)


if AVAIL_GPUS > 0:
    train()
else:
    print("Given the multiple phases of finetuning demonstrated, this notebook is best used with a GPU")

# %% [markdown]
# ### Running the Baseline and Implicit Finetuning Scenarios
#
# Let's now compare our ``nofts_baseline`` and ``fts_implicit`` scenarios with the ``fts_explicit`` one we just ran.
#
# We'll need to update our callbacks list, using the core PL ``EarlyStopping`` and ``ModelCheckpoint`` callbacks for the
# ``nofts_baseline`` (which operate identically to their FTS analogs apart from the recursive training support).
# For both core PyTorch Lightning and user-registered callbacks, we can define our callbacks using a dictionary as we do
# with the LightningCLI. This allows us to avoid managing imports and support more complex configuration separated from
# code.
#
# Note that we'll be using identical callback configurations to the ``fts_explicit`` scenario. Keeping [max_depth](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html?highlight=max_depth#finetuning_scheduler.fts.FinetuningScheduler.params.max_depth) for
# the implicit schedule will limit finetuning to just the last 4 parameters of the model, which is only a small fraction
# of the parameters you'd want to tune for maximum performance. Since the implicit schedule is quite computationally
# intensive and most useful for exploring model behavior, leaving [max_depth](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html?highlight=max_depth#finetuning_scheduler.fts.FinetuningScheduler.params.max_depth) 1 allows us to demo implicit mode
# behavior while keeping the computational cost and runtime of this notebook reasonable. To review how a full implicit
# mode run compares to the ``nofts_baseline`` and ``fts_explicit`` scenarios, please see the the following
# [tensorboard experiment summary](https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/).


# %%
nofts_callbacks = [EarlyStopping(**earlystopping_kwargs), ModelCheckpoint(**checkpoint_kwargs)]
fts_implicit_callbacks = [
    FinetuningScheduler(**fts_kwargs),  # type: ignore # noqa
    FTSEarlyStopping(**earlystopping_kwargs),  # type: ignore # noqa
    FTSCheckpoint(**checkpoint_kwargs),  # type: ignore # noqa
]
scenario_callbacks = {"nofts_baseline": nofts_callbacks, "fts_implicit": fts_implicit_callbacks}

# %%
if AVAIL_GPUS > 0:
    for scenario_name, scenario_callbacks in scenario_callbacks.items():
        model = RteBoolqModule(**lightning_module_kwargs, experiment_tag=scenario_name)
        logger = TensorBoardLogger("lightning_logs", name=scenario_name)
        callbacks = scenario_callbacks
        print(f"Beginning training the '{scenario_name}' scenario")
        train()
else:
    print("Given the multiple phases of finetuning demonstrated, this notebook is best used with a GPU")

# %% [markdown]
# ### Reviewing the Training Results
#
# See the [tensorboard experiment summaries](https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/) to get a sense
# of the relative computational and performance tradeoffs associated with these [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) configurations.
# The summary compares a full ``fts_implicit`` execution to ``fts_explicit`` and ``nofts_baseline`` scenarios using DDP
# training with 2 GPUs. The full logs/schedules for all three scenarios are available
# [here](https://drive.google.com/file/d/1LrUcisRLHeJgh_BDOOD_GUBPp5iHAkoR/view?usp=sharing) and the checkpoints
# produced in the scenarios [here](https://drive.google.com/file/d/1t7myBgcqcZ9ax_IT9QVk-vFH_l_o5UXB/view?usp=sharing)
# (caution, ~3.5GB).
#
# [![fts_explicit_accuracy](fts_explicit_accuracy.png)](https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOnRydWUsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZSwiZnRzX2ltcGxpY2l0IjpmYWxzZX0%3D)
# [![nofts_baseline](nofts_baseline_accuracy.png)](https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOmZhbHNlLCJub2Z0c19iYXNlbGluZSI6dHJ1ZSwiZnRzX2ltcGxpY2l0IjpmYWxzZX0%3D)
#
# Note there could be around ~1% variation in performance from the tensorboard summaries generated by this notebook
# which uses DP and 1 GPU.
#
# [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) expands the space of possible finetuning schedules and the composition of more sophisticated schedules can
# yield marginal finetuning performance gains. That stated, it should be emphasized the primary utility of [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) is to grant
# greater finetuning flexibility for model exploration in research. For example, glancing at DeBERTa-v3's implicit training
# run, a critical tuning transition point is immediately apparent:
#
# [![implicit_training_transition](implicit_training_transition.png)](https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOmZhbHNlLCJub2Z0c19iYXNlbGluZSI6ZmFsc2UsImZ0c19pbXBsaWNpdCI6dHJ1ZX0%3D)
#
# Our `val_loss` begins a precipitous decline at step 3119 which corresponds to phase 17 in the schedule. Referring to our
# schedule, in phase 17 we're beginning tuning the attention parameters of our 10th encoder layer (of 11). Interesting!
# Though beyond the scope of this tutorial, it might be worth investigating these dynamics further and
# [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) allows one to do just that quite easily.
#
# %% [markdown]
#
# Note that though this example is intended to capture a common usage scenario, substantial variation is expected
# among use cases and models.
# In summary, [FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) provides increased finetuning flexibility that can be useful in a variety of
# contexts from exploring model tuning behavior to maximizing performance.
# %% [markdown]
# ## Footnotes
#
# <ol>
# <li id="f1">
#
# [Howard, J., & Ruder, S. (2018)](https://arxiv.org/pdf/1801.06146.pdf). Fine-tuned Language
#  Models for Text Classification. ArXiv, abs/1801.06146. [↩](#a1)
#
#  </li>
# <li>
#
# [Chronopoulou, A., Baziotis, C., & Potamianos, A. (2019)](https://arxiv.org/pdf/1902.10547.pdf).
#  An embarrassingly simple approach for transfer learning from pretrained language models. arXiv
#  preprint arXiv:1902.10547. [↩](#a1)
#
#  </li>
# <li>
#
# [Peters, M. E., Ruder, S., & Smith, N. A. (2019)](https://arxiv.org/pdf/1903.05987.pdf). To tune or not to
#  tune? adapting pretrained representations to diverse tasks. arXiv preprint arXiv:1903.05987. [↩](#a1)
#
# </li>
# <li id="f2">
#
# [Sivaprasad, P. T., Mai, F., Vogels, T., Jaggi, M., & Fleuret, F. (2020)](https://arxiv.org/pdf/1910.11758.pdf).
#  Optimizer benchmarking needs to account for hyperparameter tuning. In International Conference on Machine Learning
# (pp. 9036-9045). PMLR. [↩](#a2)
#
# </li>
# <li id="f3">
#
# [Mosbach, M., Andriushchenko, M., & Klakow, D. (2020)](https://arxiv.org/pdf/2006.04884.pdf). On the stability of
# fine-tuning bert: Misconceptions, explanations, and strong baselines. arXiv preprint arXiv:2006.04884. [↩](#a2)
#
# </li>
# <li id="f4">
#
# [Loshchilov, I., & Hutter, F. (2016)](https://arxiv.org/pdf/1608.03983.pdf). Sgdr: Stochastic gradient descent with
# warm restarts. arXiv preprint arXiv:1608.03983. [↩](#a3)
#
# </li>
#
# </ol>
