# %% [markdown]
# ## Scheduled Finetuning
#
# <div style="display:inline" id="a1">
#
# The ``FinetuningScheduler`` callback enables multi-phase,
# scheduled finetuning of foundational models. Gradual unfreezing (i.e. thawing) can help maximize
# foundational model knowledge retention while allowing (typically upper layers of) the model to
# optimally adapt to new tasks during transfer learning [1, 2, 3](#f1)
#
# </div>
#
# ``FinetuningScheduler`` orchestrates the gradual unfreezing
# of models via a finetuning schedule that is either implicitly generated (the default) or explicitly provided by the user
# (more computationally efficient). Finetuning phase transitions are driven by
# ``FTSEarlyStopping`` criteria (a multi-phase
# extension of ``EarlyStopping``), user-specified epoch transitions or a composition of the two (the default mode).
# A ``FinetuningScheduler`` training session completes when the
# final phase of the schedule has its stopping criteria met. See
# the [early stopping documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html) for more details on that callback's configuration.
#
# ![FinetuningScheduler explicit loss animation](fts_explicit_loss_anim.gif)

# %% [markdown]
# ## Basic Usage
#
# If no finetuning schedule is user-provided, ``FinetuningScheduler`` will generate a
# [default schedule](#The-Default-Finetuning-Schedule) and proceed to finetune according to the generated schedule, using default ``FTSEarlyStopping`` and ``FTSCheckpoint`` callbacks with ``monitor=val_loss``.
#
# ```python
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.finetuning_scheduler import FinetuningScheduler
# trainer = Trainer(callbacks=[FinetuningScheduler()])
# ```

# %% [markdown]
# ## The Default Finetuning Schedule
#
# Schedule definition is facilitated via the ``gen_ft_schedule`` method which dumps a default finetuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
# desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will often be less computationally efficient than a user-defined finetuning schedule but can often serve as a good baseline for subsquent explicit schedule refinement and will marginally outperform many explicit schedules.

# %% [markdown]
# ## Specifying a Finetuning Schedule
#
# To specify a finetuning schedule, it's convenient to first generate the default schedule and then alter the thawed/unfrozen parameter groups associated with each finetuning phase as desired. Finetuning phases are zero-indexed and executed in ascending order.
#
# 1. Generate the default schedule to ``Trainer.log_dir`` with the name
#    (``lightning_module.__class__.__name__``)_ft_schedule.yaml
#
# ```python
#     from pytorch_lightning import Trainer
#     from pytorch_lightning.callbacks.finetuning_scheduler import FinetuningScheduler
#     trainer = Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])
# ```
#
# 2. Alter the schedule as desired.
#
# <img src="side_by_side_yaml.svg" style="width: 100%; height: auto; max-width:800px;">
#
# 3. Once the finetuning schedule has been altered as desired, pass it to
#    ``FinetuningScheduler`` to commence scheduled training:
#
# ```python
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.finetuning_scheduler import FinetuningScheduler
#
# trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])
# ```

# %% [markdown]
# ## Early-Stopping and Epoch-Driven Phase Transition Criteria
#
#
# By default, ``FTSEarlyStopping`` and epoch-driven
# transition criteria are composed. If a ``max_transition_epoch`` is specified for a given phase, the next finetuning phase will begin at that epoch unless ``FTSEarlyStopping`` criteria are met first.
# If ``FinetuningScheduler.epoch_transitions_only`` is ``True``, ``FTSEarlyStopping`` will not be used
# and transitions will be exclusively epoch-driven.
#
#
# <div class="alert alert-info">
#
# **Tip:** Use of regex expressions can be convenient for specifying more complex schedules:
#
# <img src="emphasized_yaml.svg" style="width: 100%; height: auto; max-width:800px;">
#
# </div>
#
#
#
# The end-to-end example in this notebook ([Scheduled Finetuning For SuperGLUE](#superglue)) uses ``FinetuningScheduler`` in explicit mode to finetune a small foundational model on the [RTE](https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte) task of [SuperGLUE](https://super.gluebenchmark.com/).

# %% [markdown]
# ## Resuming Scheduled Finetuning Training Sessions
#
# Resumption of scheduled finetuning training is identical to the continuation of
# [other training sessions](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) with the caveat that the provided checkpoint must have been saved by a ``FinetuningScheduler`` session.
# ``FinetuningScheduler`` uses ``FTSCheckpoint`` (an extension of ``ModelCheckpoint``) to maintain schedule state with special metadata.
#
#
# ```python
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.finetuning_scheduler import FinetuningScheduler
# trainer = Trainer(callbacks=[FinetuningScheduler()], resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")
# ```
#
# Training will resume at the depth/level of the provided checkpoint according the specified schedule. Schedules can be altered between training sessions but schedule compatibility is left to the user for maximal flexibility. If executing a user-defined schedule, typically the same schedule should be provided for the original and resumed training sessions.
#
# By default (``FinetuningScheduler.restore_best`` is ``True``), ``FinetuningScheduler`` will attempt to restore the best available checkpoint before finetuning depth transitions.
#
# ```python
# trainer = Trainer(
#     callbacks=[FinetuningScheduler(new_incarnation_mode=True)],
#     resume_from_checkpoint="some/path/to/my_kth_best_checkpoint.ckpt",
# )
# ```
#
# To handle the edge case wherein one is resuming scheduled finetuning from a non-best checkpoint and the previous best checkpoints may not be accessible, setting ``FinetuningScheduler.new_incarnation_mode`` to
# ``True`` as above will re-intialize the checkpoint state with a new best checkpoint at the resumption depth.

# %% [markdown]
# <div class="alert alert-warning">
#
# **Note:** Currently, _FinetuningScheduler_ only supports the following _TrainingTypePlugins_:
#
# - ``DDPPlugin``
# - ``DDPShardedPlugin``
# - ``DDPSpawnPlugin``
# - ``DDPSpawnShardedPlugin``
# - ``DataParallelPlugin``
# - ``SingleDevicePlugin``
#
# </div>

# %% [markdown]
# <div id="superglue"></div>
#
# ## Scheduled Finetuning For SuperGLUE
#
# The following example demonstrates the use of ``FinetuningScheduler`` to finetune a small foundational model on the [RTE](https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte) task of [SuperGLUE](https://super.gluebenchmark.com/). Iterative early-stopping will be applied according to a user-specified schedule.
#
# ``FinetuningScheduler`` can be used to achieve non-trivial model performance improvements in both implicit and explicit scheduling contexts at an also non-trivial computational cost.
#

# %%
import logging
import os
import warnings
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY, _Registry
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# %%
# a couple helper functions to prepare code to work with the forthcoming hub and user module registry
MOCK_HUB_REGISTRY = _Registry()


def module_hub_mock(key: str, require_fqn: bool = False) -> List:
    if key.lower() == "finetuningscheduler":
        mod = import_module("pytorch_lightning.callbacks.finetuning_scheduler")
        MOCK_HUB_REGISTRY.register_classes(mod, pl.callbacks.Callback)
    else:
        raise MisconfigurationException(f"user module key '{key}' not found")
    registered_list = []
    # make registered class available by unqualified class name by default
    if not require_fqn:
        for n, c in MOCK_HUB_REGISTRY.items():
            globals()[f"{n}"] = c
        registered_list = ", ".join([n for n in MOCK_HUB_REGISTRY.names])
    else:
        registered_list = ", ".join([c.__module__ + "." + c.__name__ for c in MOCK_HUB_REGISTRY.classes])
    print(f"Imported and registered the following callbacks: {registered_list}")


def instantiate_registered_class(init: Dict[str, Any], args: Optional[Union[Any, Tuple[Any, ...]]] = None) -> Any:
    """Instantiates a class with the given args and init. Accepts class definitions in the form of a "class_path"
    or "callback_key" associated with a _Registry.

    Args:
        init: Dict of the form {"class_path":... or "callback_key":..., "init_args":...}.
        args: Positional arguments required for instantiation.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if args and not isinstance(args, tuple):
        args = (args,)
    shortcircuit_local = False if "." in init["class_path"] else True
    class_module, class_name, args_class = None, None, None
    if init.get("class_path", None):
        if not shortcircuit_local:
            class_module, class_name = init["class_path"].rsplit(".", 1)
        else:  # class is expected to be locally defined
            args_class = globals()[init["class_path"]]
    elif init.get("callback_key", None):
        callback_path = CALLBACK_REGISTRY.get(init["callback_key"], None) or MOCK_HUB_REGISTRY.get(
            init["callback_key"], None
        )
        assert callback_path, MisconfigurationException(
            f'specified callback_key {init["callback_key"]} has not been registered'
        )
        class_module, class_name = callback_path.__module__, callback_path.__name__
    else:
        raise MisconfigurationException(
            "Neither a class_path nor callback_key were included in a configuration that" "requires one"
        )
    if not shortcircuit_local:
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(**kwargs) if not args else args_class(*args, **kwargs)


# %%
# load the pl extension module we want to use. This will import all necessary callbacks.
module_hub_mock("finetuningscheduler")
# set notebook-level variables
AVAIL_GPUS = min(1, torch.cuda.device_count())
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"

# narrow our logging to adapt it for a notebook environment
for l_key in logging.Logger.manager.loggerDict.keys():
    if "pytorch_lightning" in l_key:
        logging.getLogger(l_key).setLevel("INFO")
    else:
        logging.getLogger(l_key).setLevel("CRITICAL")
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.removeHandler(pl_logger.handlers[0])
rz_logger = logging.getLogger("pytorch_lightning.utilities.distributed")
rz_logger.addHandler(logging.StreamHandler())
rz_logger.handlers[0].setLevel("INFO")


# %%
class RteBoolqDataModule(pl.LightningDataModule):
    """A ``LightningDataModule`` for using either the RTE or BoolQ SuperGLUE Hugging Face datasets."""

    task_text_field_map = {"rte": ["premise", "hypothesis"], "boolq": ["question", "passage"]}
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    # ignore warnings related tokenizers_parallelism/DataLoader parallelism tradeoff and
    #  expected logging behavior
    for warnf in [".*does not have many workers*", ".*The number of training samples.*"]:
        warnings.filterwarnings("ignore", warnf)

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = DEFAULT_TASK,
        prep_on_init: bool = False,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pin_memory: bool = False,
        tokenizers_parallelism: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizers_parallelism = tokenizers_parallelism
        self.dataloader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
        self.text_fields = self.task_text_field_map[self.task_name]
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, local_files_only=False)
        if prep_on_init:  # useful if one wants to load datasets as soon as the ``LightningDataModule`` is
            # instantiated
            self.prepare_data()
            self.setup("fit")

    def setup(self, stage):
        self.dataset = datasets.load_dataset("super_glue", self.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.task_name)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                **self.dataloader_kwargs,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch):
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
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
        task_name: str = DEFAULT_TASK,
        experiment_tag: str = "default",
    ):
        """
        Args:
            model_name_or_path (str): Path to pretrained model or identifier from https://huggingface.co/models
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            pl_lrs_cfg (Optional[Dict[str, Any]]): Defines custom overrides of pytorch lightning lr_scheduler defaults
                defined in ``_get_default_scheduler_config``
            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
        """
        super().__init__()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.pl_lrs_cfg = pl_lrs_cfg or {}
        if task_name in TASK_NUM_LABELS.keys():
            self.task_name = task_name
        else:
            self.task_name = DEFAULT_TASK
            rank_zero_warn(f"Invalid task_name '{task_name}'. Proceeding with the default task: '{DEFAULT_TASK}'")
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        self.save_hyperparameters()
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}"
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.metric = datasets.load_metric("super_glue", self.task_name, experiment_id=self.experiment_id)
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.finetuningscheduler_callback = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
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
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)
        return loss

    def init_pgs(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        pgs = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.optimizer_init["init_args"]["weight_decay"],
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
        return pgs

    def configure_optimizers(self):
        # the phase 0 parameters will have been set to require gradients during setup
        # you can initialize the optimizer with a simple requires.grad filter as is often done,
        # but in this case we pass a list of parameter groups to ensure weight_decay is
        # not applied to the bias parameter (for completeness, in this case it won't make much
        # performance difference)
        optimizer = instantiate_registered_class(args=self.init_pgs(), init=self.optimizer_init)
        scheduler = {
            "scheduler": instantiate_registered_class(args=optimizer, init=self.lr_scheduler_init),
            **self.pl_lrs_cfg,
        }
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        found_fts = [c for c in self.trainer.callbacks if isinstance(c, FinetuningScheduler)]  # type: ignore # noqa
        if found_fts:
            self.finetuningscheduler_callback = found_fts[0]
        return super().configure_callbacks()


# %%
# let's create a finetuning schedule for our model and run an explicitly scheduled finetuning training scenario with it
# Please see the documentation for a full description of the schedule format
ft_schedule_yaml = """
0:
  params: # the parameters for each phase definition can be fully specified
  - model.classifier.bias
  - model.classifier.weight
1:
  params: # or specified via a regex
  - model.albert.pooler.*
2:
  params:
  - model.albert.encoder.*.ffn_output.*
"""
ft_schedule_name = "RteBoolqModule_ft_schedule_albert_base.yaml"
with open(ft_schedule_name, "w") as f:
    f.write(ft_schedule_yaml)

# %%
datasets.set_progress_bar_enabled(False)
pl.seed_everything(42)
dm = RteBoolqDataModule(model_name_or_path="albert-base-v2", tokenizers_parallelism=False)
dm.setup("fit")
optimizer_init = {
    "class_path": "torch.optim.AdamW",
    "init_args": {"weight_decay": 1e-05, "eps": 1e-07, "lr": 1e-05},
}
lr_scheduler_init = {
    "class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
    "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1e-07},
}
pl_lrs_cfg = {"interval": "epoch", "frequency": 1, "name": "CosineAnnealingWarmRestarts"}

model = RteBoolqModule(
    model_name_or_path="albert-base-v2",
    optimizer_init=optimizer_init,
    lr_scheduler_init=lr_scheduler_init,
    pl_lrs_cfg=pl_lrs_cfg,
    experiment_tag="fts_explicit",
)
callbacks = [
    FinetuningScheduler(ft_schedule=ft_schedule_name, max_depth=2),  # type: ignore # noqa
    FTSEarlyStopping(monitor="val_loss", min_delta=0.001, patience=2),  # type: ignore # noqa
    FTSCheckpoint(monitor="val_loss", save_top_k=5),  # type: ignore # noqa
]
example_logdir = "lightning_logs"
logger = TensorBoardLogger(example_logdir, name="fts_explicit")

# %%
# optionally start tensorboard and monitor progress graphically while viewing multi-phase finetuning specific training
# logs in the cell output below by uncommenting the next 3 lines
# # !mkdir -p $example_logdir
# # %load_ext tensorboard
# # %tensorboard --logdir example_logdir
# disable progress bar by default to focus on multi-phase training logs. Set to True to re-enable if desired
show_progress = False

# %%
trainer = pl.Trainer(
    enable_progress_bar=show_progress,
    precision=16,
    gpus=AVAIL_GPUS,
    callbacks=callbacks,
    logger=logger,
)
trainer.fit(model, datamodule=dm)

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
#  </li>
#  </ul>
