# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="CI0JECKA9AnY"
# # README
#
# MWE of fine-tuning a Transformer-based speech embedder (e.g. [wav2vec 2.0](https://arxiv.org/abs/2006.11477)) on a subset of FSD50K using `pytorch_lightning` and HuggingFace `transformers`.
#
# Please refer to this executable [Colab notebook](https://colab.research.google.com/drive/1NddRCV1BtwgK6tvnylkLHY8d7t4OhAEw?usp=sharing) importing the code from this repo as well as a 500-element subset of the original FSD50K dataset for a concrete train+test example.
#
# Note: intended as an editable incentive for jumping into FSD50K and the Pytorch-Lightning+HuggingFace framework and as a showcase for an end-of-studies project -- choices have been made, and some logic has been altered to (significantly) reduce the size of the original code.
#

# Attribution and licenses:
# - [The FSD50K dataset](https://zenodo.org/record/4060432) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
# - The 500-element subset used here only includes [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/) audio samples

# %% [markdown] id="Y3ptRruWL2OP"
# # Check GPU availability

# %% id="L-wt4ld74fjq"
import torch

# %% id="iLzcZH-DL3Sz"
if torch.cuda.device_count() < 1:
    raise ValueError("Please run this notebook inside a GPU environment.")

# %% [markdown] id="NxjeSJxGw_9S"
# # Init

# %% id="e1LjqtE67h6a"
# !pip install pytorch_lightning
# !pip install transformers

# %% id="XzeAP33RilCh"
# !pip install git+https://github.com/FlorentMeyer/fsd50k_speech_model_finetuning

# %% id="kjazzR__WBGh"
import os
import os.path as osp

import pytorch_lightning as pl
from fsd50k_speech_model_finetuning.data_preparation_inspection import (
    CollatorVariableLengths,
    FSD50KDataDict,
    FSD50KDataModule,
    gather_preds,
    get_preds_fpaths,
    get_preds_max_logits_indices,
    inspect_data,
    sort_highest_logits,
    tokens_to_names,
)
from fsd50k_speech_model_finetuning.model_architecture import (
    Classifier,
    EmbedderClassifier,
    EmbedderHF,
    EmbeddingsMerger,
    Unfreeze,
)
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import Adam

# %% id="vz8PAflMXFam"
from transformers import Wav2Vec2Model, logging

logging.set_verbosity_error()

# %% [markdown] id="_3-Aacr3KFOv"
# # Configure whole pipeline

# %% [markdown] id="RaTWmuW1WAGt"
# ## Check embedder layers names to unfreeze

# %% [markdown] id="z_XRcQmFWX47"
# (Chosen layers are to be put in `FULL_CONFIG["unfreeze"]` along with the epoch at which to unfreeze.)

# %% id="Wf6FpgQiKc_W"
wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
for n, _ in wav2vec2.named_parameters():
    print(n)

# %% [markdown] id="aU9jI2MkV6Ay"
# ## Define the configuration dict

# %% id="VMpkjV0lKHxA"
LOG_INTERVAL_SAMPLES = 10

FULL_CONFIG = {
    "seed": 42,
    "datamodule_config": {
        "datadict_prm": {
            "dpath_data": osp.join(os.getcwd(), "dataset"),
        },
        "batch_size": 4,
        "collate_cls": CollatorVariableLengths,
        "shuffle": True,
        "drop_last": True,
        "dataset_prm": {
            "orig_sr": 44_100,
            "goal_sr": 16_000,
        },
        "pin_memory": True,
        "num_workers": 10,
    },
    "model_config": {
        "embedder_cls": EmbedderHF,
        "embedder_prm": {
            "model_name": Wav2Vec2Model,
            "hubpath_weights": "facebook/wav2vec2-base-960h",
        },
        "embeddings_merger_cls": EmbeddingsMerger,
        "embeddings_merger_prm": {
            "red_T": "mean",
            "red_L": "mean",
            "Ls": [_ for _ in range(12)],
        },
        "classifier_cls": Classifier,
        "classifier_prm": {
            "in_size": 768,
            "activation": nn.ReLU,
            "hidden_size": 512,
            "normalization": nn.BatchNorm1d,
        },
        "loss_cls": nn.BCEWithLogitsLoss,
        "loss_prm": {},
        "optimizer_cls": Adam,
        "optimizer_prm": {
            "lr": 1e-5,
        },
        "unfreeze": {
            "encoder.layers.11": 1,
        },
    },
    "trainer_config": {
        "max_epochs": 3,
        "auto_select_gpus": True,
        "accelerator": "gpu",
        "devices": 1,
        "check_val_every_n_epoch": 1,
        "precision": 16,
        "callbacks": [
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.5f}",
                save_top_k=-1,
                monitor="val_loss",
                mode="min",
            ),
            Unfreeze(),
        ],
        "logger": pl.loggers.TensorBoardLogger(
            save_dir="tb_logs",
        ),
    },
}

# %% [markdown] id="dUGV1hdbVsUV"
# ## Set seed and add params deduced from user's configuration

# %% id="uNNvu1Jd5kIH"
pl.seed_everything(FULL_CONFIG["seed"])

FULL_CONFIG["trainer_config"]["log_every_n_steps"] = (
    LOG_INTERVAL_SAMPLES // FULL_CONFIG["datamodule_config"]["batch_size"]
)

# %% [markdown] id="daXR-HHz4ze7"
# # Prepare (reduced) dataset

# %% [markdown] id="KZ9vf9o4UWEg"
# ## Download and unzip

# %% id="pqQlayizo8YX"
# !gdown 1cOcOEK56p6k2RNbM-10QFHOD4jenqHym -O /content/
# !unzip './FSD50K_500.zip' -d './dataset'

# %% [markdown] id="W-vcoLY1RqxH"
# ## Inspect data

# %% id="S89t3R0zi0nj"
fsd50k_datadict = FSD50KDataDict(**FULL_CONFIG["datamodule_config"]["datadict_prm"])

# %% id="F1BlO17ViNew"
train_datadict = fsd50k_datadict.get_dict("train")

# %% [markdown] id="N4SIRhKjgqw8"
# Add a dict entry containing the labels as strings for inspection.

# %% id="LiuSKBUThw1O"
train_datadict["ys_true_names"] = tokens_to_names(train_datadict["ys_true"], fsd50k_datadict.token_to_name)

# %% id="UwDU12Ewj-fA"
inspect_data(
    datadict=train_datadict,
    show_keys=["paths", "ys_true_names"],
    samples_indices=range(5),
)

# %% [markdown] id="wVe_yrk54Axz"
# # Launch fine-tuning

# %% id="BP7HBa7C87IY"
# Inside an interactive environment, logs could be observed using:
# # %load_ext tensorboard
# # %tensorboard --logdir ./tb_logs

# %% id="dZl2TdlTxUkF"
model = EmbedderClassifier(**FULL_CONFIG["model_config"])

datamodule = FSD50KDataModule(**FULL_CONFIG["datamodule_config"])

trainer = pl.Trainer(**FULL_CONFIG["trainer_config"])

trainer.fit(model, datamodule=datamodule)

# %% [markdown] id="jzVXKW-L4E2R"
# # Evaluate performance

# %% [markdown] id="bPbyVevnZXTX"
# ## Predict on test data

# %% id="6VTZHQDw2y5p"
preds = trainer.predict(ckpt_path="best", datamodule=datamodule)

# %% id="FM0fc3UbSp-2"
preds = gather_preds(preds)

# %% [markdown] id="cD556ynvAbYa"
# ## Compute metrics

# %% id="zlTooqqp8FWk"
mAP_micro = average_precision_score(
    preds["ys_true"],
    preds["logits"],
    average="micro",
)

# %% id="D7dpSPpf9uKS"
print("mAP_micro:", mAP_micro)

# %% [markdown] id="oqwX4z1RT8Uw"
# ## Explore samples with the highest prediction scores

# %% [markdown] id="SYH5XclifZpJ"
# Retrieve audio file paths from their IDs.
#
# For each of the samples on which a prediction was made, rank (for example, the first 4, hence the `"logits_4_highest"` key) the highest confidence logits with their corresponding class names.
#
# As an example for choosing some samples among the 100 in the test set, rank samples according to the highest logit they contain to inspect audios for which the model seemed confident -- as an alternative, one could also choose a handful of samples at random.


# %% id="N3Dk23p_O-38"
preds = get_preds_fpaths(preds)
preds = sort_highest_logits(preds, fsd50k_datadict.token_to_name, num_classes=4)
preds_max_logits_indices = get_preds_max_logits_indices(preds)

# %% id="Vnyr_f7tmNYJ"
inspect_data(
    datadict=preds,
    show_keys=["paths", "logits_4_highest", "ys_true_names"],
    samples_indices=preds_max_logits_indices[:5],
)
