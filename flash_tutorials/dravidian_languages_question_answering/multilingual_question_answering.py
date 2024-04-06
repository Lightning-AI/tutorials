# %% [markdown]
# In this tutorial we'll look at using [Lightning Flash](https://github.com/PyTorchLightning/lightning-flash) and it's
# integration with [Hugging Face Transformers](https://github.com/huggingface/transformers) for question answering of
# dravidian language based corpus using [the XLM-RoBERTa model](https://arxiv.org/pdf/1911.02116.pdf).

# %%

import os

import pandas as pd
import torch
from flash import Trainer
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

# %% [markdown]
# ## Loading the Data and generating splits
#
# To load the data, we start by creating a train, validation, and test splits:

# %%
DATASET_PATH = os.environ.get("PATH_DATASETS", "_datasets")
CHAII_DATASET_PATH = os.path.join(DATASET_PATH, "chaii-hindi-and-tamil-question-answering")
INPUT_DATA_PATH = os.path.join(CHAII_DATASET_PATH, "train.csv")
TRAIN_DATA_PATH = os.path.join(CHAII_DATASET_PATH, "_train.csv")
VAL_DATA_PATH = os.path.join(CHAII_DATASET_PATH, "_val.csv")
PREDICT_DATA_PATH = os.path.join(CHAII_DATASET_PATH, "test.csv")

df = pd.read_csv(INPUT_DATA_PATH)
fraction = 0.9

tamil_examples = df[df["language"] == "tamil"]
train_split_tamil = tamil_examples.sample(frac=fraction, random_state=200)
val_split_tamil = tamil_examples.drop(train_split_tamil.index)

hindi_examples = df[df["language"] == "hindi"]
train_split_hindi = hindi_examples.sample(frac=fraction, random_state=200)
val_split_hindi = hindi_examples.drop(train_split_hindi.index)

train_split = pd.concat([train_split_tamil, train_split_hindi]).reset_index(drop=True)
val_split = pd.concat([val_split_tamil, val_split_hindi]).reset_index(drop=True)

train_split.to_csv(TRAIN_DATA_PATH, index=False)
val_split.to_csv(VAL_DATA_PATH, index=False)

# %% [markdown]
# ## Creating the Flash DataModule
#
# Now, we can create a `QuestionAnsweringData`.
# Flash supports a wide variety of input formats, each having its method with the naming format as `from_xxxx`.
# Our datasets are available as CSV files, and it is the same format in which we saved the splits. Hence, we use the
# `from_csv` method to generate the DataModule. The simplest form of the API only requires the data files, the Hugging
# Face backbone of your choice, and batch size. Flash takes care of preprocessing the data, i.e., tokenizing using the
# Hugging Face tokenizer and creating the Datasets.
#
# Here's the full preprocessing function:

# %%

datamodule = QuestionAnsweringData.from_csv(
    train_file=TRAIN_DATA_PATH,
    val_file=VAL_DATA_PATH,
    batch_size=4,
    backbone="xlm-roberta-base",
)

# %% [markdown]
# ## Creating the Flash Task
#
# The API for building the NLP Task is also simple. For all Flash models, the naming pattern follows `XYZTask`, and
# thus we will be using the `QuestionAnsweringTask` in this case. The power of Flash's simplicity comes into play here
# as we pass the required backbone, Optimizer of choice, and the preferable learning rate for the model. Then Flash
# takes care of the rest, i.e., downloading the model, instantiating the model, configuring the Optimizer, and even
# logging the losses.

# %%
model = QuestionAnsweringTask(
    backbone="xlm-roberta-base",
    learning_rate=1e-5,
    optimizer="adamw",
)

# %% [markdown]
# ## Setting up the Trainer and Fine-Tuning the model
#
# Flash's Trainer is inherited from Lightning's Trainer and provides an additional method `finetune` that takes in an
# extra argument `strategy` that lets us specify a specific strategy for fine-tuning the backbone. We will be using
# the `freeze_unfreeze` strategy to fine-tune the model, which freezes the gradients of the backbone transformer
# containing the pre-trained weights and trains just the new model head for a certain number of epochs and unfreezes
# the backbone after which the complete model (backbone + head) is trained for the remaining epochs.
#
# Check out the documentation to learn about the other strategies provided by Flash, and feel free to reach out and
# contribute any new fine-tuning methods to the project.

# %%
trainer = Trainer(
    max_epochs=5,
    accumulate_grad_batches=2,
    gpus=int(torch.cuda.is_available()),
)

trainer.finetune(model, datamodule, strategy=("freeze_unfreeze", 2))

# %% [markdown]
# ## Making predictions
#
# We convert the prediction file provided to us from a pandas DataFrame to a python dictionary object and pass it to
# the model as predictions.

# %%
predict_data = pd.read_csv(PREDICT_DATA_PATH)
predict_data = predict_data[predict_data.columns[:3]].to_dict(orient="list")

predictions = model.predict(predict_data)
print(predictions)

# %% [markdown]
# ## Closing thoughts and next steps!
#
# This tutorial has shown how Flash and Hugging Face Transformers can be used to train a state-of-the-art language
# model (such as XLM-RoBERTa).
#
# If you want to be a bit more adventurous, you could look at
# [some of the other problems that can solved with Lightning Flash](https://lightning-flash.readthedocs.io/en/stable/?badge=stable).
