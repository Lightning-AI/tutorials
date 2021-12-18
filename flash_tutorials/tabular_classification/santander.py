# %% [markdown]
# In this tutorial we'll look at using [Lightning Flash](https://github.com/PyTorchLightning/lightning-flash) and it's
# integration with [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) for autoregressive modelling of
# electricity prices using [the N-BEATS model](https://arxiv.org/abs/1905.10437).
# We'll start by using N-BEATS to uncover daily patterns (seasonality) from hourly observations and then show how we can
# resample daily averages to uncover weekly patterns too.
#
# Along the way, we'll see how the built-in tools from PyTorch Lightning, like the learning rate finder, can be used
# seamlessly with Flash to help make the process of putting a model together as smooth as possible.

# %%

import os

import flash
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchmetrics
from flash import Trainer
from flash.core.classification import LabelsOutput
from flash.core.data.utils import download_data
from flash.tabular.classification import TabularClassificationData, TabularClassifier
from imblearn.over_sampling import SMOTE
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import train_test_split

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")

# %%

seed_everything(seed=42)

# todo: to be changed
download_data("https://pl-flash-data.s3.amazonaws.com/kaggle_electricity.zip", DATASET_PATH)

# %% [markdown]
# ## Loading and inspecting the data

# %%
# ## Download the data
df_train = pd.read_csv(f"{DATASET_PATH}/train.csv")
df_predict = pd.read_csv(f"{DATASET_PATH}/test.csv")

df_train = df_train.drop("ID_code", axis=1)

# %%
# Some data inspection
print(df_train.shape)
print(df_predict.shape)

df_train.describe()

# We check that there is no missing data in the training (nor test) set
df_train.isnull().values.any()

# %% [markdown]
# ## Data preprocessing
# We first do two things:
# 1. Divide the training data into a training, validation and test sets.
# 2. Resample the training data to have the same number of samples for each class, as the original training data is unbalanced.

# %%

train, rem = train_test_split(df_train, test_size=0.2)
validation, test = train_test_split(rem, test_size=0.5)

# %%

sm = SMOTE(sampling_strategy="auto", random_state=42)

oversampled_X, oversampled_Y = sm.fit_resample(train.drop("target", axis=1), train["target"])
df_upsampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

# %% [markdown]
# ## Flash TabularClassificationData DataModule
# The objective now is to create a datamodule from TabularClassificationData that can be used with Flash.
# %%

datamodule = TabularClassificationData.from_data_frame(
    numerical_fields=["var_" + str(i) for i in range(200)],
    target_fields="target",
    train_data_frame=df_upsampled,
    val_data_frame=validation,
    test_data_frame=test,
    predict_data_frame=df_predict,
    batch_size=256,
)

# %% [markdown]
# ## Flash TabularClassifier Model.
# We indicate the torchmetrics.AUROC metric to be used for evaluating the model, an LR scheduler, and the optimizer.
# %%
# It is important that one uses the modular, not the function form of the metric. That is, do not use torchmetrics.functional.auroc()

model = TabularClassifier.from_data(
    datamodule,
    metrics=[torchmetrics.AUROC(num_classes=datamodule.num_classes)],
    lr_scheduler=("ExponentialLR", {"gamma": 0.95}),
    optimizer="adamw",
)
model.output = LabelsOutput()

# %% [markdown]
# ## Trainer

# %%

trainer = Trainer(max_epochs=10, gpus=torch.cuda.device_count())

res = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-5)
print(f"Suggested learning rate: {res.suggestion()}")
res.plot(show=True, suggest=True).show()

# Automatic learning rate suggestion seems unstable, so we'll use a fixed learning rate
model.learning_rate = 5e-3  # res.suggestion()

# %% [markdown]
# ## Training
# Finally, we train the model for a few epochs and evaluate it on the test set.

# %%
trainer.fit(model, datamodule=datamodule)

# %%
trainer.validate(model, datamodule=datamodule)

# %%
trainer.test(model, datamodule=datamodule)

# %% [markdown]
# ## Predictions
# We can now predict the test set and submit it to Kaggle.
# %%
predictions = trainer.predict(model, datamodule=datamodule)

# %%
# ## We create another datamodule for predictions

predict_datamodule = TabularClassificationData.from_data_frame(
    numerical_fields=["var_" + str(i) for i in range(200)],
    predict_data_frame=df_predict,
    batch_size=df_predict.shape[0],
    parameters=datamodule.parameters,
)

predictions = trainer.predict(model, datamodule=predict_datamodule)

# %%

df_predict["target"] = predictions[0]
id_code = ["test_" + str(i) for i in range(len(predictions[0]))]
df_predict["ID_code"] = id_code

df_predict.to_csv("submission.csv", columns=["ID_code", "target"], index=False)
