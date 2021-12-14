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
from flash import Trainer
from flash.core.classification import LabelsOutput
from flash.core.data.utils import download_data

from flash.tabular.classification import TabularClassifier, TabularClassificationData

import torchmetrics

from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")

# %%
#todo: to be changed
download_data("https://pl-flash-data.s3.amazonaws.com/kaggle_electricity.zip", DATASET_PATH)

# %%
# ## Download the data
df_train = pd.read_csv(f"{DATASET_PATH}/train.csv")
df_predict = pd.read_csv(f"{DATASET_PATH}/test.csv")

df_train = df_train.drop('ID_code', axis = 1)

# %%

# Some data inspection

# %%

sm = SMOTE(sampling_strategy='auto', random_state=42)

oversampled_X, oversampled_Y = sm.fit_resample(df_train.drop('target', axis = 1), df_train['target'])
df_upsampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
# %%

# df_upsampled can be substituted by df_train

datamodule = TabularClassificationData.from_data_frame(
    numerical_fields=["var_" + str(i) for i in range(200)],
    target_fields="target",
    train_data_frame=df_upsampled,
    predict_data_frame=df_predict,
    batch_size=256,
)

# %%

model = TabularClassifier.from_data(datamodule,
                                metrics = [torchmetrics.AUROC(num_classes=datamodule.num_classes)])
model.output = LabelsOutput()

# %%

trainer = Trainer(gpus=torch.cuda.device_count())

res = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-5)
print(f"Suggested learning rate: {res.suggestion()}")
res.plot(show=True, suggest=True).show()

model.learning_rate = res.suggestion()

trainer.fit(model, datamodule=datamodule)

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
#todo: warning, the order of the predictions might not be ordered!
id_code = ["test_" + str(i) for i in range(len(predictions[0]))]
data = {"ID_code": id_code, "target": predictions[0]}
pred_df = pd.DataFrame(data)

pred_df.to_csv("submission.csv", index=False)
