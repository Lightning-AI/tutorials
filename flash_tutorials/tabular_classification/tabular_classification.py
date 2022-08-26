#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# In this notebook, we'll go over the basics of lightning Flash by training a TabularClassifier on [Titanic Dataset](https://www.kaggle.com/c/titanic).

# # Training
# %%

from torchmetrics.classification import Accuracy, Precision, Recall

import flash
from flash.core.data.utils import download_data
from flash.tabular import TabularClassifier, TabularClassificationData


# %% [markdown]
# ###  1. Download the data
# The data are downloaded from a URL, and save in a 'data' directory.
# %%
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')


# %% [markdown]
# ###  2. Load the data
# Flash Tasks have built-in DataModules that you can use to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# 
# Creates a TabularData relies on [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 
# %%
datamodule = TabularClassificationData.from_csv(
    ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    ["Fare"],
    target_fields="Survived",
    train_file="./data/titanic/titanic.csv",
    test_file="./data/titanic/test.csv",
    val_split=0.25,
    batch_size=8,
)


# %% [markdown]
# ###  3. Build the model
# 
# Note: Categorical columns will be mapped to the embedding space. Embedding space is set of tensors to be trained associated to each categorical column. 
# %%
model = TabularClassifier.from_data(datamodule)


# %% [markdown]
# ###  4. Create the trainer. Run 10 times on data
# %%
trainer = flash.Trainer(max_epochs=10)


# %% [markdown]
# ###  5. Train the model
# %%
trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# ###  6. Test model
# %5
trainer.test(model, datamodule=datamodule)


# %% [markdown]
# ###  7. Save it!
# %5
trainer.save_checkpoint("tabular_classification_model.pt")


# # Predicting

# %% [markdown]
# ###  8. Load the model from a checkpoint
# 
# `TabularClassifier.load_from_checkpoint` supports both url or local_path to a checkpoint. If provided with an url, the checkpoint will first be downloaded and laoded to re-create the model. 
# %%
model = TabularClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/0.7.0/tabular_classification_model.pt")


# %% [markdown]
# ###  9. Generate predictions from a sheet file! Who would survive?
# 
# `TabularClassifier.predict` support both DataFrame and path to `.csv` file.
# %%
datamodule = TabularClassificationData.from_csv(
    predict_file="data/titanic/titanic.csv",
    parameters=datamodule.parameters,
    batch_size=8,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
