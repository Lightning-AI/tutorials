import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from flash import Trainer
from flash.tabular import TabularClassificationData, TabularClassifier

# %% [markdown]
# ## 1. Create the DataModule
#
# ### Variable & Definition
#
# - survival: Survival (0 = No, 1 = Yes)
# - pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# - sex: Sex
# - Age: Age in years
# - sibsp: number of siblings / spouses aboard the Titanic
# - parch: number of parents / children aboard the Titanic
# - ticket: Ticket number
# - fare: Passenger fare
# - cabin: Cabin number
# - embarked: Port of Embarkation

# %%
data_path = os.environ.get("PATH_DATASETS", "_datasets")
path_titanic = os.path.join(data_path, "titanic")
csv_train = os.path.join(path_titanic, "train.csv")
csv_test = os.path.join(path_titanic, "test.csv")

df_train = pd.read_csv(csv_train)
df_train["Survived"].hist(bins=2)

# %%
datamodule = TabularClassificationData.from_csv(
    categorical_fields=["Sex", "Embarked", "Cabin"],
    numerical_fields=["Fare", "Age", "Pclass", "SibSp", "Parch"],
    target_fields="Survived",
    train_file=csv_train,
    val_split=0.1,
    batch_size=8,
)

# %% [markdown]
# ## 2. Build the task

# %%
model = TabularClassifier.from_data(
    datamodule,
    learning_rate=0.1,
    optimizer="Adam",
    n_a=8,
    gamma=0.3,
)

# %% [markdown]
# ## 3. Create the trainer and train the model

# %%
from pytorch_lightning.loggers import CSVLogger  # noqa: E402]

logger = CSVLogger(save_dir="logs/")
trainer = Trainer(
    max_epochs=10,
    gpus=torch.cuda.device_count(),
    logger=logger,
    accumulate_grad_batches=12,
    gradient_clip_val=0.1,
)

# %%

trainer.fit(model, datamodule=datamodule)

# %%

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics.set_index("step", inplace=True)
del metrics["epoch"]
sns.relplot(data=metrics, kind="line")
plt.gca().set_ylim([0, 1.25])
plt.gcf().set_size_inches(10, 5)

# %% [markdown]
# ## 4. Generate predictions from a CSV

# %%
df_test = pd.read_csv(csv_test)

predictions = model.predict(csv_test)
print(predictions[0])

# %%
import numpy as np  # noqa: E402]

assert len(df_test) == len(predictions)

df_test["Survived"] = np.argmax(predictions, axis=-1)
df_test.set_index("PassengerId", inplace=True)
df_test["Survived"].hist(bins=5)
