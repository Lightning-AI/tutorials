import flash
import pandas as pd
import torch
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
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train["Survived"].hist(bins=2)

# %%
datamodule = TabularClassificationData.from_csv(
    categorical_fields=["Sex", "Embarked", "Cabin"],
    numerical_fields=["Fare", "Age", "Pclass", "SibSp", "Parch"],
    target_fields="Survived",
    train_file="/kaggle/input/titanic/train.csv",
    val_split=0.1,
    batch_size=64,
)

# %% [markdown]
# ## 2. Build the task

# %%
model = TabularClassifier.from_data(
    datamodule,
    learning_rate=0.1,
    optimizer="Adam",
    lr_scheduler=("StepLR", {"step_size": 100}),
    n_a=8,
    gamma=0.3,
)

# %% [markdown]
# ## 3. Create the trainer and train the model

from pytorch_lightning import seed_everything  # noqa: E402]
from pytorch_lightning.callbacks import StochasticWeightAveraging  # noqa: E402]

# %%
from pytorch_lightning.loggers import CSVLogger  # noqa: E402]

seed_everything(7)
swa = StochasticWeightAveraging(swa_epoch_start=0.6)
logger = CSVLogger(save_dir="logs/")
trainer = flash.Trainer(
    max_epochs=75,
    gpus=torch.cuda.device_count(),
    logger=logger,
    accumulate_grad_batches=4,
    gradient_clip_val=0.1,
    auto_lr_find=True,
)

# %%

trainer.tune(
    model,
    datamodule=datamodule,
    lr_find_kwargs=dict(min_lr=1e-5, max_lr=0.1, num_training=65),
)
print(f"Learning Rate: {model.learning_rate}")

# %%

trainer.fit(model, datamodule=datamodule)

# %%
import matplotlib.pyplot as plt  # noqa: E402]
import seaborn as sns  # noqa: E402]

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics.set_index("step", inplace=True)
del metrics["epoch"]
sns.relplot(data=metrics, kind="line")
plt.gca().set_ylim([0, 1.25])
plt.gcf().set_size_inches(10, 5)

# %% [markdown]
# ## 4. Generate predictions from a CSV

# %%
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

predictions = model.predict("/kaggle/input/titanic/test.csv")
print(predictions[0])

# %%
import numpy as np  # noqa: E402]

assert len(df_test) == len(predictions)

df_test["Survived"] = np.argmax(predictions, axis=-1)
df_test.set_index("PassengerId", inplace=True)
df_test["Survived"].hist(bins=5)
