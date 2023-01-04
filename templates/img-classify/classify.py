# %%
import os

import flash
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from flash.image import ImageClassificationData, ImageClassifier
from IPython.core.display import display
from pytorch_lightning.loggers import CSVLogger

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
# this dataset is automatically downloaded and extracted based on meta link
# this archive includes the one more level - folder with the same name
DATA_HYMENOPLERA = os.path.join(PATH_DATASETS, "hymenoptera_data", "hymenoptera_data")

# %% [markdown]
# ## 1. Create the DataModule

# %%
datamodule = ImageClassificationData.from_folders(
    train_folder=f"{DATA_HYMENOPLERA}/train/",
    val_folder=f"{DATA_HYMENOPLERA}/val/",
    batch_size=1024,
)

# %% [markdown]
# ## 2. Build the task

# %%
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# %% [markdown]
# ## 3. Create the trainer and finetune the model

# %%
logger = CSVLogger(save_dir="logs/")
trainer = flash.Trainer(logger=logger, max_epochs=3, gpus=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# %%
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())

g = sn.relplot(data=metrics, kind="line")
plt.gcf().set_size_inches(12, 4)
plt.grid()

# %% [markdown]
# ## 4. Predict what's on a few images! ants or bees?

# %%
datamodule = ImageClassificationData.from_files(
    predict_files=[
        f"{DATA_HYMENOPLERA}/val/bees/65038344_52a45d090d.jpg",
        f"{DATA_HYMENOPLERA}/val/bees/590318879_68cf112861.jpg",
        f"{DATA_HYMENOPLERA}/val/ants/540543309_ddbb193ee5.jpg",
    ],
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# %% [markdown]
# ## 5. Save the model!

# %%
trainer.save_checkpoint("image_classification_model.pt")
