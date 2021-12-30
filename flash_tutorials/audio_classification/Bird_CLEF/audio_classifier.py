# %% [markdown]
# # Bird CLF challenge

# %% [markdown]
# ### Imports

# %%
import os
from typing import Any, Dict, List

import flash
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
from flash.audio import AudioClassificationData
from flash.audio.classification.input import AudioClassificationInput
from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.paths import PATH_TYPE, filter_valid_files, has_file_allowed_extension, make_dataset
from flash.core.data.utilities.samples import to_samples
from flash.image import ImageClassifier
from pytorch_lightning import seed_everything
from torchaudio.transforms import Spectrogram

# %%
# Seeding for reproducibility
seed_everything(1234)

# %% [markdown]
# #### Audio parameters change as per your data location

# %%

DATA_PATH = os.environ.get("PATH_DATASETS", "_datasets")
DATASET_LOC =  = os.path.join(DATA_PATH, "birdclef-2021")
AUDIO_FOLDER = "train_short_audio"

# %% [markdown]
# #### Sneak peak of the data

# %%
train_metadata = pd.read_csv(os.path.join(DATASET_LOC, "train_metadata.csv"))
print(train_metadata.shape)
train_metadata.head()

# %% [markdown]
# ## Adding Dataloader

# %%
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")


def waveform_loader(filepath: str):
    if has_file_allowed_extension(filepath, AUDIO_EXTENSIONS):
        waveform, sr = torchaudio.load(filepath)
    else:
        raise Exception(f"File {filepath} has unsupported extension. Can only load {AUDIO_EXTENSIONS}")

    return waveform, sr


# %%


class AudioClassificationFileInputToSpectrogram(AudioClassificationInput):
    def load_data(self, folder: PATH_TYPE) -> List[Dict[str, Any]]:

        files, targets = make_dataset(folder, extensions=AUDIO_EXTENSIONS)

        if targets is None:
            files = filter_valid_files(files, valid_extensions=AUDIO_EXTENSIONS)
            return to_samples(files)

        files, targets = filter_valid_files(files, targets, valid_extensions=AUDIO_EXTENSIONS)
        self.load_target_metadata(targets)
        return to_samples(files, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        waveform, sr = waveform_loader(filepath)
        sample[DataKeys.INPUT] = Spectrogram(sr, normalized=True)(waveform).squeeze(0).numpy()

        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        # plot_specgram(sample[DataKeys.INPUT])
        return sample


# %%
datamodule = AudioClassificationData.from_folders(
    train_folder=os.path.join(DATASET_LOC, AUDIO_FOLDER),
    input_cls=AudioClassificationFileInputToSpectrogram,
    batch_size=64,
    transform_kwargs=dict(spectrogram_size=(64, 64)),
    val_split=0.2,
)

# %%
len(datamodule.train_dataset)

# %% [markdown]
# ## Model

# %%
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, backbone_kwargs={"in_chans": 1})

# %% [markdown]
# ## Training

# %%
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy=("freeze_unfreeze", 1))

# %%
trainer.save_checkpoint("audio_classification_model.pt")


# %%

# %%
