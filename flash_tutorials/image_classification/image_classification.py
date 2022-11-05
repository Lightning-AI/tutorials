# %% [markdown]
# In this tutorial, we'll go over the basics of lightning Flash by finetuning/predictin with an ImageClassifier on [Hymenoptera Dataset](https://www.kaggle.com/ajayrana/hymenoptera-data) containing ants and bees images.
#
# # Finetuning
#
# Finetuning consists of four steps:
#
#  - 1. Train a source neural network model on a source dataset. For computer vision, it is traditionally  the [ImageNet dataset](http://www.image-net.org). As training is costly, library such as [Torchvision](https://pytorch.org/vision/stable/index.html) library supports popular pre-trainer model architectures . In this notebook, we will be using their [resnet-18](https://pytorch.org/hub/pytorch_vision_resnet/).
#
#  - 2. Create a new neural network  called the target model. Its architecture replicates the source model and parameters, expect the latest layer which is removed. This model without its latest layer is traditionally called a backbone
#
#  - 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.
#
#  - 4. Train the target model on a target dataset, such as Hymenoptera Dataset with ants and bees. However, freezing some layers at training start such as the backbone tends to be more stable. In Flash, it can easily be done with `trainer.finetune(..., strategy="freeze")`. It is also common to `freeze/unfreeze` the backbone. In `Flash`, it can be done with `trainer.finetune(..., strategy="freeze_unfreeze")`. If one wants more control on the unfreeze flow, Flash supports `trainer.finetune(..., strategy=MyFinetuningStrategy())` where `MyFinetuningStrategy` is subclassing `pytorch_lightning.callbacks.BaseFinetuning`.

# %%

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# %% [markdown]
# ## Download data
# The data are downloaded from a URL, and save in a 'data' directory.

# %%
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")


# %% [markdown]
# ## Load the data
#
# Flash Tasks have built-in DataModules that you can use to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# Creates a ImageClassificationData object from folders of images arranged in this way:</h4>
#
#    train/dog/xxx.png
#    train/dog/xxy.png
#    train/dog/xxz.png
#    train/cat/123.png
#    train/cat/nsdf3.png
#    train/cat/asd932.png

# %%
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
    batch_size=1,
)


# %% [markdown]
# ## Build the model
# Create the ImageClassifier task. By default, the ImageClassifier task uses a [resnet-18](https://pytorch.org/hub/pytorch_vision_resnet/) backbone to train or finetune your model.
# For [Hymenoptera Dataset](https://www.kaggle.com/ajayrana/hymenoptera-data) containing ants and bees images, ``datamodule.num_classes`` will be 2.
# Backbone can easily be changed with `ImageClassifier(backbone="resnet50")` or you could provide your own `ImageClassifier(backbone=my_backbone)`

# %%
model = ImageClassifier(num_classes=datamodule.num_classes)


# %% [markdown]
# ## Create the trainer. Run once on data
# The trainer object can be used for training or fine-tuning tasks on new sets of data.
# You can pass in parameters to control the training routine- limit the number of epochs, run on GPUs or TPUs, etc.
# For more details, read the  [Trainer Documentation](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=Trainer).
# In this demo, we will limit the fine-tuning to run just one epoch using max_epochs=2.

# %%
trainer = flash.Trainer(max_epochs=1)


# %% [markdown]
# ## Finetune the model

# %%
trainer.finetune(model, datamodule=datamodule, strategy="freeze")


# %% [markdown]
# ## Test the model

# %%
trainer.test(model, datamodule=datamodule)


# %% [markdown]
# ## Save it!

# %%
trainer.save_checkpoint("image_classification_model.pt")

# %% [markdown]
# ## Predicting
# **Load the model from a checkpoint**

# %%
model = ImageClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/0.7.0/image_classification_model.pt"
)

# %% [markdown]
# **Predict what's on a few images! ants or bees?**

# %%
datamodule = ImageClassificationData.from_files(
    predict_files=[
        "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
        "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
        "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
    ],
    batch_size=1,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
