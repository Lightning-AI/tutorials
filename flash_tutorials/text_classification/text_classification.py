# %% [markdown]
# In this notebook, we'll go over the basics of lightning Flash by finetunig a TextClassifier on [IMDB Dataset](https://paperswithcode.com/dataset/imdb-movie-reviews).
#
# # Finetuning
#
# Finetuning consists of four steps:
#
#  - 1. Train a source neural network model on a source dataset. For text classication, it is traditionally  a transformer model such as BERT [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) trained on wikipedia.
# As those model are costly to train, [Transformers](https://github.com/huggingface/transformers) or [FairSeq](https://github.com/pytorch/fairseq) libraries provides popular pre-trained model architectures for NLP. In this notebook, we will be using [tiny-bert](https://huggingface.co/prajjwal1/bert-tiny).
#
#  - 2. Create a new neural network the target model. Its architecture replicates all model designs and their parameters on the source model, expect the latest layer which is removed. This model without its latest layers is traditionally called a backbone
#
# - 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head, will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.
#
# - 4. Train the target model on a target dataset, such as Hymenoptera Dataset with ants and bees. However, freezing some layers at training start such as the backbone tends to be more stable. In Flash, it can easily be done with `trainer.finetune(..., strategy="freeze")`. It is also common to `freeze/unfreeze` the backbone. In `Flash`, it can be done with `trainer.finetune(..., strategy="freeze_unfreeze")`. If a one wants more control on the unfreeze flow, Flash supports `trainer.finetune(..., strategy=MyFinetuningStrategy())` where `MyFinetuningStrategy` is subclassing `pytorch_lightning.callbacks.BaseFinetuning`.

# %%

import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

# %% [markdown]
# ## Download the data
# The data are downloaded from a URL, and save in a 'data' directory.

# %%
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "data/")


# %% [markdown]
# ## Load the data</h2>
#
# Flash Tasks have built-in DataModules that you can use to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# Creates a TextClassificationData object from csv file.

# %%
datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=512,  # just increased for the example to run fast
)


# %% [markdown]
# ## Build the model
#
# Create the TextClassifier task. By default, the TextClassifier task uses a [tiny-bert](https://huggingface.co/prajjwal1/bert-tiny) backbone to train or finetune your model demo. You could use any models from [transformers - Text Classification](https://huggingface.co/models?filter=text-classification,pytorch)
#
# Backbone can easily be changed with such as `TextClassifier(backbone='bert-tiny-mnli')`

# %%
model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")


# %% [markdown]
# ## Create the trainer. Run once on data

# %%
trainer = flash.Trainer(max_epochs=1)


# %% [markdown]
# ## Fine-tune the model
#
# The backbone won't be freezed and the entire model will be finetuned on the imdb dataset

# %%
trainer.finetune(model, datamodule=datamodule, strategy="freeze")


# %% [markdown]
# ## Test model

# %%
trainer.test(model, datamodule=datamodule)


# %% [markdown]
# ## Save it!

# %%
trainer.save_checkpoint("text_classification_model.pt")


# %% [markdown]
# ## Predicting
# **Load the model from a checkpoint**

# %%
model = TextClassifier.load_from_checkpoint("text_classification_model.pt")


# %% [markdown]
# **Classify a few sentences! How was the movie?**

# %%
datamodule = TextClassificationData.from_lists(
    predict_data=[
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado.",
    ],
    batch_size=4,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
