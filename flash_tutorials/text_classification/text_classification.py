#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PyTorchLightning/lightning-flash/blob/master/flash_notebooks/text_classification.ipynb" target="_parent">
#     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# In this notebook, we'll go over the basics of lightning Flash by finetunig a TextClassifier on [IMDB Dataset](https://www.imdb.com/interfaces/).
# 
# # Finetuning
# 
# Finetuning consists of four steps:
#  
#  - 1. Train a source neural network model on a source dataset. For text classication, it is traditionally  a transformer model such as BERT [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) trained on wikipedia.
# As those model are costly to train, [Transformers](https://github.com/huggingface/transformers) or [FairSeq](https://github.com/pytorch/fairseq) libraries provides popular pre-trained model architectures for NLP. In this notebook, we will be using [tiny-bert](https://huggingface.co/prajjwal1/bert-tiny).
# 
#  
#  - 2. Create a new neural network the target model. Its architecture replicates all model designs and their parameters on the source model, expect the latest layer which is removed. This model without its latest layers is traditionally called a backbone
#  
# 
# - 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head, will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.
#  
# 
# - 4. Train the target model on a target dataset, such as Hymenoptera Dataset with ants and bees. However, freezing some layers at training start such as the backbone tends to be more stable. In Flash, it can easily be done with `trainer.finetune(..., strategy="freeze")`. It is also common to `freeze/unfreeze` the backbone. In `Flash`, it can be done with `trainer.finetune(..., strategy="freeze_unfreeze")`. If a one wants more control on the unfreeze flow, Flash supports `trainer.finetune(..., strategy=MyFinetuningStrategy())` where `MyFinetuningStrategy` is subclassing `pytorch_lightning.callbacks.BaseFinetuning`.
# 
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [Flash documentation](https://lightning-flash.readthedocs.io/en/latest/)
#   - Check out [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://www.pytorchlightning.ai/community)

# ### Setup  
# Lightning Flash is easy to install. Simply ```pip install lightning-flash```

# In[ ]:


get_ipython().run_cell_magic('capture', '', "! pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[text]'\n")


# In[ ]:


import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier


# ###  1. Download the data
# The data are downloaded from a URL, and save in a 'data' directory.

# In[ ]:


download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')


# <h2>2. Load the data</h2>
# 
# Flash Tasks have built-in DataModules that you can use to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# Creates a TextClassificationData object from csv file.

# In[ ]:


datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=4,
)


# ###  3. Build the model
# 
# Create the TextClassifier task. By default, the TextClassifier task uses a [tiny-bert](https://huggingface.co/prajjwal1/bert-tiny) backbone to train or finetune your model demo. You could use any models from [transformers - Text Classification](https://huggingface.co/models?filter=text-classification,pytorch)
# 
# Backbone can easily be changed with such as `TextClassifier(backbone='bert-tiny-mnli')`

# In[ ]:


model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")


# ###  4. Create the trainer. Run once on data

# In[ ]:


trainer = flash.Trainer(max_epochs=1)


# ###  5. Fine-tune the model
# 
# The backbone won't be freezed and the entire model will be finetuned on the imdb dataset 

# In[ ]:


trainer.finetune(model, datamodule=datamodule, strategy="freeze")


# ###  6. Test model

# In[ ]:


trainer.test(model, datamodule=datamodule)


# ###  7. Save it!

# In[ ]:


trainer.save_checkpoint("text_classification_model.pt")


# # Predicting

# ### 1. Load the model from a checkpoint

# In[ ]:


model = TextClassifier.load_from_checkpoint("text_classification_model.pt")


# ### 2. Classify a few sentences! How was the movie?

# In[ ]:


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


# <code style="color:#792ee5;">
#     <h1> <strong> Congratulations - Time to Join the Community! </strong>  </h1>
# </code>
# 
# Congratulations on completing this notebook tutorial! If you enjoyed it and would like to join the Lightning movement, you can do so in the following ways!
# 
# ### Help us build Flash by adding support for new data-types and new tasks.
# Flash aims at becoming the first task hub, so anyone can get started to great amazing application using deep learning. 
# If you are interested, please open a PR with your contributions !!! 
# 
# 
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool tools we're building.
# 
# * Please, star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
# 
# ### Join our [Slack](https://www.pytorchlightning.ai/community)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself and share your interests in `#general` channel
# 
# ### Interested by SOTA AI models ! Check out [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# Bolts has a collection of state-of-the-art models, all implemented in [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and can be easily integrated within your own projects.
# 
# * Please, star [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# 
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/lightning-bolts) GitHub Issues page and filter for "good first issue". 
# 
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
# 
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
# 
# <img src="https://raw.githubusercontent.com/PyTorchLightning/lightning-flash/18c591747e40a0ad862d4f82943d209b8cc25358/docs/source/_static/images/logo.svg" width="800" height="200" />
