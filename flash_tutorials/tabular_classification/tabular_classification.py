#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PyTorchLightning/lightning-flash/blob/master/flash_notebooks/tabular_classification.ipynb" target="_parent">
#     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# In this notebook, we'll go over the basics of lightning Flash by training a TabularClassifier on [Titanic Dataset](https://www.kaggle.com/c/titanic).
# 
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [Flash documentation](https://lightning-flash.readthedocs.io/en/latest/)
#   - Check out [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://www.pytorchlightning.ai/community)

# # Training

# In[ ]:


# %%capture
get_ipython().system(" pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[tabular]'")


# In[ ]:


from torchmetrics.classification import Accuracy, Precision, Recall

import flash
from flash.core.data.utils import download_data
from flash.tabular import TabularClassifier, TabularClassificationData


# ###  1. Download the data
# The data are downloaded from a URL, and save in a 'data' directory.

# In[ ]:


download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')


# ###  2. Load the data
# Flash Tasks have built-in DataModules that you can use to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# 
# Creates a TabularData relies on [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 

# In[ ]:


datamodule = TabularClassificationData.from_csv(
    ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    ["Fare"],
    target_fields="Survived",
    train_file="./data/titanic/titanic.csv",
    test_file="./data/titanic/test.csv",
    val_split=0.25,
    batch_size=8,
)


# ###  3. Build the model
# 
# Note: Categorical columns will be mapped to the embedding space. Embedding space is set of tensors to be trained associated to each categorical column. 

# In[ ]:


model = TabularClassifier.from_data(datamodule)


# ###  4. Create the trainer. Run 10 times on data

# In[ ]:


trainer = flash.Trainer(max_epochs=10)


# ###  5. Train the model

# In[ ]:


trainer.fit(model, datamodule=datamodule)


# ###  6. Test model

# In[ ]:


trainer.test(model, datamodule=datamodule)


# ###  7. Save it!

# In[ ]:


trainer.save_checkpoint("tabular_classification_model.pt")


# # Predicting

# ###  8. Load the model from a checkpoint
# 
# `TabularClassifier.load_from_checkpoint` supports both url or local_path to a checkpoint. If provided with an url, the checkpoint will first be downloaded and laoded to re-create the model. 

# In[ ]:


model = TabularClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/0.7.0/tabular_classification_model.pt")


# ###  9. Generate predictions from a sheet file! Who would survive?
# 
# `TabularClassifier.predict` support both DataFrame and path to `.csv` file.

# In[ ]:


datamodule = TabularClassificationData.from_csv(
    predict_file="data/titanic/titanic.csv",
    parameters=datamodule.parameters,
    batch_size=8,
)
predictions = trainer.predict(model, datamodule=datamodule)


# In[ ]:


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
