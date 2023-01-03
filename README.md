# PytorchLightning Tutorials

[![CI internal](https://github.com/Lightning-AI/tutorials/actions/workflows/ci_test-acts.yml/badge.svg?event=push)](https://github.com/Lightning-AI/tutorials/actions/workflows/ci_test-acts.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/Tutorials/_apis/build/status/Lightning-AI.tutorials%20%5Bpublish%5D?branchName=main)](https://dev.azure.com/Lightning-AI/Tutorials/_build/latest?definitionId=29&branchName=main)
[![codecov](https://codecov.io/gh/Lightning-AI/tutorials/branch/main/graph/badge.svg?token=C6T3XOOR56)](https://codecov.io/gh/Lightning-AI/tutorials)
[![Deploy Docs](https://github.com/Lightning-AI/tutorials/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/Lightning-AI/tutorials/actions/workflows/docs-deploy.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/tutorials/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/tutorials/main)

This is the Lightning Library - collection of Lightning related notebooks which are pulled back to the main repo as submodule and rendered inside the main documentations.
The key features/highlights:

- we keep the repo **light-weighted** - notebooks are stored in rich script format
- all scripts/notebooks are tested to be **fully executable**
- fully **reproducible** by saving runtime env. details

For more details read our blogpost - [Best Practices for Publishing PyTorch Lightning Tutorial Notebooks](https://devblog.pytorchlightning.ai/publishing-lightning-tutorials-cbea3eaa4b2c)

## Adding/Editing notebooks

This repo in main branch contain only python scripts with markdown extensions, and notebooks are generated in special publication branch, so no raw notebooks are accepted as PR.
On the other hand we highly recommend creating a notebooks and convert it script with [jupytext](https://jupytext.readthedocs.io/en/latest/) as

```bash
jupytext --set-formats ipynb,py:percent my-notebook.ipynb
```

### Contribution structure

The addition has to formed as new folder

- the folder name is used for the future notebooks
- single python scripts with converted notebooks (name does not matter)
- metadata named `.meta.yaml` including following info:
  ```yaml
  title: Sample notebooks
  author: [User](contact)
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  license: CC BY-SA
  # multi-line
  description: |
    This notebook will walk you through ...
  requirements:
    - package  # with version if needed
  # define supported - CPU|GPU|TPU
  accelerator:
    - CPU
  ```

### Using datasets

It is quite common to use some public or competition's dataset for your example.
We facilitate this via defining the data sources in the metafile.
There are two basic options, download a file from web or pul Kaggle dataset:

```yaml
datasets:
  web:
    - https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  kaggle:
    - titanic
```

In both cases, the downloaded archive (Kaggle dataset is originally downloaded as zip file) is extracted to the default dataset folder under sub-folder with the same name as the downloaded file.
To get path to this dataset folder, please use environment variable `PATH_DATASETS`, so in your script use:

```py
import os

data_path = os.environ.get("PATH_DATASETS", "_datasets")
path_titanic = os.path.join(data_path, "titatnic")
```

**Warning:** some Kaggle datasets can be quite large and the process is - downloading and extracting, which means that particular runner needs to have double free space. For this reason, the CPU runner is limited to 3GB datasets.

### Suggestions

- For inserting images into text cells use MarkDown formatting, so we can insert inline images to the notebooks directly and drop eventual dependency on internet connection -> generated notebooks could be better shared offline
- If your images need special sizes, use `![Cation](my-image.png){height="60px" width="240px"}`
- If your notebook is computational or any other resource (CPU/RAM) demanding use only GPU accelerator option in meta config

### Known limitations

- Nothing major at this moment

## Meantime notes

On the back side of publishing workflow you can find in principle these three steps

```bash
# 1) convert script to notebooks
jupytext --set-formats ipynb,py:percent notebook.py

# 2) testing the created notebook
pytest -v notebook.ipynb  --nbval

# 3) generating notebooks outputs
papermill in-notebook.ipynb out-notebook.ipynb
```
