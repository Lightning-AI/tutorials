# PytorchLightning Tutorials

[![Build Status](https://dev.azure.com/PytorchLightning/Tutorials/_apis/build/status/PyTorchLightning.Tutorials-publishing?branchName=main)](https://dev.azure.com/PytorchLightning/Tutorials/_build/latest?definitionId=11&branchName=main)
[![Code formatting](https://github.com/PyTorchLightning/lightning-tutorials/actions/workflows/ci_code-format.yml/badge.svg?event=push)](https://github.com/PyTorchLightning/lightning-tutorials/actions/workflows/ci_code-format.yml)
[![Deploy Docs](https://github.com/PyTorchLightning/lightning-tutorials/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/PyTorchLightning/lightning-tutorials/actions/workflows/docs-deploy.yml)

This is the Lightning Library - collection of Lightning related notebooks which are pulled back to the main repo as submodule and rendered inside the main documentations.
The key features/highlights:
* we keep the repo **light-weighted** - notebooks are stored in rich script format
* all scripts/notebooks are tested to be **fully executable**
* fully **reproducible** by saving runtime env. details

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
    license: CC
    # multi-line
    description: |
      This notebook will walk you through ...
    requirements:
      - package  # with version if needed
    # define supported - CPU|GPU|TPU
    accelerator:
      - CPU
    ```

### Known limitations

When you want to rename folder/notebooks, it has to be split as add new and remove in two separate PRs.

## Meantime notes

On the back side of publishing workflow you can find in principle these three steps
```bash
# 1) convert script to notebooks
jupytext --set-formats ipynb,py:percent notebook.py

# 2) testing the created notebook
treon -v notebook.ipynb

# 3) generating notebooks outputs
papermill in-notebook.ipynb out-notebook.ipynb
```
