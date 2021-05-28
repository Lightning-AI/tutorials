# Pytorch lightning Examples

[![Publish notebook](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/pub_notebooks.yml/badge.svg)](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/pub_notebooks.yml)
[![Code formatting](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/ci_code-format.yml/badge.svg?event=push)](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/ci_code-format.yml)
[![Deploy Docs](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/PyTorchLightning/lightning-examples/actions/workflows/docs-deploy.yml)

inspiration:
- https://keras.io/examples/
- https://project-awesome.org/markusschanta/awesome-jupyter

## Meantime notes

### Preparing source scripts

Converting notebooks to script
```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```


### Preparing notebook

```bash
# convert
jupytext --set-formats ipynb,py:percent notebook.py

# testing
treon -v notebook.ipynb

# generating
papermill in-notebook.ipynb out-notebook.ipynb
```


## Remaining ideas/tasks

- add header according `.meta.yml`
- add badges GridAI/Colab
