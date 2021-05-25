# Pytorch lightning Examples

inspiration:
- https://keras.io/examples/
- https://project-awesome.org/markusschanta/awesome-jupyter

## notes

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
treon -v notebook.py

# generating 
papermill in-notebook.ipynb out-notebook.ipynb
```

alternative execution
```bash
jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute notebook.py
```
