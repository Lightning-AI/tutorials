shadow folder for generated notebooks, no uploading here

## reBuild all notebooks

```bash
git checkout -b publication main
make ipynb
git commit -m "regenerate all notebooks"
git push
```
