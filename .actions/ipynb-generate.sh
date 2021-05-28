#!/bin/bash

echo "Converting: $1"

# check that there is only one python script
python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '*.py'))) == 1)"
# check that there is meta file
python -c "import os ; assert(os.path.isfile(os.path.join('$1', '.meta.yml')))"
py_file=( $(ls "$1"/*.py) )
echo $py_file

python .actions/helpers.py expand_script $py_file

jupytext --set-formats "ipynb,py:percent" $py_file
