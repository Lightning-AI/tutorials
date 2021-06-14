#!/bin/bash

set -e
printf "Converting: $1"

# check that there is only one python script
python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '*.py'))) == 1)"
# check that there is meta file
python -c "import os ; assert any(os.path.isfile(os.path.join('$1', f'.meta{ext}')) for ext in ['.yml', '.yaml'])"
py_file=( $(ls "$1"/*.py) )
printf $py_file

python .actions/helpers.py augment-script $py_file

python -m jupytext --set-formats "ipynb,py:percent" $py_file
