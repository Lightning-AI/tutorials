#!/bin/bash

set -e
printf "Converting: $1\n\n"

# check that there is only one python script
python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '*.py'))) == 1)"
# check that there is exactly one meta recipe
python -c "import os ; assert sum(os.path.isfile(os.path.join('$1', f'.meta{ext}')) for ext in ['.yml', '.yaml']) == 1"
py_file=( $(ls "$1"/*.py) )
printf $py_file

# add header and footer to the python script
python .actions/assistant.py augment-script $py_file

# generate notebook from given script
python -m jupytext --set-formats "ipynb,py:percent" $py_file
