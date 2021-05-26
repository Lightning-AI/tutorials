#!/bin/bash

echo "Converting: $1"

python -c "import glob ; assert(len(glob.glob('$1/*.py')) == 1)"
py_file=( $(ls "$1"/*.py) )
echo $py_file

jupytext --set-formats "ipynb,py:percent" $py_file
