#!/bin/bash

set -e
echo "Testing: $1"

python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
py_file=( $(ls "$1"/*.py) )
echo $ipynb_file

pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed

python .actions/helpers.py parse-requirements $1
cat "$1/requirements.txt"

python -m virtualenv --system-site-packages "$1/venv"
source "$1/venv/bin/activate"
pip --version
pip install --requirement "$1/requirements.txt"
pip list

#python $py_file
python -m treon -v $ipynb_file

deactivate
