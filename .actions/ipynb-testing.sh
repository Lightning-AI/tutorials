#!/bin/bash

echo "Testing: $1"

python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
echo $ipynb_file

python .actions/helpers.py parse-requirements $1
pip install --quiet --requirement requirements.txt

python -m venv --system-site-packages "$1/venv"
source "$1/venv/bin/activate"
pip install --requirement "$1/requirements.txt"

treon -v $ipynb_file

deactivate
