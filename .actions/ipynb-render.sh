#!/bin/bash

echo "Rendering: $1"

python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
echo $ipynb_file

pub_file=".notebooks/$1.ipynb"
echo $pub_file

pub_dir="$(dirname "$pub_file")"
mkdir -p $pub_dir

python .actions/helpers.py parse-requirements $1
pip install --quiet --requirement requirements.txt
pip install --requirement "$1/requirements.txt"

papermill $ipynb_file $pub_file

git add $pub_file
