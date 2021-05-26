#!/bin/bash

echo "Rendering: $1"

python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
echo $ipynb_file

pub_file=".notebook/$1.ipynb"
echo $pub_file

pub_dir="$(dirname "$pub_file")"
mkdir -p $pub_dir

# todo: parse requirements

# todo: install requirements

papermill $ipynb_file $pub_file


git add $pub_file