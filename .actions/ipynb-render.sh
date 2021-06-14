#!/bin/bash

set -e
printf "Rendering: $1\n\n"

python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
printf $ipynb_file

pub_file=".notebooks/$1.ipynb"
printf $pub_file

pub_dir=$(dirname "$pub_file")
mkdir -p $pub_dir

python .actions/helpers.py parse-requirements $1
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
cat "$1/requirements.txt"
pip install --requirement "$1/requirements.txt"

printf "available: $ACCELERATOR\n"
accel=$(python .actions/helpers.py valid-accelerator $1 2>&1)
if [ $accel == 1 ]
then
  printf "Processing: $ipynb_file\n"
  python -m papermill.cli $ipynb_file $pub_file
  python .actions/helpers.py update-env-details $1
else
  printf "WARNING: not valid accelerator so no outputs will be generated.\n"
  cp $ipynb_file $pub_file
fi

git add ".notebooks/$1.yaml"
git add $pub_file
