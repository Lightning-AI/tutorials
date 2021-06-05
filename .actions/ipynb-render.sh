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
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
cat "$1/requirements.txt"
pip install --requirement "$1/requirements.txt"

echo "available: $ACCELERATOR"
accel=$(python .actions/helpers.py valid-accelerator $1 2>&1)
if [ $accel == 1 ]
then
  echo "Processing: $ipynb_file"
  python -m papermill.cli $ipynb_file $pub_file
  python .actions/helpers.py update-env-details $1
else
  echo "WARNING: not valid accelerator so no outputs will be generated"
  cp $ipynb_file $pub_file
fi

git add ".notebooks/$1.yaml"
git add $pub_file
