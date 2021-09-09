#!/bin/bash

set -e
printf "Rendering: $1\n\n"

python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '*.ipynb'))) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
printf "$ipynb_file"

python -c "import os ; assert any(os.path.isfile(os.path.join('$1', f'.meta{ext}')) for ext in ['.yml', '.yaml'])"
meta_file=( $(ls "$1"/.meta.*) )
printf $meta_file

python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '.thumb.*'))) <= 1)"
thumb_file=( $(ls "$1"/.thumb.* 2>/dev/null || echo "") ) || true
printf $thumb_file

pub_file=".notebooks/$1.ipynb"
printf $pub_file

pub_meta_file=".notebooks/$1.yaml"
printf $pub_meta_file

if [ ! -z $thumb_file ]; then
  pub_thumb_file=".notebooks/"${thumb_file/"$1"\/.thumb/$1}
  printf $pub_thumb_file
fi

pub_dir=$(dirname "$pub_file")
mkdir -p $pub_dir

python .actions/helpers.py parse-requirements $1
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
cat "$1/requirements.txt"
pip_args=$(cat "$1/pip_arguments.txt")
printf "pip arguments:\n $pip_args\n\n"
pip install --requirement "$1/requirements.txt" $pip_args

if [ ! -z "${DRY_RUN}" ] && [ "${DRY_RUN}" = true ]; then
  cp $ipynb_file $pub_file
else
  printf "available: $ACCELERATOR\n"
  accel=$(python .actions/helpers.py valid-accelerator $1 2>&1)
  if [ $accel == 1 ]
  then
    printf "Processing: $ipynb_file\n"
    python -m papermill.cli $ipynb_file $pub_file --kernel python
    python .actions/helpers.py update-env-details $1
  else
    printf "WARNING: not valid accelerator so no outputs will be generated.\n"
    cp $ipynb_file $pub_file
  fi
fi

cp $meta_file $pub_meta_file
git add $pub_meta_file

if [ ! -z $thumb_file ]; then
  cp $thumb_file $pub_thumb_file
  git add $pub_thumb_file
fi

git add $pub_file
