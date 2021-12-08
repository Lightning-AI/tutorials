#!/bin/bash

set -e
printf "Rendering: $1\n\n"

# check that there is no ipython file in the folder
python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '*.ipynb'))) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
printf $ipynb_file

# check that there i exactly one meta recipe
python -c "import os ; assert sum(os.path.isfile(os.path.join('$1', f'.meta{ext}')) for ext in ['.yml', '.yaml']) == 1"
meta_file=( $(ls "$1"/.meta.*) )
printf $meta_file

# check that folder has at most one thumb file
python -c "import os, glob ; assert(len(glob.glob(os.path.join('$1', '.thumb.*'))) <= 1)"
thumb_file=( $(ls "$1"/.thumb.* 2>/dev/null || echo "") ) || true
[ ! -z $thumb_file ] && printf $thumb_file

pub_file=".notebooks/$1.ipynb"
printf $pub_file

pub_meta_file=".notebooks/$1.yaml"
printf $pub_meta_file

# if a notebook has thumb image gets its path
if [ ! -z $thumb_file ]; then
  pub_thumb_file=".notebooks/"${thumb_file/"$1"\/.thumb/$1}
  printf $pub_thumb_file
fi

pub_dir=$(dirname "$pub_file")
mkdir -p $pub_dir

# just in case reinstall the global
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
# prepare the requirements specific for the particular notebook
python .actions/assistant.py parse-requirements $1
cat "$1/requirements.txt"
pip_args=$(cat "$1/pip_arguments.txt")
printf "pip arguments:\n $pip_args\n\n"
# and install specific packages
pip install --requirement "$1/requirements.txt" $pip_args

# dry run does not execute the notebooks just takes them as they are
if [ ! -z "${DRY_RUN}" ] && [ "${DRY_RUN}" == "1" ]; then
  cp $ipynb_file $pub_file
else
  printf "available: $ACCELERATOR\n"
  accel=$(python .actions/assistant.py valid-accelerator $1 2>&1)
  if [ $accel == 1 ]
  then
    printf "Processing: $ipynb_file\n"
    python -m papermill.cli $ipynb_file $pub_file --kernel python
  else
    printf "WARNING: not valid accelerator so no outputs will be generated.\n"
    cp $ipynb_file $pub_file
  fi
fi

# Export the actual packages used in runtime
python .actions/assistant.py update-env-details $1

# copy and add to version the enriched meta config
cp $meta_file $pub_meta_file
git add $pub_meta_file

# if thumb image is linked to to the notebook, copy and version it too
if [ ! -z $thumb_file ]; then
  cp $thumb_file $pub_thumb_file
  git add $pub_thumb_file
fi

# add the generated notebook to version
git add $pub_file
