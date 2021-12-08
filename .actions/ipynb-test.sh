#!/bin/bash

set -e
printf "Testing: $1\n\n"

# check that there is no ipython file in the folder
python -c "import glob ; assert(len(glob.glob('$1/*.ipynb')) == 1)"
ipynb_file=( $(ls "$1"/*.ipynb) )
py_file=( $(ls "$1"/*.py) )
printf $ipynb_file

pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed

# prepare the requirements specific for the particular notebook
python .actions/assistant.py parse-requirements $1
cat "$1/requirements.txt"

# prepare isolated environment with inheriting the global packages
python -m virtualenv --system-site-packages "$1/venv"
source "$1/venv/bin/activate"
pip --version
# just in case reinstall the global
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
pip_args=$(cat "$1/pip_arguments.txt")
printf "pip arguments:\n $pip_args\n\n"
# and install specific packages
pip install --requirement "$1/requirements.txt" $pip_args
pip list

printf "available: $ACCELERATOR\n"
accel=$(python .actions/assistant.py valid-accelerator $1 2>&1)
if [ $accel == 1 ]
then
  #python $py_file
  printf "Testing: $ipynb_file\n"
  python -m pytest $ipynb_file -v --nbval
else
  printf "WARNING: not valid accelerator so no tests will be run.\n"
fi

# deactivate and clean local environment
deactivate
rm -rf "$1/venv"
