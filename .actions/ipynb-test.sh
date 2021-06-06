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
pip install --quiet --requirement requirements.txt --upgrade-strategy only-if-needed
pip install --requirement "$1/requirements.txt"
pip list

echo "available: $ACCELERATOR"
accel=$(python .actions/helpers.py valid-accelerator $1 2>&1)
if [ $accel == 1 ]
then
  #python $py_file
  echo "Testing: $ipynb_file"
  python -m pytest $ipynb_file -v --nbval
else
  echo "WARNING: not valid accelerator so no tests will be run"
fi

deactivate
