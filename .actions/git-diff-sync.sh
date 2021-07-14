#!/bin/bash

set -e
printf "Detect changes for: $1 >> $2\n\n"

b1="${1//'/'/'_'}"
printf "Branch alias: $b1\n"
# list all dirs in source branch
python -c "import os, glob ; dirs = glob.glob('*') + glob.glob('*/**') ; dirs = sorted([p for p in dirs if os.path.isdir(p)]) ; print(os.linesep.join(dirs))" > "dirs-$b1.txt"
cat "dirs-$b1.txt"

head=$(git rev-parse origin/$2)
git diff --name-only $head --output=target-diff.txt
printf "\nRaw changes:\n"
cat target-diff.txt

git checkout $2
b2="${2//'/'/'_'}"
printf "Branch alias: $b2\n"
# list all dirs in target branch
python -c "
import os
from glob import glob
from os.path import sep, splitext
ipynbs = glob('.notebooks/*.ipynb') + glob('.notebooks/**/*.ipynb')
ipynbs = sorted([splitext(sep.join(p.split(sep)[1:]))[0] for p in ipynbs])
print(os.linesep.join(ipynbs))" > "dirs-$b2.txt"
cat "dirs-$b2.txt"

printf "\n\n"
git merge -s resolve origin/$1

python .actions/helpers.py group-folders target-diff.txt --fpaths_actual_dirs "['dirs-$b1.txt', 'dirs-$b2.txt']"
printf "\n\nChanged folders:\n"
cat changed-folders.txt
printf "\n\nDropped folders:\n"
cat dropped-folders.txt
printf "\n"
