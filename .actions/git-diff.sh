#!/bin/bash

set -e
echo "Detect changes for: $1 >> $2"

b1="${1//'/'/'_'}"
echo "Branch alias: $b1"
# list all dirs in source branch
python -c "import os, glob ; dirs = [p for p in glob.glob('*/**') if os.path.isdir(p)] ; print(os.linesep.join(dirs))" > "dirs-$b1.txt"
cat "dirs-$b1.txt"

head=$(git rev-parse origin/$2)
# todo: still missing past names for rename/modified
git diff --name-only $head --output=target-diff.txt
echo "Raw chnages:"
cat target-diff.txt

git checkout $2
b2="${2//'/'/'_'}"
echo "Branch alias: $b2"
# list all dirs in target branch
python -c "import os, glob ; dirs = [p for p in glob.glob('*/**') if os.path.isdir(p)] ; print(os.linesep.join(dirs))" > "dirs-$b2.txt"
cat "dirs-$b2.txt"

git merge -s resolve origin/$1

python .actions/helpers.py group-folders target-diff.txt --fpaths_actual_dirs "['dirs-$b1.txt', 'dirs-$b2.txt']"
echo "Changed folder:"
cat changed-folders.txt
echo "Dropped folder:"
cat dropped-folders.txt
