#!/bin/bash

set -e
printf "Detect changes for: $1 >> $2\n\n"

b1="${1//'/'/'_'}"
printf "Branch alias: $b1\n"
# list all dirs in source branch
python .actions/assistant.py list_dirs > "dirs-$b1.txt"
cat "dirs-$b1.txt"

head=$(git rev-parse origin/$2)
git diff --name-only $head --output=target-diff.txt
printf "\nRaw changes:\n"
cat target-diff.txt
# transfer the source CLI version
mkdir -p _TEMP
cp -r .actions/ _TEMP/.actions/

git checkout $2
b2="${2//'/'/'_'}"
printf "Branch alias: $b2\n"
# recover the original CLI
#rm -rf .actions && mv _TEMP/.actions .actions
# list all dirs in target branch
python _TEMP/.actions/assistant.py list_dirs ".notebooks" --include_file_ext=".ipynb"  > "dirs-$b2.txt"
cat "dirs-$b2.txt"

printf "\n\n"
git merge --ff -s resolve origin/$1

python _TEMP/.actions/assistant.py group-folders target-diff.txt --fpath_actual_dirs "['dirs-$b1.txt', 'dirs-$b2.txt']"
printf "\n\nChanged folders:\n"
cat changed-folders.txt
printf "\n\nDropped folders:\n"
cat dropped-folders.txt
printf "\n"
