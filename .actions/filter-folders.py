"""
Generate the git change list:
> head=$(git rev-parse origin/main)
> git diff --name-only $head --output=master-diff.txt

"""

import os
import sys

SKIP_DIRS = (
    ".actions",
    ".github",
)
METAFILE = ".meta.yml"


def main(fpath_gitdiff: str = "master-diff.txt", fpath_folders: str = "changed_folders.txt") -> None:
    with open(fpath_gitdiff, "r") as fp:
        changed = [ln.strip() for ln in fp.readlines()]

    # unique folders
    dirs = set([os.path.dirname(ln) for ln in changed])
    # not empty paths
    dirs = [ln for ln in dirs if ln]
    # drop folder with skip folder
    dirs = [pd for pd in dirs if not any(nd in SKIP_DIRS for nd in pd.split(os.path.sep))]
    # valid folder has meta
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, METAFILE))]

    with open(fpath_folders, "w") as fp:
        fp.write(os.linesep.join(dirs))


if __name__ == "__main__":
    main(*sys.argv[1:])
