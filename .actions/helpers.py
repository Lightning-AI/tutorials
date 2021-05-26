

import os
from pprint import pprint

import click
import yaml

SKIP_DIRS = (
    ".actions",
    ".github",
)
META_FILE = ".meta.yml"
REQUIREMENTS_FILE = "requirements.txt"


@click.group()
def cli():
    pass


@click.command()
@click.argument('fpath_gitdiff')
@click.argument('fpath_folders')
def filter_folders(fpath_gitdiff: str = "master-diff.txt", fpath_folders: str = "changed_folders.txt") -> None:
    """
    Generate the git change list:
    > head=$(git rev-parse origin/main)
    > git diff --name-only $head --output=master-diff.txt
    """
    with open(fpath_gitdiff, "r") as fp:
        changed = [ln.strip() for ln in fp.readlines()]

    # unique folders
    dirs = set([os.path.dirname(ln) for ln in changed])
    # not empty paths
    dirs = [ln for ln in dirs if ln]
    # drop folder with skip folder
    dirs = [pd for pd in dirs if not any(nd in SKIP_DIRS for nd in pd.split(os.path.sep))]
    # valid folder has meta
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, META_FILE))]

    with open(fpath_folders, "w") as fp:
        fp.write(os.linesep.join(dirs))


@click.command()
@click.argument('dir_path')
def parse_requirements(dir_path: str):
    fpath = os.path.join(dir_path, META_FILE)
    assert os.path.isfile(fpath)
    meta = yaml.safe_load(open(fpath))
    pprint(meta)

    req = meta.get('requirements', [])
    fname = os.path.join(dir_path, REQUIREMENTS_FILE)
    print(fname)
    with open(fname, "w") as fp:
        fp.write(os.linesep.join(req))


if __name__ == '__main__':
    cli.add_command(filter_folders)
    cli.add_command(parse_requirements)
    cli()
