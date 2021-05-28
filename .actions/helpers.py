import os
from pprint import pprint

import fire
import yaml


class Helper:

    SKIP_DIRS = (
        "docs",
        ".actions",
        ".github",
    )
    META_FILE = ".meta.yml"
    REQUIREMENTS_FILE = "requirements.txt"

    @staticmethod
    def group_folders(fpath_gitdiff: str = "master-diff.txt", fpath_folders: str = "changed-folders.txt") -> None:
        """Group changes by folders

        Args:
            fpath_gitdiff: raw git changes

                Generate the git change list:
                > head=$(git rev-parse origin/main)
                > git diff --name-only $head --output=master-diff.txt

            fpath_folders: output file with folders
        """
        with open(fpath_gitdiff, "r") as fp:
            changed = [ln.strip() for ln in fp.readlines()]

        # unique folders
        dirs = set([os.path.dirname(ln) for ln in changed])
        # not empty paths
        dirs = [ln for ln in dirs if ln]
        # drop folder with skip folder
        dirs = [pd for pd in dirs if not any(nd in Helper.SKIP_DIRS for nd in pd.split(os.path.sep))]
        # valid folder has meta
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, Helper.META_FILE))]

        with open(fpath_folders, "w") as fp:
            fp.write(os.linesep.join(dirs))

    @staticmethod
    def parse_requirements(dir_path: str):
        """Parse standard requirements from meta file

        :param dir_path: path to the folder
        """
        fpath = os.path.join(dir_path, Helper.META_FILE)
        assert os.path.isfile(fpath)
        meta = yaml.safe_load(open(fpath))
        pprint(meta)

        req = meta.get('requirements', [])
        fname = os.path.join(dir_path, Helper.REQUIREMENTS_FILE)
        print(fname)
        with open(fname, "w") as fp:
            fp.write(os.linesep.join(req))


if __name__ == '__main__':
    fire.Fire(Helper)
