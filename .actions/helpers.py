import glob
import os
import shutil
from pprint import pprint

import fire
import tqdm
import yaml

REPO_NAME = "lightning-examples"
DEFAULT_BRANCH = "main"
TEMPLATE_HEADER = f"""
# %%%% [markdown] colab_type="text" id="view-in-github"
#
# # %(title)s
#
# %(description)s
#
# ---
# Open in
# <a href="https://colab.research.google.com/github/PytorchLightning/{REPO_NAME}/blob/{DEFAULT_BRANCH}/%(local_path)s" target="_parent">
# <img src="https://colab.research.google.com/assets/colab-badge.png" alt="Open In Colab" width="117" height="20px"/></a>
#
# Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
# | Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
# | Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

"""
TEMPLATE_FOOTER = """
# %% [markdown]
# <code style="color:#792ee5;">
#     <h1> <strong> Congratulations - Time to Join the Community! </strong>  </h1>
# </code>
#
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the Lightning movement, you can do so in the following ways!
#
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool tools we're building.
#
# * Please, star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
#
# ### Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself and share your interests in `#general` channel
#
# ### Interested by SOTA AI models ! Check out [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# Bolts has a collection of state-of-the-art models, all implemented in [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and can be easily integrated within your own projects.
#
# * Please, star [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
#
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/lightning-bolts) GitHub Issues page and filter for "good first issue".
#
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
#
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
#
# <img src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png?raw=true" width="800" height="200" />

"""


class HelperCLI:

    SKIP_DIRS = (
        "docs",
        ".actions",
        ".github",
    )
    META_FILE = ".meta.yml"
    REQUIREMENTS_FILE = "requirements.txt"

    @staticmethod
    def expand_script(fpath: str):
        with open(fpath, "r") as fp:
            py_file = fp.readlines()
        fpath_meta = os.path.join(os.path.dirname(fpath), HelperCLI.META_FILE)
        meta = yaml.safe_load(open(fpath_meta))
        meta.update(dict(local_path=fpath))

        first_empty = min([i for i, ln in enumerate(py_file) if not ln.startswith("#")])
        header = TEMPLATE_HEADER % meta
        py_file[first_empty] = header
        py_file.append(TEMPLATE_FOOTER)

        with open(fpath, "w") as fp:
            fp.writelines(py_file)

    @staticmethod
    def group_folders(
        fpath_gitdiff: str,
        fpath_change_folders: str = "changed-folders.txt",
        fpath_drop_folders: str = "dropped-folders.txt",
    ) -> None:
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
        dirs = [pd for pd in dirs if not any(nd in HelperCLI.SKIP_DIRS for nd in pd.split(os.path.sep))]
        # valid folder has meta
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, HelperCLI.META_FILE))]

        with open(fpath_change_folders, "w") as fp:
            fp.write(os.linesep.join([d for d in dirs if os.path.isdir(d)]))

        with open(fpath_drop_folders, "w") as fp:
            fp.write(os.linesep.join([d for d in dirs if not os.path.isdir(d)]))

    @staticmethod
    def parse_requirements(dir_path: str):
        """Parse standard requirements from meta file

        :param dir_path: path to the folder
        """
        fpath = os.path.join(dir_path, HelperCLI.META_FILE)
        assert os.path.isfile(fpath)
        meta = yaml.safe_load(open(fpath))
        pprint(meta)

        req = meta.get('requirements', [])
        fname = os.path.join(dir_path, HelperCLI.REQUIREMENTS_FILE)
        print(fname)
        with open(fname, "w") as fp:
            fp.write(os.linesep.join(req))

    @staticmethod
    def copy_notebooks(path_root: str, path_docs_ipynb: str = "docs/source/notebooks"):
        ls_ipynb = []
        for sub in (['*.ipynb'], ['**', '*.ipynb']):
            ls_ipynb += glob.glob(os.path.join(path_root, '.notebooks', *sub))

        os.makedirs(path_docs_ipynb, exist_ok=True)
        ipynb_content = []
        for path_ipynb in tqdm.tqdm(ls_ipynb):
            ipynb = path_ipynb.split(os.path.sep)
            sub_ipynb = os.path.sep.join(ipynb[ipynb.index('.notebooks') + 1:])
            new_ipynb = os.path.join(path_docs_ipynb, sub_ipynb)
            os.makedirs(os.path.dirname(new_ipynb), exist_ok=True)
            print(f'{path_ipynb} -> {new_ipynb}')
            shutil.copy(path_ipynb, new_ipynb)
            ipynb_content.append(os.path.join('notebooks', sub_ipynb))

    @staticmethod
    def valid_accelerator(dir_path: str):
        """Parse standard requirements from meta file

        :param dir_path: path to the folder
        """
        fpath = os.path.join(dir_path, HelperCLI.META_FILE)
        assert os.path.isfile(fpath)
        meta = yaml.safe_load(open(fpath))
        # default is COU runtime
        accels = [acc.lower() for acc in meta.get("accelerator", ('CPU'))]
        os_acc = os.environ.get("ACCLERATOR", 'cpu')
        return int(os_acc in accels)


if __name__ == '__main__':
    fire.Fire(HelperCLI)
