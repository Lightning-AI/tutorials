# -*- coding: utf-8 -*-
import os
import re
import shutil
from datetime import datetime
from pprint import pprint
from typing import Sequence

import fire
import tqdm
import yaml
from pip._internal.operations import freeze
from wcmatch import glob

PATH_HERE = os.path.dirname(__file__)
PATH_ROOT = os.path.dirname(PATH_HERE)
PATH_REQ_DEFAULT = os.path.join(PATH_ROOT, "requirements", "default.txt")
REPO_NAME = "lightning-tutorials"
COLAB_REPO_LINK = "https://colab.research.google.com/github/PytorchLightning"
DEFAULT_BRANCH = "main"
PUBLIC_BRANCH = "publication"
URL_DOWNLOAD = f"https://github.com/PyTorchLightning/{REPO_NAME}/raw/{DEFAULT_BRANCH}"
ENV_DEVICE = "ACCELERATOR"
DEVICE_ACCELERATOR = os.environ.get(ENV_DEVICE, 'cpu').lower()
TEMPLATE_HEADER = f"""
# %%%% [markdown] colab_type="text" id="view-in-github"
#
# # %(title)s
#
# %(description)s
#
# ---
# Open in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.png){{height="20px" width="117px"}}]({COLAB_REPO_LINK}/{REPO_NAME}/blob/{PUBLIC_BRANCH}/.notebooks/%(local_ipynb)s)
#
# Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
# | Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
# | Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

"""
TEMPLATE_SETUP = """
# %%%% [markdown] colab_type="text" id="kg2MKpRmybht"
# ### Setup
# This notebook requires some packages besides pytorch-lightning.

# %%%% colab={} colab_type="code" id="LfrJLKPFyhsK"
# ! pip install --quiet %(requirements)s

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
# ![Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png?raw=true){height="60px" height="60px" width="240px"}]

"""


def default_requirements(path_req: str = PATH_REQ_DEFAULT) -> list:
    with open(path_req, 'r') as fp:
        req = fp.readlines()
    req = [r[:r.index("#")] if "#" in r else r for r in req]
    req = [r.strip() for r in req]
    req = [r for r in req if r]
    return req


def get_running_cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or ""
    except ImportError:
        return ""


def get_running_torch_version():
    try:
        import torch
        ver = torch.__version__
        return ver[:ver.index('+')] if '+' in ver else ver
    except ImportError:
        return ""


TORCH_VERSION = get_running_torch_version()
CUDA_VERSION = get_running_cuda_version()
RUNTIME_VERSIONS = dict(
    TORCH_VERSION_FULL=TORCH_VERSION,
    TORCH_VERSION=TORCH_VERSION[:TORCH_VERSION.index('+')] if '+' in TORCH_VERSION else TORCH_VERSION,
    TORCH_MAJOR_DOT_MINOR='.'.join(TORCH_VERSION.split('.')[:2]),
    CUDA_VERSION=CUDA_VERSION,
    CUDA_MAJOR_MINOR=CUDA_VERSION.replace(".", ""),
    DEVICE=f"cu{CUDA_VERSION.replace('.', '')}" if CUDA_VERSION else "cpu",
)


class HelperCLI:

    DIR_NOTEBOOKS = ".notebooks"
    SKIP_DIRS = (
        ".actions",
        ".azure-pipelines",
        ".datasets",
        ".github",
        "docs",
        DIR_NOTEBOOKS,
    )
    META_FILE_REGEX = ".meta.{yaml,yml}"
    REQUIREMENTS_FILE = "requirements.txt"
    PIP_ARGS_FILE = "pip_arguments.txt"
    META_PIP_KEY = 'pip__'

    @staticmethod
    def _meta_file(folder: str) -> str:
        files = glob.glob(os.path.join(folder, HelperCLI.META_FILE_REGEX), flags=glob.BRACE)
        if len(files) == 1:
            return files[0]

    @staticmethod
    def augment_script(fpath: str):
        """Add template header and footer to the python base script.
        Args:
            fpath: path to python script
        """
        with open(fpath, "r") as fp:
            py_file = fp.readlines()
        fpath_meta = HelperCLI._meta_file(os.path.dirname(fpath))
        meta = yaml.safe_load(open(fpath_meta))
        meta.update(dict(local_ipynb=f"{os.path.dirname(fpath)}.ipynb"))
        meta['description'] = meta['description'].replace(os.linesep, f"{os.linesep}# ")

        py_file = HelperCLI._replace_images(py_file, os.path.dirname(fpath))

        first_empty = min([i for i, ln in enumerate(py_file) if not ln.startswith("#")])
        header = TEMPLATE_HEADER % meta
        requires = set(default_requirements() + meta["requirements"])
        setup = TEMPLATE_SETUP % dict(requirements=" ".join(requires))
        py_file[first_empty] = header + setup
        py_file.append(TEMPLATE_FOOTER)

        with open(fpath, "w") as fp:
            fp.writelines(py_file)

    @staticmethod
    def _replace_images(lines: list, local_dir: str) -> list:
        """Update images by URL to GitHub raw source
        Args:
            lines: string lines from python script
            local_dir: relative path to the folder with script
        """
        md = os.linesep.join([ln.rstrip() for ln in lines])
        imgs = []
        # because * is a greedy quantifier, trying to match as much as it can. Make it *?
        imgs += re.findall(r"src=\"(.*?)\"", md)
        imgs += re.findall(r"!\[.*?\]\((.*?)\)", md)

        # update all images
        for img in set(imgs):
            url_path = '/'.join([URL_DOWNLOAD, local_dir, img])
            md = md.replace(img, url_path)

        return [ln + os.linesep for ln in md.split(os.linesep)]

    @staticmethod
    def group_folders(
        fpath_gitdiff: str,
        fpath_change_folders: str = "changed-folders.txt",
        fpath_drop_folders: str = "dropped-folders.txt",
        fpaths_actual_dirs: Sequence[str] = tuple(),
        strict: bool = True,
    ) -> None:
        """Group changes by folders
        Args:
            fpath_gitdiff: raw git changes

                Generate the git change list:
                > head=$(git rev-parse origin/main)
                > git diff --name-only $head --output=master-diff.txt

            fpath_change_folders: output file with changed folders
            fpath_drop_folders: output file with deleted folders
            fpaths_actual_dirs: files with listed all folder in particular stat
            strict: raise error if some folder outside skipped does not have valid meta file

        Example:
            >> python helpers.py group-folders ../target-diff.txt --fpaths_actual_dirs "['../dirs-main.txt', '../dirs-publication.txt']"
        """
        with open(fpath_gitdiff, "r") as fp:
            changed = [ln.strip() for ln in fp.readlines()]
        dirs = [os.path.dirname(ln) for ln in changed]
        # not empty paths
        dirs = [ln for ln in dirs if ln]

        if fpaths_actual_dirs:
            assert isinstance(fpaths_actual_dirs, list)
            assert all(os.path.isfile(p) for p in fpaths_actual_dirs)
            dir_sets = [set([ln.strip() for ln in open(fp).readlines()]) for fp in fpaths_actual_dirs]
            # get only different
            dirs += list(set.union(*dir_sets) - set.intersection(*dir_sets))

        # unique folders
        dirs = set(dirs)
        # drop folder with skip folder
        dirs = [pd for pd in dirs if not any(nd in HelperCLI.SKIP_DIRS for nd in pd.split(os.path.sep))]
        # valid folder has meta
        dirs_exist = [d for d in dirs if os.path.isdir(d)]
        dirs_invalid = [d for d in dirs_exist if not HelperCLI._meta_file(d)]
        if strict and dirs_invalid:
            raise FileNotFoundError(
                f"Following folders do not have valid `{HelperCLI.META_FILE_REGEX}` \n {os.linesep.join(dirs_invalid)}"
            )

        dirs_change = [d for d in dirs_exist if HelperCLI._meta_file(d)]
        with open(fpath_change_folders, "w") as fp:
            fp.write(os.linesep.join(dirs_change))

        dirs_drop = [d for d in dirs if not os.path.isdir(d)]
        with open(fpath_drop_folders, "w") as fp:
            fp.write(os.linesep.join(dirs_drop))

    @staticmethod
    def parse_requirements(dir_path: str):
        """Parse standard requirements from meta file
        Args:
            dir_path: path to the folder
        """
        fpath = HelperCLI._meta_file(dir_path)
        assert fpath, f"Missing Meta file in {dir_path}"
        meta = yaml.safe_load(open(fpath))
        pprint(meta)

        req = meta.get('requirements', [])
        fname = os.path.join(dir_path, HelperCLI.REQUIREMENTS_FILE)
        print(f"File for requirements: {fname}")
        with open(fname, "w") as fp:
            fp.write(os.linesep.join(req))

        pip_args = {
            k.replace(HelperCLI.META_PIP_KEY, ''): v
            for k, v in meta.items() if k.startswith(HelperCLI.META_PIP_KEY)
        }
        cmd_args = []
        for pip_key in pip_args:
            if not isinstance(pip_args[pip_key], (list, tuple, set)):
                pip_args[pip_key] = [pip_args[pip_key]]
            for arg in pip_args[pip_key]:
                arg = arg % RUNTIME_VERSIONS
                cmd_args.append(f"--{pip_key} {arg}")

        fname = os.path.join(dir_path, HelperCLI.PIP_ARGS_FILE)
        print(f"File for PIP arguments: {fname}")
        with open(fname, "w") as fp:
            fp.write(" ".join(cmd_args))

    @staticmethod
    def copy_notebooks(path_root: str, path_docs_ipynb: str = "docs/source/notebooks"):
        """Copy all notebooks from a folder to doc folder.
        Args:
            path_root: source path to the project root in this tutorials
            path_docs_ipynb: destination path to the notebooks location
        """
        ls_ipynb = []
        for sub in (['*.ipynb'], ['**', '*.ipynb']):
            ls_ipynb += glob.glob(os.path.join(path_root, HelperCLI.DIR_NOTEBOOKS, *sub))

        os.makedirs(path_docs_ipynb, exist_ok=True)
        ipynb_content = []
        for path_ipynb in tqdm.tqdm(ls_ipynb):
            ipynb = path_ipynb.split(os.path.sep)
            sub_ipynb = os.path.sep.join(ipynb[ipynb.index(HelperCLI.DIR_NOTEBOOKS) + 1:])
            new_ipynb = os.path.join(path_docs_ipynb, sub_ipynb)
            os.makedirs(os.path.dirname(new_ipynb), exist_ok=True)
            print(f'{path_ipynb} -> {new_ipynb}')
            shutil.copy(path_ipynb, new_ipynb)
            ipynb_content.append(os.path.join('notebooks', sub_ipynb))

    @staticmethod
    def valid_accelerator(dir_path: str):
        """Parse standard requirements from meta file
        Args:
            dir_path: path to the folder
        """
        fpath = HelperCLI._meta_file(dir_path)
        assert fpath, f"Missing Meta file in {dir_path}"
        meta = yaml.safe_load(open(fpath))
        # default is CPU runtime
        accels = [acc.lower() for acc in meta.get("accelerator", ('CPU'))]
        dev_accels = DEVICE_ACCELERATOR.split(",")
        return int(any(ac in accels for ac in dev_accels))

    @staticmethod
    def update_env_details(dir_path: str):
        """Export the actual packages used in runtime
        Args:
             dir_path: path to the folder
        """
        fpath = HelperCLI._meta_file(dir_path)
        assert fpath, f"Missing Meta file in {dir_path}"
        meta = yaml.safe_load(open(fpath))
        # default is COU runtime
        with open(PATH_REQ_DEFAULT) as fp:
            req = fp.readlines()
        req += meta.get('requirements', [])
        req = [r.strip() for r in req]

        def _parse(pkg: str, keys: str = " <=>") -> str:
            """Parsing just the package name"""
            if any(c in pkg for c in keys):
                ix = min([pkg.index(c) for c in keys if c in pkg])
                pkg = pkg[:ix]
            return pkg

        require = set([_parse(r) for r in req if r])
        env = {_parse(p): p for p in freeze.freeze()}
        meta['environment'] = [env[r] for r in require]
        meta['published'] = datetime.now().isoformat()

        fmeta = os.path.join(HelperCLI.DIR_NOTEBOOKS, dir_path) + ".yaml"
        yaml.safe_dump(meta, stream=open(fmeta, 'w'), sort_keys=False)


if __name__ == '__main__':
    fire.Fire(HelperCLI)
