import base64
import itertools
import json
import os
import platform
import re
import warnings
from datetime import datetime
from shutil import copyfile
from textwrap import wrap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import fire
import requests
import tqdm
import yaml
from pip._internal.operations import freeze
from wcmatch import glob

_PATH_HERE = os.path.dirname(os.path.realpath(__file__))
_PATH_ROOT = os.path.dirname(_PATH_HERE)
PATH_SCRIPT_RENDER = os.path.join(_PATH_HERE, "_ipynb-render.sh")
PATH_SCRIPT_TEST = os.path.join(_PATH_HERE, "_ipynb-validate.sh")
PATH_DEV_SCRIPT = os.path.join(".actions", "assistant.py")
PATH_DEV_REQUIREMENTS = os.path.join("_requirements", "devel.txt")
# https://askubuntu.com/questions/909918/how-to-show-unzip-progress
UNZIP_PROGRESS_BAR = ' | awk \'BEGIN {ORS=" "} {if(NR%10==0)print "."}\''
REPO_NAME = "lightning-tutorials"
COLAB_REPO_LINK = "https://colab.research.google.com/github/PytorchLightning"
BRANCH_DEFAULT = "main"
BRANCH_PUBLISHED = "publication"
DIR_NOTEBOOKS = ".notebooks"
URL_PL_DOWNLOAD = f"https://github.com/Lightning-AI/{REPO_NAME}/raw/{BRANCH_DEFAULT}"
TEMPLATE_HEADER = f"""# %%%% [markdown]
#
# # %(title)s
#
# * **Author:** %(author)s
# * **License:** %(license)s
# * **Generated:** %(generated)s
#
# %(description)s
#
# ---
# Open in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.png){{height="20px" width="117px"}}]({COLAB_REPO_LINK}/{REPO_NAME}/blob/{BRANCH_PUBLISHED}/{DIR_NOTEBOOKS}/%(local_ipynb)s)
#
# Give us a ⭐ [on Github](https://www.github.com/Lightning-AI/lightning/)
# | Check out [the documentation](https://lightning.ai/docs/)
# | Join us [on Discord](https://discord.com/invite/tfXFetEZxv)

"""
TEMPLATE_SETUP = """# %%%% [markdown]
# ## Setup
# This notebook requires some packages besides pytorch-lightning.

# %%%% colab={} colab_type="code" id="LfrJLKPFyhsK"
# ! pip install --quiet %(requirements)s

"""
TEMPLATE_FOOTER = """
# %% [markdown]
# ## Congratulations - Time to Join the Community!
#
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the Lightning
# movement, you can do so in the following ways!
#
# ### Star [Lightning](https://github.com/Lightning-AI/lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool
# tools we're building.
#
# ### Join our [Discord](https://discord.com/invite/tfXFetEZxv)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself
# and share your interests in `#general` channel
#
#
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to
# [Lightning](https://github.com/Lightning-AI/lightning) or [Bolt](https://github.com/Lightning-AI/lightning-bolts)
# GitHub Issues page and filter for "good first issue".
#
# * [Lightning good first issue](https://github.com/Lightning-AI/lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/Lightning-AI/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
#
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
#
# [![Pytorch Lightning](https://raw.githubusercontent.com/Lightning-AI/lightning/master/docs/source/_static/images/logo.png){height="60px" width="240px"}](https://pytorchlightning.ai)

"""
TEMPLATE_CARD_ITEM = """
.. customcarditem::
   :header: %(title)s
   :card_description: %(short_description)s
   :tags: %(tags)s
"""


def load_requirements(folder: str = ".", fname: str = "requirements.txt", full_path: str = "") -> List[str]:
    """Load the requirements from a file.

    Args:
        folder: path to the folder with requirements
        fname: filename in the given folder
        full_path: alternative and overwrite the composition of folder + fname

    """
    if not full_path:
        full_path = os.path.join(folder, fname)
    if not os.path.isfile(full_path):
        warnings.warn(f"Missing expected requirement file '{full_path}'")
        return []
    with open(full_path) as fopen:
        req = fopen.readlines()
    req = [r[: r.index("#")] if "#" in r else r for r in req]
    req = [r.strip() for r in req]
    req = [r for r in req if r]
    return req


def get_running_cuda_version() -> str:
    """Extract the version of actual CUDA for this runtime."""
    try:
        import torch

        return torch.version.cuda or ""
    except ImportError:
        return ""


def get_running_torch_version() -> str:
    """Extract the version of actual PyTorch for this runtime."""
    try:
        import torch

        ver = torch.__version__
        return ver[: ver.index("+")] if "+" in ver else ver
    except ImportError:
        return ""


def _parse_package_name(pkg: str, keys: str = " !<=>[]@", egg_name: str = "#egg=") -> str:
    """Parsing just the package name.

    Args:
        pkg: full package info including version coming from `pip freeze
        keys: char separators for parsing package name and version or other info
        egg_name: eventual passing egg names once isnralled from URL

    Examples:
        >>> _parse_package_name("torch==2.0.1+cu118")
        'torch'
        >>> _parse_package_name("torch-cluster==1.6.3+pt20cu118")
        'torch-cluster'
        >>> _parse_package_name("torch_geometric==2.4.0")
        'torch_geometric'

    """
    if egg_name in pkg:
        pkg = pkg[pkg.index(egg_name) + len(egg_name) :]
    if any(c in pkg for c in keys):
        ix = min(pkg.index(c) for c in keys if c in pkg)
        pkg = pkg[:ix]
    return pkg


_TORCH_VERSION = get_running_torch_version()
_CUDA_VERSION = get_running_cuda_version()
_RUNTIME_VERSIONS = dict(
    TORCH_VERSION_FULL=_TORCH_VERSION,
    TORCH_VERSION=_TORCH_VERSION[: _TORCH_VERSION.index("+")] if "+" in _TORCH_VERSION else _TORCH_VERSION,
    TORCH_MAJOR_DOT_MINOR=".".join(_TORCH_VERSION.split(".")[:2]),
    CUDA_VERSION=_CUDA_VERSION,
    CUDA_MAJOR_MINOR=_CUDA_VERSION.replace(".", ""),
    DEVICE=f"cu{_CUDA_VERSION.replace('.', '')}" if _CUDA_VERSION else "cpu",
)


class AssistantCLI:
    """Collection of handy CLI commands."""

    _LOCAL_ACCELERATOR = "cpu,gpu" if get_running_cuda_version() else "cpu"
    DEVICE_ACCELERATOR = os.environ.get("ACCELERATOR", _LOCAL_ACCELERATOR).lower()
    DATASETS_FOLDER = os.environ.get("PATH_DATASETS", "_datasets")
    DRY_RUN = bool(int(os.environ.get("DRY_RUN", 0)))
    _META_REQUIRED_FIELDS = ("title", "author", "license", "description")
    _META_FILE_REGEX = ".meta.{yaml,yml}"
    _META_PIP_KEY = "pip__"
    _META_ACCEL_DEFAULT = _LOCAL_ACCELERATOR.split(",")

    # Map directory names to tag names. Note that dashes will be replaced with spaces in rendered tags in the docs.
    _DIR_TO_TAG = {
        "course_UvA-DL": "UvA-DL-Course",
        "lightning_examples": "Lightning-Examples",
        "flash_tutorials": "Kaggle",
    }
    _BASH_SCRIPT_BASE = ("#!/bin/bash", "set -ex", "")
    _EXT_ARCHIVE_ZIP = (".zip",)
    _EXT_ARCHIVE_TAR = (".tar", ".gz")
    _EXT_ARCHIVE = _EXT_ARCHIVE_ZIP + _EXT_ARCHIVE_TAR
    _AZURE_POOL = "lit-rtx-3090"
    _AZURE_DOCKER = "pytorchlightning/tutorials:ubuntu22.04-cuda12.1.1-py3.10"

    @staticmethod
    def _find_meta(folder: str) -> str:
        """Search for a meta file in given folder and return its path.

        Args:
            folder: path to the folder with python script, meta and artefacts

        """
        files = glob.glob(os.path.join(folder, AssistantCLI._META_FILE_REGEX), flags=glob.BRACE)
        if len(files) == 1:
            return files[0]
        return ""

    @staticmethod
    def _load_meta(folder: str, strict: bool = False) -> Optional[dict]:
        """Loading meta-data for a particular notebook with given folder path.

        Args:
            folder: path to the folder with python script, meta and artefacts
            strict: raise error if meta is missing required fields

        """
        fpath = AssistantCLI._find_meta(folder)
        assert fpath, f"Missing meta file in folder: {folder}"
        meta = yaml.safe_load(open(fpath))

        if strict:
            meta_miss = [fl for fl in AssistantCLI._META_REQUIRED_FIELDS if fl not in meta]
            if meta_miss:
                raise ValueError(f"Meta file '{fpath}' is missing the following fields: {meta_miss}")
        return meta

    @staticmethod
    def _valid_conf_folder(folder: str) -> Tuple[str, str]:
        """Validate notebook folder if it has required meta file and optional thumb.

        Args:
            folder: path to the folder with python script, meta and artefacts

        """
        meta_files = [os.path.join(folder, f".meta.{ext}") for ext in ("yml", "yaml")]
        meta_files = [pf for pf in meta_files if os.path.isfile(pf)]
        if len(meta_files) != 1:
            raise FileExistsError(f"found {len(meta_files)} meta (yaml|yml) files in folder: {folder}")
        thumb_files = glob.glob(os.path.join(folder, ".thumb.*"))
        thumb_names = list(map(os.path.basename, thumb_files))
        if len(thumb_files) > 1:
            raise FileExistsError(f"Too many thumb files ({thumb_names}) found in folder: {folder}")
        thumb = thumb_files[0] if thumb_files else ""
        return meta_files[0], thumb

    @staticmethod
    def _valid_folder(folder: str, ext: str) -> Tuple[str, str, str]:
        """Validate notebook folder if it has required meta file, python script or ipython notebook (depending on the
        stage) and optional thumb.

        Args:
            folder: path to the folder with python script, meta and artefacts
            ext: extension determining the stage - ".py" for python script nad ".ipynb" for notebook

        """
        files = glob.glob(os.path.join(folder, f"*{ext}"))
        if len(files) != 1:
            names = list(map(os.path.basename, files))
            raise FileNotFoundError(f"Missing required '{ext}' file in folder: {folder} among {names}")
        meta_file, thumb_file = AssistantCLI._valid_conf_folder(folder)
        return files[0], meta_file, thumb_file

    @staticmethod
    def _valid_accelerator(folder: str) -> bool:
        """Parse standard requirements from meta file.

        Args:
            folder: path to the folder with python script, meta and artefacts

        """
        meta = AssistantCLI._load_meta(folder)
        meta_accels = [acc.lower() for acc in meta.get("accelerator", AssistantCLI._META_ACCEL_DEFAULT)]
        device_accels = AssistantCLI.DEVICE_ACCELERATOR.lower().split(",")
        return any(ac in meta_accels for ac in device_accels)

    @staticmethod
    def _parse_requirements(folder: str, formatted: bool = True) -> Union[Tuple[str, str], Tuple[list, list]]:
        """Parse standard requirements from meta file.

        Args:
            folder: path to the folder with python script, meta and artefacts
            formatted: format it into two strings

        """
        meta = AssistantCLI._load_meta(folder)
        requires = set(load_requirements(folder) + load_requirements(os.path.dirname(folder)))

        meta_pip_args = {
            k.replace(AssistantCLI._META_PIP_KEY, ""): v
            for k, v in meta.items()
            if k.startswith(AssistantCLI._META_PIP_KEY)
        }
        pip_args = [f'--extra-index-url="https://download.pytorch.org/whl/{_RUNTIME_VERSIONS.get("DEVICE")}"']
        for pip_key in meta_pip_args:
            if not isinstance(meta_pip_args[pip_key], (list, tuple, set)):
                meta_pip_args[pip_key] = [meta_pip_args[pip_key]]
            for arg in meta_pip_args[pip_key]:
                arg = arg % _RUNTIME_VERSIONS
                pip_args.append(f"--{pip_key} {arg}")
        if formatted:
            return " ".join([f'"{req}"' for req in requires]), " ".join(pip_args)
        return list(requires), pip_args

    @staticmethod
    def pip_install(folder: str) -> str:
        """Print all notebook requirements to be pre-installed in format of requirements file.

        Args:
            folder: path to the folder with python script, meta and artefacts

        """
        req, args = AssistantCLI._parse_requirements(folder, formatted=False)
        return os.linesep.join(req) + os.linesep + os.linesep.join(args)

    @staticmethod
    def _bash_download_data(folder: str) -> List[str]:
        """Generate sequence of commands for optional downloading dataset specified in the meta file.

        Args:
            folder: path to the folder with python script, meta and artefacts

        """
        meta = AssistantCLI._load_meta(folder)
        datasets = meta.get("datasets", {})
        data_kaggle = datasets.get("kaggle", [])
        cmd = [f"python -m kaggle competitions download -c {name}" for name in data_kaggle]
        files = [f"{name}.zip" for name in data_kaggle]
        data_web = datasets.get("web", [])
        cmd += [f"wget {web} --progress=bar:force:noscroll --tries=3" for web in data_web]
        files += [os.path.basename(web) for web in data_web]
        for fn in files:
            name, ext = os.path.splitext(fn)
            if ext not in AssistantCLI._EXT_ARCHIVE:
                continue
            if ext in AssistantCLI._EXT_ARCHIVE_ZIP:
                cmd += [f"unzip -o {fn} -d {AssistantCLI.DATASETS_FOLDER}/{name} {UNZIP_PROGRESS_BAR}"]
            else:
                cmd += [f"tar -zxvf {fn} --overwrite"]
            cmd += [f"rm {fn}"]
        cmd += [f"tree -L 2 {AssistantCLI.DATASETS_FOLDER}"]
        return cmd

    @staticmethod
    def bash_render(folder: str, output_file: str = PATH_SCRIPT_RENDER) -> Optional[str]:
        """Prepare bash script for running rendering of a particular notebook.

        Args:
            folder: name/path to a folder with notebook files
            output_file: if defined, stream the commands to the file

        Returns:
            string with nash script content

        """
        cmd = list(AssistantCLI._BASH_SCRIPT_BASE) + [f"# Rendering: {folder}"]
        if not AssistantCLI.DRY_RUN:
            cmd += AssistantCLI._bash_download_data(folder)
        ipynb_file, meta_file, thumb_file = AssistantCLI._valid_folder(folder, ext=".ipynb")
        pub_ipynb = os.path.join(DIR_NOTEBOOKS, f"{folder}.ipynb")
        pub_meta = pub_ipynb.replace(".ipynb", ".yaml")
        pub_dir = os.path.dirname(pub_ipynb)
        thumb_ext = os.path.splitext(thumb_file)[-1] if thumb_file else "."
        pub_thumb = os.path.join(DIR_NOTEBOOKS, f"{folder}{thumb_ext}") if thumb_file else ""
        cmd.append(f"mkdir -p {pub_dir}")
        if AssistantCLI.DRY_RUN:
            # dry run does not execute the notebooks just takes them as they are
            cmd.append(f"cp {ipynb_file} {pub_ipynb}")
            # copy and add meta config
            cmd += [
                f"cp {meta_file} {pub_meta}",
                'echo "#====== START OF YAML FILE ======#"',
                f"cat {pub_meta}",
                'echo "#======= END OF YAML FILE =======#"',
                f"git add {pub_meta}",
            ]
        else:
            pip_req, pip_args = AssistantCLI._parse_requirements(folder)
            cmd += [f"pip install {pip_req} --quiet {pip_args}", "pip list"]
            cmd.append(f"# available: {AssistantCLI.DEVICE_ACCELERATOR}\n")
            if AssistantCLI._valid_accelerator(folder):
                cmd.append(f"python -m papermill {ipynb_file} {pub_ipynb} --kernel python")
            else:
                warn("Invalid notebook's accelerator for this device. So no outputs will be generated.", RuntimeWarning)
                cmd.append(f"cp {ipynb_file} {pub_ipynb}")
            # Export the actual packages used in runtime
            cmd.append(f"meta_file=$(python .actions/assistant.py update-env-details {folder})")
            # copy and add to version the enriched meta config
            cmd += [
                "echo $meta_file",
                'echo "#====== START OF YAML FILE ======#"',
                "cat $meta_file",
                'echo "#======= END OF YAML FILE =======#"',
                "git add $meta_file",
            ]
        # if thumb image is linked to the notebook, copy and version it too
        if thumb_file:
            cmd += [f"cp {thumb_file} {pub_thumb}", f"git add {pub_thumb}"]
        # add the generated notebook to version
        cmd.append(f"git add {pub_ipynb}")
        if not output_file:
            return os.linesep.join(cmd)
        with open(output_file, "w") as fopen:
            fopen.write(os.linesep.join(cmd))

    @staticmethod
    def bash_validate(folder: str, output_file: str = PATH_SCRIPT_TEST, virtualenv: bool = False) -> Optional[str]:
        """Prepare bash script for running tests of a particular notebook.

        Args:
            folder: name/path to a folder with notebook files
            output_file: if defined, stream the commands to the file

        Returns:
            string with nash script content

        """
        cmd = list(AssistantCLI._BASH_SCRIPT_BASE) + [f"# Testing: {folder}"]
        cmd += AssistantCLI._bash_download_data(folder)
        ipynb_file, meta_file, _ = AssistantCLI._valid_folder(folder, ext=".ipynb")

        # prepare isolated environment with inheriting the global packages
        path_venv = os.path.join(folder, "venv")
        if virtualenv:
            cmd += [
                f"python -m virtualenv --system-site-packages {path_venv}",
                f"source {os.path.join(path_venv, 'bin', 'activate')}",
                "pip --version",
            ]

        cmd.append(f"# available: {AssistantCLI.DEVICE_ACCELERATOR}")
        if AssistantCLI._valid_accelerator(folder):
            # and install specific packages
            pip_req, pip_args = AssistantCLI._parse_requirements(folder)
            cmd += [f"pip install {pip_req} --quiet {pip_args}", "pip list"]
            # Export the actual packages used in runtime
            cmd.append(f"meta_file=$(python .actions/assistant.py update-env-details {folder} --base_path .)")
            # show created meta config
            cmd += [
                "echo $meta_file",
                'echo "#====== START OF YAML FILE ======#"',
                "cat $meta_file",
                'echo "#======= END OF YAML FILE =======#"',
            ]
            # use standard jupyter's executable via CMD
            cmd.append(f"jupyter execute {ipynb_file} --inplace")
        else:
            pub_ipynb = os.path.join(DIR_NOTEBOOKS, f"{folder}.ipynb")
            pub_meta = pub_ipynb.replace(".ipynb", ".yaml")
            # copy and add meta config
            cmd += [
                f"mkdir -p {os.path.dirname(pub_meta)}",
                f"cp {meta_file} {pub_meta}",
                'echo "#====== START OF YAML FILE ======#"',
                f"cat {pub_meta}",
                'echo "#======= END OF YAML FILE =======#"',
                f"git add {pub_meta}",
            ]
            warn("Invalid notebook's accelerator for this device. So no tests will be run!!!", RuntimeWarning)
        # deactivate and clean local environment
        if virtualenv:
            cmd += ["deactivate", f"rm -rf {os.path.join(folder, 'venv')}"]
        if not output_file:
            return os.linesep.join(cmd)
        with open(output_file, "w") as fopen:
            fopen.write(os.linesep.join(cmd))

    @staticmethod
    def convert_ipynb(folder: str) -> None:
        """Add template header and footer to the python base script.

        Args:
            folder: folder with python script

        """
        fpath, _, _ = AssistantCLI._valid_folder(folder, ext=".py")
        with open(fpath) as fopen:
            py_script = fopen.readlines()

        meta = AssistantCLI._load_meta(folder, strict=True)
        meta.update(
            dict(local_ipynb=f"{folder}.ipynb"),
            generated=datetime.now().isoformat(),
        )
        meta["description"] = meta["description"].replace(os.linesep, f"{os.linesep}# ")

        header = TEMPLATE_HEADER % meta
        # load local and parent requirements
        requires = set(load_requirements(folder) + load_requirements(os.path.dirname(folder)))
        setup = TEMPLATE_SETUP % dict(requirements=" ".join([f'"{req}"' for req in requires]))
        py_script = [header + setup] + py_script + [TEMPLATE_FOOTER]

        py_script = AssistantCLI._replace_images(py_script, folder)

        with open(fpath, "w") as fopen:
            fopen.writelines(py_script)

        os.system(f'python -m jupytext --set-formats "ipynb,py:percent" {fpath}')

    @staticmethod
    def _replace_images(lines: list, local_dir: str) -> list:
        """Update images by URL to GitHub raw source.

        Args:
            lines: string lines from python script
            local_dir: relative path to the folder with script

        """
        md = os.linesep.join([ln.rstrip() for ln in lines])
        p_imgs = []
        # todo: add a rule to replace this paths only i md sections
        # because * is a greedy quantifier, trying to match as much as it can. Make it *?
        p_imgs += re.findall(r"src=\"(.*?)\"", md)
        p_imgs += re.findall(r"!\[.*?\]\((.*?)\)", md)

        # update all images
        for p_img in set(p_imgs):
            if p_img.startswith("http://") or p_img.startswith("https://"):
                url_path = p_img
                im = requests.get(p_img, stream=True).raw.read()
            else:
                url_path = "/".join([URL_PL_DOWNLOAD, local_dir, p_img])
                p_local_img = os.path.join(local_dir, p_img)
                with open(p_local_img, "rb") as fopen:
                    im = fopen.read()
            im_base64 = base64.b64encode(im).decode("utf-8")
            _, ext = os.path.splitext(p_img)
            md = md.replace(f'src="{p_img}"', f'src="{url_path}"')
            md = md.replace(f"]({p_img})", f"](data:image/{ext[1:]};base64,{im_base64})")

        return [ln + os.linesep for ln in md.split(os.linesep)]

    @staticmethod
    def _is_ipynb_parent_dir(dir_path: str) -> bool:
        """Determine in recursive fashion of a folder is valid notebook file or any of sub-folders is."""
        if dir_path.startswith("_"):
            return False
        if AssistantCLI._find_meta(dir_path):
            return True
        sub_dirs = [d for d in glob.glob(os.path.join(dir_path, "*")) if os.path.isdir(d)]
        return any(AssistantCLI._is_ipynb_parent_dir(d) for d in sub_dirs)

    @staticmethod
    def group_folders(
        fpath_gitdiff: str,
        fpath_change_folders: str = "changed-folders.txt",
        fpath_drop_folders: str = "dropped-folders.txt",
        fpath_actual_dirs: Sequence[str] = tuple(),
        strict: bool = True,
        root_path: str = "",
    ) -> None:
        """Parsing the raw git diff and group changes by folders.

        Args:
            fpath_gitdiff: raw git changes

                Generate the git change list:
                > head=$(git rev-parse origin/main)
                > git diff --name-only $head --output=master-diff.txt

            fpath_change_folders: output file with changed folders
            fpath_drop_folders: output file with deleted folders
            fpath_actual_dirs: files with listed all folder in particular stat
            strict: raise error if some folder outside skipped does not have valid meta file
            root_path: path to the root to be added for all local folder paths in files

        Example:
            $ python assistant.py group-folders ../target-diff.txt \
                --fpath_actual_dirs "['../dirs-main.txt', '../dirs-publication.txt']"

        """
        dirs = []
        # replace here notation with path, so later it does not fall to ignore
        if root_path == ".":
            root_path = os.getcwd()
        # loading the generated changes with git diff
        with open(fpath_gitdiff) as fopen:
            changed = [ln.strip() for ln in fopen.readlines()]
        # in case of ay key component changed, later rebuild all notebooks
        rebuild_all = any(
            p_key in p_changed for p_key in (PATH_DEV_SCRIPT, PATH_DEV_REQUIREMENTS) for p_changed in changed
        )
        dirs_changed = [os.path.dirname(ln) for ln in changed]
        if rebuild_all:
            dirs_changed += [p for p in os.listdir(_PATH_ROOT) if os.path.isdir(p) and p[0] not in (".", "_")]
        # append a path to root in case you call this from other path then root
        if root_path:
            dirs_changed = [os.path.join(root_path, d) for d in dirs_changed]
        # append all subfolders in case of parent requirements has been changed all related notebooks shall be updated
        for dir in dirs_changed:
            # in case that the diff item comes from removed folder
            if not os.path.isdir(dir):
                dirs += [dir]
                continue
            # list folder and skip all internal files, starting with . or _
            sub_dirs = [os.path.join(dir, it) for it in os.listdir(dir) if it[0] not in (".", "_")]
            # filter only folders
            sub_dirs = [it for it in sub_dirs if os.path.isdir(it)]
            # if the dir has sub-folder append then otherwise append the dir itself
            dirs += sub_dirs if sub_dirs else [dir]

        if fpath_actual_dirs:
            assert isinstance(fpath_actual_dirs, list)
            assert all(os.path.isfile(p) for p in fpath_actual_dirs)
            dir_sets = [{ln.strip() for ln in open(fp).readlines()} for fp in fpath_actual_dirs]
            # get only different
            dirs_diff = list(set.union(*dir_sets) - set.intersection(*dir_sets))
            # append a path to root in case you call this from other path then root
            if root_path:
                dirs_diff = [os.path.join(root_path, d) for d in dirs_diff]
            dirs += dirs_diff

        # not empty paths
        dirs = [ln for ln in dirs if ln]
        # drop folder  that start with . or _ as they are meant to be internal use only
        # when walk the folder skip empty names which is artifact of absolute path starting with /
        dirs = [
            pdir for pdir in dirs if not any(folder[0] in (".", "_") for folder in pdir.split(os.path.sep) if folder)
        ]
        # unique folders only, drop duplicates
        dirs = set(dirs)
        # valid folder has meta
        dirs_exist = [d for d in dirs if os.path.isdir(d)]
        dirs_invalid = [d for d in dirs_exist if not AssistantCLI._find_meta(d)]
        if strict and dirs_invalid:
            msg = f"Following folders do not have valid `{AssistantCLI._META_FILE_REGEX}`"
            warn(f"{msg}: \n {os.linesep.join(dirs_invalid)}")
            # check if there is other valid folder in its tree
            dirs_invalid = [pd for pd in dirs_invalid if not AssistantCLI._is_ipynb_parent_dir(pd)]
            if dirs_invalid:
                raise FileNotFoundError(f"{msg} nor sub-folder: \n {os.linesep.join(dirs_invalid)}")

        dirs_with_change = [d for d in dirs_exist if AssistantCLI._find_meta(d)]
        with open(fpath_change_folders, "w") as fopen:
            fopen.write(os.linesep.join(sorted(dirs_with_change)))

        dirs_were_dropped = [d for d in dirs if not os.path.isdir(d)]
        with open(fpath_drop_folders, "w") as fopen:
            fopen.write(os.linesep.join(sorted(dirs_were_dropped)))

    @staticmethod
    def generate_matrix(fpath_change_folders: str, json_indent: Optional[int] = None) -> str:
        """Generate Azure matrix with leaf for each changed notebook.

        Args:
            fpath_change_folders: output of previous ``group_folders``
            json_indent: makes the json more readable, recommendation is 4

        """
        with open(fpath_change_folders) as fopen:
            folders = [ln.strip() for ln in fopen.readlines()]
        # set default so the matrix has at least one runner
        if not folders:
            return ""
        mtx = {}
        for ln in folders:
            mtx[ln] = {
                "notebook": ln,
                # TODO: allow defining some custom pools with different devices
                "agent-pool": AssistantCLI._AZURE_POOL,
                # TODO: allow defining some custom images with with python or PT
                "docker-image": AssistantCLI._AZURE_DOCKER,
            }
        return json.dumps(mtx, indent=json_indent)

    @staticmethod
    def _get_card_item_cell(path_ipynb: str, path_meta: str, path_thumb: Optional[str]) -> Dict[str, Any]:
        """Build the card item cell for the given notebook path."""
        meta = yaml.safe_load(open(path_meta))

        # Clamp description length
        wrapped_description = wrap(
            meta.get("short_description", meta["description"]).strip().replace(os.linesep, " "), 175
        )
        suffix = "..." if len(wrapped_description) > 1 else ""
        meta["short_description"] = wrapped_description[0] + suffix

        # Resolve some default tags based on accelerators and directory name
        meta["tags"] = meta.get("tags", [])

        accelerators = meta.get("accelerator", ("CPU",))
        if ("GPU" in accelerators) or ("TPU" in accelerators):
            meta["tags"].append("GPU/TPU")

        dirname = os.path.basename(os.path.dirname(path_ipynb))
        if dirname != ".notebooks":
            meta["tags"].append(AssistantCLI._DIR_TO_TAG.get(dirname, dirname))

        meta["tags"] = [tag.replace(" ", "-") for tag in meta["tags"]]
        meta["tags"] = ",".join(meta["tags"])

        # Build the notebook cell
        rst_cell = TEMPLATE_CARD_ITEM % meta

        # Split lines
        rst_cell_lines = rst_cell.strip().splitlines(True)

        if path_thumb is not None:
            rst_cell_lines[-1] += "\n"
            rst_cell_lines.append(f"   :image: {path_thumb}")

        return {
            "cell_type": "raw",
            "metadata": {"raw_mimetype": "text/restructuredtext"},
            "source": rst_cell_lines,
        }

    @staticmethod
    def _resolve_path_thumb(path_ipynb: str, path_meta: str) -> Optional[str]:
        """Find the thumbnail (assumes thumbnail to be any file that isn't metadata or notebook)."""
        paths = list(set(glob.glob(path_ipynb.replace(".ipynb", ".*"))) - {path_ipynb, path_meta})
        if len(paths) == 0:
            return None
        assert len(paths) == 1, f"Found multiple possible thumbnail paths for notebook: {path_ipynb}."
        path_thumb = paths[0]
        path_thumb = path_thumb.split(os.path.sep)
        path_thumb = os.path.sep.join(path_thumb[path_thumb.index(DIR_NOTEBOOKS) + 1 :])
        return path_thumb

    @staticmethod
    def copy_notebooks(
        path_root: str,
        docs_root: str = "_docs/source",
        path_docs_ipynb: str = "notebooks",
        path_docs_images: str = "_static/images",
        patterns: Sequence[str] = (".", "**"),
        ignore: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> None:
        """Copy all notebooks from a folder to doc folder.

        Args:
            path_root: source path to the project root in these tutorials
            docs_root: docs source directory
            path_docs_ipynb: destination path to the notebooks' location relative to ``docs_root``
            path_docs_images: destination path to the images' location relative to ``docs_root``
            patterns: patterns to use when glob-ing notebooks
            ignore: ignore some specific notebooks even when the given string is in path
            strict: raise exception if copy fails

        """
        os.makedirs(os.path.join(docs_root, path_docs_ipynb), exist_ok=True)
        all_ipynb = [
            os.path.realpath(ipynb)
            for pattern in patterns
            for ipynb in glob.glob(os.path.join(path_root, DIR_NOTEBOOKS, pattern, "*.ipynb"))
        ]
        print(f"Copy following notebooks to docs folder: {all_ipynb}")
        if ignore and not isinstance(ignore, (list, set, tuple)):
            ignore = [ignore]
        elif not ignore:
            ignore = []

        ipynb_content = []
        for path_ipynb in tqdm.tqdm(all_ipynb):
            if any(skip in path_ipynb for skip in ignore):
                print(f"ignore/skip copy: {path_ipynb}")
                continue
            try:
                path_ipynb_in_dir = AssistantCLI._copy_notebook(
                    path_ipynb,
                    path_root=path_root,
                    docs_root=docs_root,
                    path_docs_ipynb=path_docs_ipynb,
                    path_docs_images=path_docs_images,
                )
            except Exception as ex:
                msg = f"Failed to copy notebook: {path_ipynb}\n{ex}"
                if not strict:
                    warnings.warn(msg, ResourceWarning)
                    continue
                raise FileNotFoundError(msg)
            ipynb_content.append(os.path.join(path_docs_ipynb, path_ipynb_in_dir))

    @staticmethod
    def _copy_notebook(
        path_ipynb: str,
        path_root: str,
        docs_root: str,
        path_docs_ipynb: str,
        path_docs_images: str,
        json_indent: Optional[int] = None,
    ) -> str:
        """Copy particular notebook."""
        ipynb = path_ipynb.split(os.path.sep)
        path_ipynb_in_dir = os.path.sep.join(ipynb[ipynb.index(DIR_NOTEBOOKS) + 1 :])
        new_ipynb = os.path.join(docs_root, path_docs_ipynb, path_ipynb_in_dir)
        os.makedirs(os.path.dirname(new_ipynb), exist_ok=True)

        path_meta = path_ipynb.replace(".ipynb", ".yaml")
        path_thumb = AssistantCLI._resolve_path_thumb(path_ipynb, path_meta)

        if path_thumb is not None:
            new_thumb = os.path.join(docs_root, path_docs_images, path_thumb)
            old_path_thumb = os.path.join(path_root, DIR_NOTEBOOKS, path_thumb)
            os.makedirs(os.path.dirname(new_thumb), exist_ok=True)
            copyfile(old_path_thumb, new_thumb)
            path_thumb = os.path.join(path_docs_images, path_thumb)

        print(f"{path_ipynb} -> {new_ipynb}")
        with open(path_ipynb) as fopen:
            ipynb = json.load(fopen)
        ipynb["cells"].append(AssistantCLI._get_card_item_cell(path_ipynb, path_meta, path_thumb))
        with open(new_ipynb, "w") as fopen:
            json.dump(ipynb, fopen, indent=json_indent)
        return path_ipynb_in_dir

    @staticmethod
    def update_env_details(folder: str, base_path: str = DIR_NOTEBOOKS) -> str:
        """Export the actual packages used in runtime.

        Args:
             folder: path to the folder
             base_path: base path with notebooks

        Returns:
            path the updated YAML file

        """
        meta = AssistantCLI._load_meta(folder)
        meta.update(
            {
                "OS": platform.system(),
                "python": platform.python_version(),
            }
        )
        # load local and parent requirements
        requires = set(load_requirements(folder) + load_requirements(os.path.dirname(folder)))

        requires = {_parse_package_name(r) for r in requires if r}
        # duplicate package name/key for cases with -/_ separator
        env = {
            **{_parse_package_name(p).replace("-", "_"): p for p in freeze.freeze()},
            **{_parse_package_name(p).replace("_", "-"): p for p in freeze.freeze()},
        }
        # for debugging reasons print the env and requested packages
        try:
            meta["environment"] = [env[r] for r in requires]
        except KeyError:
            raise KeyError(f"Missing matching requirements: {requires}\n within environment: {env}")
        meta["published"] = datetime.now().isoformat()

        fmeta = os.path.join(base_path, folder) + ".yaml"
        yaml.safe_dump(meta, stream=open(fmeta, "w"), sort_keys=False)
        return fmeta

    @staticmethod
    def list_dirs(folder: str = "", include_file_ext: str = "") -> str:
        """List all sub-folders in a given tree including any ipynb."""
        dirs = glob.glob(os.path.join(folder, "*" + include_file_ext))
        dirs += glob.glob(os.path.join(folder, "**", "*" + include_file_ext))
        if include_file_ext:
            _ignore_base_dir = lambda p: os.path.sep.join(p.split(os.path.sep)[1:])  # noqa: E731
            # Take the notebook as a folder (notebook are on the same level as the raw tutorial file mix)
            dirs = [os.path.splitext(_ignore_base_dir(p))[0] for p in dirs]
        else:
            dirs = [p for p in dirs if os.path.isdir(p)]
        return os.linesep.join(sorted(dirs))

    @staticmethod
    def aggregate_requirements(req_pattern: str, result_file: str) -> None:
        """Load all requirements from given path pattern and dump them to a single file.

        Args:
            req_pattern: search pattern such as '*/requirements.txt'
            result_file: path to save aggregated requirements

        """
        ls_requirements = glob.glob(req_pattern)
        assert len(ls_requirements) > 0, "no requirements found"
        # load and append all particular requirements
        reqs = [load_requirements(full_path=fp) for fp in ls_requirements]
        # try to filer only unique
        reqs = set([r.replace(" ", "") for r in itertools.chain(*reqs)])
        # dump result to a file
        with open(result_file, "w") as fopen:
            fopen.writelines(os.linesep.join(reqs))


if __name__ == "__main__":
    fire.Fire(AssistantCLI)
