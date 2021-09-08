import base64
import json
import os
import re
from datetime import datetime
from pprint import pprint
from shutil import copyfile
from textwrap import wrap
from typing import Any, Dict, Optional, Sequence
from warnings import warn

import fire
import requests
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
DEVICE_ACCELERATOR = os.environ.get(ENV_DEVICE, "cpu").lower()
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
# Open in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.png){{height="20px" width="117px"}}]({COLAB_REPO_LINK}/{REPO_NAME}/blob/{PUBLIC_BRANCH}/.notebooks/%(local_ipynb)s)
#
# Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
# | Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
# | Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

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
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool
# tools we're building.
#
# ### Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself
# and share your interests in `#general` channel
#
#
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to
# [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# GitHub Issues page and filter for "good first issue".
#
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
#
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
#
# ![Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png){height="60px" width="240px"}

"""
TEMPLATE_CARD_ITEM = """
.. customcarditem::
   :header: %(title)s
   :card_description: %(short_description)s
   :tags: %(tags)s
"""


def default_requirements(path_req: str = PATH_REQ_DEFAULT) -> list:
    with open(path_req) as fp:
        req = fp.readlines()
    req = [r[: r.index("#")] if "#" in r else r for r in req]
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
        return ver[: ver.index("+")] if "+" in ver else ver
    except ImportError:
        return ""


TORCH_VERSION = get_running_torch_version()
CUDA_VERSION = get_running_cuda_version()
RUNTIME_VERSIONS = dict(
    TORCH_VERSION_FULL=TORCH_VERSION,
    TORCH_VERSION=TORCH_VERSION[: TORCH_VERSION.index("+")] if "+" in TORCH_VERSION else TORCH_VERSION,
    TORCH_MAJOR_DOT_MINOR=".".join(TORCH_VERSION.split(".")[:2]),
    CUDA_VERSION=CUDA_VERSION,
    CUDA_MAJOR_MINOR=CUDA_VERSION.replace(".", ""),
    DEVICE=f"cu{CUDA_VERSION.replace('.', '')}" if CUDA_VERSION else "cpu",
)


class HelperCLI:

    DIR_NOTEBOOKS = ".notebooks"
    META_REQUIRED_FIELDS = ("title", "author", "license", "description")
    SKIP_DIRS = (
        ".actions",
        ".azure-pipelines",
        ".datasets",
        ".github",
        "docs",
        "_TEMP",
        "requirements",
        DIR_NOTEBOOKS,
    )
    META_FILE_REGEX = ".meta.{yaml,yml}"
    REQUIREMENTS_FILE = "requirements.txt"
    PIP_ARGS_FILE = "pip_arguments.txt"
    META_PIP_KEY = "pip__"

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
        with open(fpath) as fp:
            py_file = fp.readlines()
        fpath_meta = HelperCLI._meta_file(os.path.dirname(fpath))
        meta = yaml.safe_load(open(fpath_meta))
        meta_miss = [fl for fl in HelperCLI.META_REQUIRED_FIELDS if fl not in meta]
        if meta_miss:
            raise ValueError(f"Meta file '{fpath_meta}' is missing the following fields: {meta_miss}")
        meta.update(
            dict(local_ipynb=f"{os.path.dirname(fpath)}.ipynb"),
            generated=datetime.now().isoformat(),
        )

        meta["description"] = meta["description"].replace(os.linesep, f"{os.linesep}# ")

        header = TEMPLATE_HEADER % meta
        requires = set(default_requirements() + meta["requirements"])
        setup = TEMPLATE_SETUP % dict(requirements=" ".join([f'"{req}"' for req in requires]))
        py_file = [header + setup] + py_file + [TEMPLATE_FOOTER]

        py_file = HelperCLI._replace_images(py_file, os.path.dirname(fpath))

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
                url_path = "/".join([URL_DOWNLOAD, local_dir, p_img])
                p_local_img = os.path.join(local_dir, p_img)
                with open(p_local_img, "rb") as fp:
                    im = fp.read()
            im_base64 = base64.b64encode(im).decode("utf-8")
            _, ext = os.path.splitext(p_img)
            md = md.replace(f'src="{p_img}"', f'src="{url_path}"')
            md = md.replace(f"]({p_img})", f"](data:image/{ext[1:]};base64,{im_base64})")

        return [ln + os.linesep for ln in md.split(os.linesep)]

    @staticmethod
    def _is_ipynb_parent_dir(dir_path: str) -> bool:
        if HelperCLI._meta_file(dir_path):
            return True
        sub_dirs = [d for d in glob.glob(os.path.join(dir_path, "*")) if os.path.isdir(d)]
        return any(HelperCLI._is_ipynb_parent_dir(d) for d in sub_dirs)

    @staticmethod
    def group_folders(
        fpath_gitdiff: str,
        fpath_change_folders: str = "changed-folders.txt",
        fpath_drop_folders: str = "dropped-folders.txt",
        fpaths_actual_dirs: Sequence[str] = tuple(),
        strict: bool = True,
        root_path: str = "",
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
            root_path: path to the root tobe added for all local folder paths in files

        Example:
            >> python helpers.py group-folders ../target-diff.txt --fpaths_actual_dirs "['../dirs-main.txt', '../dirs-publication.txt']"
        """
        with open(fpath_gitdiff) as fp:
            changed = [ln.strip() for ln in fp.readlines()]
        dirs = [os.path.dirname(ln) for ln in changed]
        # not empty paths
        dirs = [ln for ln in dirs if ln]

        if fpaths_actual_dirs:
            assert isinstance(fpaths_actual_dirs, list)
            assert all(os.path.isfile(p) for p in fpaths_actual_dirs)
            dir_sets = [{ln.strip() for ln in open(fp).readlines()} for fp in fpaths_actual_dirs]
            # get only different
            dirs += list(set.union(*dir_sets) - set.intersection(*dir_sets))

        if root_path:
            dirs = [os.path.join(root_path, d) for d in dirs]
        # unique folders
        dirs = set(dirs)
        # drop folder with skip folder
        dirs = [pd for pd in dirs if not any(nd in HelperCLI.SKIP_DIRS for nd in pd.split(os.path.sep))]
        # valid folder has meta
        dirs_exist = [d for d in dirs if os.path.isdir(d)]
        dirs_invalid = [d for d in dirs_exist if not HelperCLI._meta_file(d)]
        if strict and dirs_invalid:
            msg = f"Following folders do not have valid `{HelperCLI.META_FILE_REGEX}`"
            warn(f"{msg}: \n {os.linesep.join(dirs_invalid)}")
            # check if there is other valid folder in its tree
            dirs_invalid = [pd for pd in dirs_invalid if not HelperCLI._is_ipynb_parent_dir(pd)]
            if dirs_invalid:
                raise FileNotFoundError(f"{msg} nor sub-folder: \n {os.linesep.join(dirs_invalid)}")

        dirs_change = [d for d in dirs_exist if HelperCLI._meta_file(d)]
        with open(fpath_change_folders, "w") as fp:
            fp.write(os.linesep.join(sorted(dirs_change)))

        dirs_drop = [d for d in dirs if not os.path.isdir(d)]
        with open(fpath_drop_folders, "w") as fp:
            fp.write(os.linesep.join(sorted(dirs_drop)))

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

        req = meta.get("requirements", [])
        fname = os.path.join(dir_path, HelperCLI.REQUIREMENTS_FILE)
        print(f"File for requirements: {fname}")
        with open(fname, "w") as fp:
            fp.write(os.linesep.join(req))

        pip_args = {
            k.replace(HelperCLI.META_PIP_KEY, ""): v for k, v in meta.items() if k.startswith(HelperCLI.META_PIP_KEY)
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
            meta["tags"].append(dirname)

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
        path_thumb = os.path.sep.join(path_thumb[path_thumb.index(HelperCLI.DIR_NOTEBOOKS) + 1 :])
        return path_thumb

    @staticmethod
    def copy_notebooks(
        path_root: str,
        docs_root: str = "docs/source",
        path_docs_ipynb: str = "notebooks",
        path_docs_images: str = "_static/images",
    ):
        """Copy all notebooks from a folder to doc folder.

        Args:
            path_root: source path to the project root in this tutorials
            docs_root: docs source directory
            path_docs_ipynb: destination path to the notebooks location relative to ``docs_root``
            path_docs_images: destination path to the images location relative to ``docs_root``
        """
        ls_ipynb = []
        for sub in (["*.ipynb"], ["**", "*.ipynb"]):
            ls_ipynb += glob.glob(os.path.join(path_root, HelperCLI.DIR_NOTEBOOKS, *sub))

        os.makedirs(os.path.join(docs_root, path_docs_ipynb), exist_ok=True)
        ipynb_content = []
        for path_ipynb in tqdm.tqdm(ls_ipynb):
            ipynb = path_ipynb.split(os.path.sep)
            sub_ipynb = os.path.sep.join(ipynb[ipynb.index(HelperCLI.DIR_NOTEBOOKS) + 1 :])
            new_ipynb = os.path.join(docs_root, path_docs_ipynb, sub_ipynb)
            os.makedirs(os.path.dirname(new_ipynb), exist_ok=True)

            path_meta = path_ipynb.replace(".ipynb", ".yaml")
            path_thumb = HelperCLI._resolve_path_thumb(path_ipynb, path_meta)

            if path_thumb is not None:
                new_thumb = os.path.join(docs_root, path_docs_images, path_thumb)
                old_path_thumb = os.path.join(path_root, HelperCLI.DIR_NOTEBOOKS, path_thumb)
                os.makedirs(os.path.dirname(new_thumb), exist_ok=True)
                copyfile(old_path_thumb, new_thumb)
                path_thumb = os.path.join(path_docs_images, path_thumb)

            print(f"{path_ipynb} -> {new_ipynb}")

            with open(path_ipynb) as f:
                ipynb = json.load(f)

            ipynb["cells"].append(HelperCLI._get_card_item_cell(path_ipynb, path_meta, path_thumb))

            with open(new_ipynb, "w") as f:
                json.dump(ipynb, f)

            ipynb_content.append(os.path.join("notebooks", sub_ipynb))

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
        accels = [acc.lower() for acc in meta.get("accelerator", ("CPU"))]
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
        req += meta.get("requirements", [])
        req = [r.strip() for r in req]

        def _parse(pkg: str, keys: str = " <=>") -> str:
            """Parsing just the package name."""
            if any(c in pkg for c in keys):
                ix = min(pkg.index(c) for c in keys if c in pkg)
                pkg = pkg[:ix]
            return pkg

        require = {_parse(r) for r in req if r}
        env = {_parse(p): p for p in freeze.freeze()}
        meta["environment"] = [env[r] for r in require]
        meta["published"] = datetime.now().isoformat()

        fmeta = os.path.join(HelperCLI.DIR_NOTEBOOKS, dir_path) + ".yaml"
        yaml.safe_dump(meta, stream=open(fmeta, "w"), sort_keys=False)


if __name__ == "__main__":
    fire.Fire(HelperCLI)
