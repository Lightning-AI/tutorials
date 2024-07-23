import os

import pytest
from assistant import AssistantCLI

_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH_TEMPLATES = os.path.join(_PATH_ROOT, "templates")
_PATH_DIR_SIMPLE = os.path.join(_PATH_TEMPLATES, "simple")
# _PATH_DIR_TITANIC = os.path.join(_PATH_TEMPLATES, "titanic")


def _path_in_dir(fname: str, folder: str = _PATH_ROOT) -> str:
    return os.path.join(folder, fname)


@pytest.mark.parametrize(
    "cmd,kwargs",
    [
        ("list_dirs", {}),
        ("list_dirs", dict(folder=".", include_file_ext=".ipynb")),
        ("bash_render", dict(folder=_PATH_DIR_SIMPLE)),
        ("bash_validate", dict(folder=_PATH_DIR_SIMPLE)),
        (
            "group_folders",
            dict(
                fpath_gitdiff=_path_in_dir("master-diff.txt"),
                fpath_change_folders=_path_in_dir("dirs-b1.txt"),
                fpath_drop_folders=_path_in_dir("dirs-b2.txt"),
                root_path=_PATH_ROOT,
            ),
        ),
        ("convert_ipynb", dict(folder=_PATH_DIR_SIMPLE)),
        ("copy_notebooks", dict(path_root=_PATH_ROOT)),
        ("update_env_details", dict(folder=_PATH_DIR_SIMPLE)),
    ],
)
def test_assistant_commands(cmd: str, kwargs: dict):
    AssistantCLI().__getattribute__(cmd)(**kwargs)
