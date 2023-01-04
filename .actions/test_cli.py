import os

import pytest
from assistant import AssistantCLI

_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH_TEMPLATES = os.path.join(_PATH_ROOT, "templates")
_PATH_DIR_SIMPLE = os.path.join(_PATH_TEMPLATES, "simple")
_PATH_DIR_TITANIC = os.path.join(_PATH_TEMPLATES, "titanic")


def _path_in_dir(fname: str, folder: str = _PATH_ROOT) -> str:
    return os.path.join(folder, fname)


@pytest.mark.parametrize(
    "cmd,args",
    [
        ("list_dirs", []),
        ("list_dirs", [".", ".ipynb"]),
        ("bash_render", [_PATH_DIR_SIMPLE]),
        ("bash_test", [_PATH_DIR_SIMPLE]),
        ("group_folders", [_path_in_dir("master-diff.txt"), _path_in_dir("dirs-b1.txt"), _path_in_dir("dirs-b2.txt")]),
        ("convert_ipynb", [_PATH_DIR_SIMPLE]),
        ("copy_notebooks", [_PATH_ROOT]),
        ("update_env_details", [_PATH_DIR_SIMPLE]),
    ],
)
def test_assistant_commands(cmd: str, args: list):
    AssistantCLI().__getattribute__(cmd)(*args)
