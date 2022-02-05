import os

import pytest

from assistant import AssistantCLI

_PATH_DIR = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "cmd", "args",
    [
        ("list_dirs", None),
    ],
)
def test_assistant_commands(cmd, args):
    AssistantCLI().__getattribute__(cmd)(*args)
