import os
import sys

import yaml
from pprint import pprint

META_FILE = ".meta.yml"
REQUIREMENTS_FILE = "requirements.txt"


def main(dir_path: str):
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
    main(*sys.argv[1:])
