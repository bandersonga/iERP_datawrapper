import os
import sys

from dotenv import load_dotenv
load_dotenv()

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "ierp"))
sys.path.insert(0, os.path.join(_here, "lost_decades"))

import update_ierp_datawrapper as ierp_mod
import update_ld_datawrapper   as ld_mod


def main():
    print("=== iERP chart ===")
    ierp_mod.main()
    print()
    print("=== Lost Decade chart ===")
    ld_mod.main()


if __name__ == "__main__":
    main()
