"""Kinex compatibility checks: everything in rosbridge's tests that needs Kinex.

unit/ needs a Kinex checkout next to this one, ros/ a sourced simulator overlay and native/ a simulator's
native environment. The Python checks import Kinex from KINEX_PYTHON (its packages/core/assets/python) and are
skipped without it.
"""
import os
import sys

if os.environ.get("KINEX_PYTHON") and os.environ["KINEX_PYTHON"] not in sys.path:
    sys.path.insert(0, os.environ["KINEX_PYTHON"])
