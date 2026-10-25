import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for package in ("src", "ros2/rosbridge_runtime"):
    sys.path.insert(0, str(ROOT / package))
for manifest in (ROOT / "simulators").glob("*/simulator.toml"):
    sys.path.insert(0, str(manifest.parent))
# Kinex-backed checks import Kinex Python from KINEX_PYTHON (packages/core/assets/python).
if os.environ.get("KINEX_PYTHON"):
    sys.path.insert(0, os.environ["KINEX_PYTHON"])
