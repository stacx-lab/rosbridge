"""The imported G1 (g1_model.py) stored as a binary USD crate.

The import writes the robot as 281 MB of text USD. Every boot parses it twice (OmniGibson's
articulation-root pass opens it, exports a temporary copy in the same format and the stage then
opens that copy), about 4.5 s per parse; the same layer as a crate opens in 0.1 s. Run as a
module inside the native environment, this copies the model directory, writes its USD as a crate
under the .usd extension (USD detects the format; OmniGibson's loader and Isaac's references take
.usd), checks that the crate holds exactly the layer of the text, and points the robot definition
at it. Nothing else in the model changes.

    python -m rosbridge_behavior.g1_crate TEXT_MODEL_DIR OUTPUT_MODEL_DIR
"""
import os
from pathlib import Path
import shutil
import sys
import traceback

import yaml

from .g1_model import MODEL

TEXT = f"usd/{MODEL}.usda"
USD = f"usd/{MODEL}.usd"


def definition(text_definition):
    """The robot definition of the crate model: the text model's, naming the crate as its USD."""
    if "usd_path" in text_definition:
        raise ValueError("the text model's definition already names its USD")
    return {"usd_path": f"models/{MODEL}/{USD}", **text_definition}


def convert(text_model, output):
    """Copy text_model to output with its USD as a crate; the definition names the crate."""
    from pxr import Sdf

    text_model, output = Path(text_model), Path(output)
    text = text_model / TEXT
    shutil.copytree(text_model, output, ignore=lambda directory, names: [
        name for name in names if Path(directory, name) == text])
    layer = Sdf.Layer.FindOrOpen(str(text))
    if layer is None or not layer.Export(str(output / USD), args={"format": "usdc"}):
        raise RuntimeError(f"could not write {output / USD}")
    if Sdf.Layer.OpenAsAnonymous(str(output / USD)).ExportToString() != layer.ExportToString():
        raise RuntimeError(f"{output / USD} does not hold the layer of {text}")
    text_definition = yaml.safe_load((text_model / f"{MODEL}.yaml").read_text())
    (output / f"{MODEL}.yaml").write_text(yaml.safe_dump(definition(text_definition), sort_keys=False))
    return output


def main():
    text_model, output = sys.argv[1:3]
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp  # makes pxr importable

    app = SimulationApp({"headless": True})
    try:
        convert(text_model, output)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # closing Kit could end the process with success
    app.close()


if __name__ == "__main__":
    main()
