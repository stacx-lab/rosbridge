"""HandsOn's archived materials and the Isaac Sim MDL modules preparation links beside them, without Isaac."""
import hashlib
from pathlib import Path
import re

import pytest

from rosbridge_handson.assets import (ARCHIVED_OMNIGLASS, MDL_MODULES, SURFACES, check_mdl_modules,
                                      link_mdl_modules)

RUNTIME = Path(__file__).resolve().parents[4] / ".runtime/handson"


def test_preparation_links_each_module_once_and_the_check_requires_them(tmp_path):
    base = tmp_path / "isaacsim/kit/mdl/core/Base"
    base.mkdir(parents=True)
    for name in MDL_MODULES:
        (base / name).write_text(name)
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="does not link"):
        check_mdl_modules(runtime, base)
    link_mdl_modules(runtime, base)
    link_mdl_modules(runtime, base)  # every task's preparation repeats it
    check_mdl_modules(runtime, base)
    objects = runtime / SURFACES
    assert sorted(path.name for path in objects.iterdir()) == sorted(MDL_MODULES)
    stray = objects / MDL_MODULES[0]
    stray.unlink()
    stray.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="does not link"):
        check_mdl_modules(runtime, base)


def test_the_archive_imports_exactly_these_modules_by_relative_name():
    objects = RUNTIME / SURFACES
    if not objects.is_dir():
        pytest.skip("HandsOn is not prepared")
    archived = [path for path in objects.glob("*.mdl") if not path.is_symlink()]
    assert ARCHIVED_OMNIGLASS in {path.stem for path in archived}
    imported = set()
    for path in archived:
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == path.stem
        text = data.decode()
        if re.search(r"^mdl 1\.[6-9];", text, re.M):  # MDL 1.6 made relative imports strict
            imported |= {name + ".mdl" for name in re.findall(r"^(?:import|using)\s+(\w+)", text, re.M)}
    assert imported == set(MDL_MODULES)
