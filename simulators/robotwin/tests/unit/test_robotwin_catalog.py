import io
import zipfile

import pytest

from rosbridge.assets import AssetStore
from rosbridge.catalog import resolve, tasks
from rosbridge_robotwin import assets


def test_catalog_pairs_every_evaluation_task_with_the_aloha_rig():
    entries = list(tasks(name="robotwin").values())
    assert len(entries) == 50 and len({t["native_task"] for t in entries}) == 50
    for task in entries:
        assert task["id"] == "robotwin/" + task["native_task"].replace("_", "-") + "-aloha-agilex"
        assert (task["robot"], task["embodiment"], task["task_config"]) == (
            "robotwin/robotwin-aloha-agilex", "aloha-agilex", "demo_clean")
        assert resolve(task["id"])[2] == task
    assert resolve("robotwin")[2]["id"] == "robotwin/handover-block-aloha-agilex"


def test_seed_adoption_verifies_archive_and_every_extracted_file(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr("pack/a.txt", "alpha")
        bundle.writestr("pack/sub/b.bin", b"\x00\x01")
        bundle.writestr("__MACOSX/pack/._a.txt", "metadata")
    archive = tmp_path / "pack.zip"
    archive.write_bytes(payload.getvalue())
    seed = tmp_path / "pack-extracted"
    (seed / "sub").mkdir(parents=True)
    (seed / "a.txt").write_text("alpha")
    (seed / "sub/b.bin").write_bytes(b"\x00\x02")
    digest = assets._sha256(archive)
    verify = assets.verify_seed(archive, "pack", archive.stat().st_size, digest)
    store = AssetStore(tmp_path / "store")
    with pytest.raises(ValueError, match="differs"):
        store.adopt("robotwin", "pack", digest, seed, verify=verify, subdirectory="pack")
    (seed / "sub/b.bin").write_bytes(b"\x00\x01")
    (seed / "extra").write_text("not in the archive")
    with pytest.raises(ValueError, match="exactly"):
        store.adopt("robotwin", "pack", digest, seed, verify=verify, subdirectory="pack")
    (seed / "extra").unlink()
    data = store.adopt("robotwin", "pack", digest, seed, verify=verify, subdirectory="pack")
    assert (data / "pack/sub/b.bin").read_bytes() == b"\x00\x01" and not seed.exists()
    with pytest.raises(ValueError, match="pinned"):
        assets.verify_seed(archive, "pack", 1, digest)(data / "pack")
