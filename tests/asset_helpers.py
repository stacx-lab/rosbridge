import hashlib
import io
import tarfile

def package(tmp_path, name="robot", contents=b"mesh bytes"):
    archive = tmp_path / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(f"{name}/mesh.obj")
        info.size = len(contents)
        bundle.addfile(info, io.BytesIO(contents))
    return {
        "name": name, "uri": archive.as_uri(), "extract_dir": "abc_sim/models/assets",
        "contains": [name], "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }

def ensure(store, pkg):
    return store.ensure("abc", pkg["name"], pkg["uri"], checksum=pkg["sha256"],
                        size=pkg["size_bytes"], contains=pkg["contains"])
