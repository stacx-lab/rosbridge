"""Real git fixture: ordered patches apply only to the generated copy."""
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rosbridge.catalog import preparation_tasks
from rosbridge.provisioning import setup

spec = importlib.util.spec_from_file_location(
    "prepare_sim", Path(__file__).resolve().parents[2] / "src/rosbridge/provisioning/source.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_patches_do_not_change_upstream(tmp_path):
    source = tmp_path / "third_party/fake"
    source.mkdir(parents=True)
    git(source, "init", "-q")
    (source / "value.txt").write_text("before\n")
    git(source, "add", "value.txt")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    revision = git(source, "rev-parse", "HEAD")
    adapter = tmp_path / "simulators/fake"
    adapter.mkdir(parents=True)
    (adapter / "simulator.toml").write_text(
        f'name="fake"\nsource="third_party/fake"\npatches="patches/fake"\nsource_revision="{revision}"\n')
    (tmp_path / "src/rosbridge").mkdir(parents=True)
    (tmp_path / "pixi.toml").write_text("")
    patches = tmp_path / "patches/fake"
    patches.mkdir(parents=True)
    (patches / "series").write_text("# ordered\nfix.patch\nsecond.patch\n")
    (patches / "fix.patch").write_text(
        "diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n"
        "@@ -1 +1 @@\n-before\n+after\n")
    (patches / "second.patch").write_text(
        "diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n"
        "@@ -1 +1 @@\n-after\n+final\n")
    destination, _ = prepare.materialize(tmp_path, "fake")
    assert (destination / "value.txt").read_text() == "final\n"
    assert (source / "value.txt").read_text() == "before\n"
    assert git(source, "status", "--porcelain") == ""
    assert json.loads((destination / ".prepared.json").read_text())["revision"] == revision
    assert prepare.materialize(tmp_path, "fake")[0] == destination
    (patches / "series").write_text("second.patch\nfix.patch\n")
    with pytest.raises(ValueError, match="stale"):
        prepare.materialize(tmp_path, "fake")
    (patches / "series").write_text("fix.patch\nsecond.patch\n")
    (patches / "fix.patch").write_text("changed patch\n")
    with pytest.raises(ValueError, match="stale"):
        prepare.materialize(tmp_path, "fake")
    shutil.rmtree(destination)
    with pytest.raises(subprocess.CalledProcessError):
        prepare.materialize(tmp_path, "fake")
    assert not destination.exists()  # Interrupted/failed export cannot poison a retry.
    (patches / "series").write_text("")
    recovered, _ = prepare.materialize(tmp_path, "fake")
    assert (recovered / "value.txt").read_text() == "before\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src/rosbridge").mkdir(parents=True)
    (tmp_path / "pixi.toml").touch()
    adapter = tmp_path / "simulators/fake"
    adapter.mkdir(parents=True)
    (adapter / "simulator.toml").write_text(
        'name="fake"\nsource="third_party/fake"\npython_path="simulators/fake"\n'
        'default_task="first_native"\ntask_catalog="tasks.toml"\n'
        'asset_preparer="fake_assets:prepare"\nenvironment="fake"\n')
    (adapter / "tasks.toml").write_text(
        '[[tasks]]\nid="fake/first-task"\nnative_task="first_native"\n'
        '[[tasks]]\nid="fake/second-task"\nnative_task="second_native"\n')
    return tmp_path


@pytest.mark.parametrize("task,expected", [(None, ["first_native", "second_native"]),
                                        ("second-task", ["second_native"]),
                                        ("second_native", ["second_native"])])
def test_source_prepares_catalog_or_explicit_task_once(project, monkeypatch, task, expected):
    _, config, entries = preparation_tasks("fake", project, task)
    assert [entry["native_task"] for entry in entries] == expected
    calls, installs = [], []
    monkeypatch.setattr(prepare, "materialize", lambda *args: (project / ".runtime/fake", config))
    monkeypatch.setattr(prepare.subprocess, "run", lambda command, **kw: installs.append(command))
    plugin = SimpleNamespace(prepare=lambda runtime, task: calls.append(task))
    monkeypatch.setattr(prepare.importlib, "import_module", lambda name: plugin)
    argv = ["source", "fake", "--project", str(project)]
    if task is not None:
        argv += ["--task", task]
    monkeypatch.setattr(sys, "argv", argv)
    # Source activation is a child-process concern; keep the test environment isolated.
    monkeypatch.setattr(prepare, "local_environment", lambda root: {})
    prepare.main()
    assert calls == expected
    assert len(installs) == 1


@pytest.mark.parametrize("selection,task,expected", [("fake", None, None),
                                                   ("fake", "second-task", "second_native"),
                                                   ("fake/second-task", None, "second_native")])
def test_prepare_builds_once_and_checks_every_selected_task(project, monkeypatch, selection, task, expected):
    calls, checks = [], []
    monkeypatch.setattr("rosbridge.provisioning.source.sync_source", lambda *args: None)
    monkeypatch.setattr(setup, "activated", lambda *args: (sys.executable, {}))
    monkeypatch.setattr(setup, "build_identity", lambda *args: {"test": True})
    monkeypatch.setattr(setup, "kinex_packages", lambda: project)
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *args, **kw: '{}')

    def run(command, **kwargs):
        calls.append(command)
        if "rosbridge.provisioning.workspace" in command:
            (project / ".workspaces/fake").mkdir(parents=True)

    def doctor(selection, **kwargs):
        checks.append(kwargs)
        return {"ready": True}

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup, "doctor", doctor)
    assert setup.prepare(selection, project=project, task=task)["ready"]
    source, = [command for command in calls if "rosbridge.provisioning.source" in command]
    assert (source[source.index("--task") + 1] if "--task" in source else None) == expected
    assert len([command for command in calls if "rosbridge.provisioning.workspace" in command]) == 1
    assert checks[0]["all_tasks"] == (expected is None)


def test_source_sync_without_parent_git_and_dirty_source_preservation(tmp_path):
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '-q')
    (origin / 'value').write_text('original')
    git(origin, 'add', 'value')
    git(origin, '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'initial')
    revision = git(origin, 'rev-parse', 'HEAD')
    root = tmp_path / 'project'
    root.mkdir()
    config = {'source': 'third_party/fake', 'source_url': str(origin), 'source_revision': revision}
    source = prepare.sync_source(root, config)
    assert git(source, 'rev-parse', 'HEAD') == revision
    assert prepare.sync_source(root, config) == source
    (source / 'value').write_text('local edit')
    with pytest.raises(ValueError, match='local changes'):
        prepare.sync_source(root, config)
    assert (source / 'value').read_text() == 'local edit'


def test_nested_sources_are_pinned_exported_and_patched_together(tmp_path):
    def repository(path, text):
        path.mkdir(parents=True)
        git(path, "init", "-q")
        (path / "value.txt").write_text(text)
        git(path, "add", "value.txt")
        git(path, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
        return git(path, "rev-parse", "HEAD")
    revision = repository(tmp_path / "third_party/main", "main\n")
    fork = repository(tmp_path / "third_party/fork", "fork\n")
    adapter = tmp_path / "simulators/fake"
    adapter.mkdir(parents=True)
    (adapter / "simulator.toml").write_text(
        f'name="fake"\nsource="third_party/main"\npatches="patches/fake"\nsource_revision="{revision}"\n'
        f'[[nested_sources]]\npath="vendor/fork"\nsource="third_party/fork"\nsource_url="unused"\n'
        f'source_revision="{fork}"\n')
    (tmp_path / "src/rosbridge").mkdir(parents=True)
    (tmp_path / "pixi.toml").write_text("")
    patches = tmp_path / "patches/fake"
    patches.mkdir(parents=True)
    (patches / "series").write_text("nested.patch\n")
    # Patches apply to the assembled tree, so they may edit nested sources.
    (patches / "nested.patch").write_text(
        "diff --git a/vendor/fork/value.txt b/vendor/fork/value.txt\n--- a/vendor/fork/value.txt\n"
        "+++ b/vendor/fork/value.txt\n@@ -1 +1 @@\n-fork\n+patched\n")
    destination, config = prepare.materialize(tmp_path, "fake")
    assert (destination / "value.txt").read_text() == "main\n"
    assert (destination / "vendor/fork/value.txt").read_text() == "patched\n"
    assert json.loads((destination / ".prepared.json").read_text())["nested"] == [
        {"path": "vendor/fork", "revision": fork}]
    assert prepare.fingerprint(tmp_path, config, revision)["nested"][0]["revision"] == fork
    (tmp_path / "third_party/fork/value.txt").write_text("dirty\n")
    shutil.rmtree(destination)
    with pytest.raises(ValueError, match="tracked changes"):
        prepare.materialize(tmp_path, "fake")
