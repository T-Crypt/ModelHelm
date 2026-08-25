import subprocess
from modelhelm.git.inspector import GitInspector

def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)

def test_snapshot_clean_repo(tmp_path):
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))
    snapshot = inspector.snapshot()

    assert snapshot.branch == "main"
    assert len(snapshot.commit) == 40
    assert snapshot.is_dirty is False

def test_snapshot_dirty_after_edit(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed\n")
    inspector = GitInspector(str(tmp_path))

    assert inspector.snapshot().is_dirty is True
    assert inspector.files_changed_count() == 1
    assert "README.md" in inspector.diff_summary()
