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


def test_files_changed_since_counts_uncommitted_work(tmp_path):
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))
    base = inspector.snapshot().commit

    (tmp_path / "new.txt").write_text("x\n")

    assert inspector.files_changed_since(base) == 1


def test_files_changed_since_still_counts_after_commit(tmp_path):
    """The core bug: committing cleans the tree, so a raw status count drops to
    zero even though the run did change files."""
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))
    base = inspector.snapshot().commit

    (tmp_path / "new.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "add new"],
        check=True, capture_output=True,
    )

    assert inspector.files_changed_count() == 0  # the old, wrong answer
    assert inspector.files_changed_since(base) == 1


def test_files_changed_since_unions_committed_and_dirty_without_double_count(tmp_path):
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))
    base = inspector.snapshot().commit

    (tmp_path / "committed.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "one"],
        check=True, capture_output=True,
    )
    # Touch the same file again, plus a brand new one.
    (tmp_path / "committed.txt").write_text("a modified\n")
    (tmp_path / "dirty.txt").write_text("b\n")

    # committed.txt counted once despite being both committed and dirty.
    assert inspector.files_changed_since(base) == 2


def test_files_changed_since_excludes_preexisting_dirt(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "preexisting.txt").write_text("dirt\n")
    inspector = GitInspector(str(tmp_path))
    base = inspector.snapshot().commit
    base_dirty = inspector.dirty_files()

    assert inspector.files_changed_since(base, base_dirty) == 0

    (tmp_path / "agent.txt").write_text("mine\n")
    assert inspector.files_changed_since(base, base_dirty) == 1


def test_files_changed_since_counts_preexisting_file_once_agent_commits_it(tmp_path):
    """If the agent commits a file that was already dirty, that commit IS the
    agent's work and must count."""
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("pre-existing edit\n")
    inspector = GitInspector(str(tmp_path))
    base = inspector.snapshot().commit
    base_dirty = inspector.dirty_files()

    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "agent commits it"],
        check=True, capture_output=True,
    )

    assert inspector.files_changed_since(base, base_dirty) == 1


def test_files_changed_since_reports_zero_when_nothing_changed(tmp_path):
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))

    assert inspector.files_changed_since(inspector.snapshot().commit) == 0
