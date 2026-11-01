import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_locked_test  # noqa: E402


def make_tree(root: Path):
    (root / "bidrig" / "detectors").mkdir(parents=True)
    (root / "bidrig" / "a.py").write_text("A = 1\n")
    (root / "bidrig" / "detectors" / "previous.py").write_text("B = 1\n")


def test_hash_changes_when_a_detector_file_changes(tmp_path, monkeypatch):
    make_tree(tmp_path)
    monkeypatch.setattr(run_locked_test, "ROOT", tmp_path)
    h0 = run_locked_test.design_hash("previous")
    assert h0 == run_locked_test.design_hash("previous")  # deterministic
    (tmp_path / "bidrig" / "detectors" / "previous.py").write_text("B = 2\n")
    assert run_locked_test.design_hash("previous") != h0


def test_hash_changes_with_package_file_and_design_name(tmp_path, monkeypatch):
    make_tree(tmp_path)
    monkeypatch.setattr(run_locked_test, "ROOT", tmp_path)
    h0 = run_locked_test.design_hash("previous")
    assert run_locked_test.design_hash("full_gbm") != h0
    (tmp_path / "bidrig" / "a.py").write_text("A = 2\n")
    assert run_locked_test.design_hash("previous") != h0


def test_hash_works_without_a_detectors_folder(tmp_path, monkeypatch):
    (tmp_path / "bidrig").mkdir()
    (tmp_path / "bidrig" / "a.py").write_text("A = 1\n")
    monkeypatch.setattr(run_locked_test, "ROOT", tmp_path)
    assert len(run_locked_test.design_hash("full_gbm")) == 16


def test_hash_covers_every_subpackage(tmp_path, monkeypatch):
    make_tree(tmp_path)
    (tmp_path / "bidrig" / "features").mkdir()
    feat = tmp_path / "bidrig" / "features" / "dependence.py"
    feat.write_text("C = 1\n")
    monkeypatch.setattr(run_locked_test, "ROOT", tmp_path)
    h0 = run_locked_test.design_hash("full_gbm")
    feat.write_text("C = 2\n")
    assert run_locked_test.design_hash("full_gbm") != h0
