"""Lightweight structural checks for the project layout.

These have no heavy (torch / paddleocr / playwright) dependencies, so they run
anywhere and guard against the entry points or package modules being moved
without the imports being updated.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_entry_points_exist():
    assert os.path.isfile(os.path.join(ROOT, "apps", "cli", "inference.py"))
    assert os.path.isfile(os.path.join(ROOT, "apps", "server", "app.py"))


def test_package_modules_exist():
    assert os.path.isfile(os.path.join(ROOT, "lib", "config.py"))
    assert os.path.isfile(os.path.join(ROOT, "lib", "labeling", "label_eml.py"))


def test_scripts_moved():
    assert os.path.isfile(os.path.join(ROOT, "scripts", "get_model.sh"))


def test_no_stale_root_modules():
    # These were relocated into lib/ and apps/; they must not linger at the root.
    for stale in ("config.py", "label_eml.py", "inference.py", "app.py"):
        assert not os.path.isfile(os.path.join(ROOT, stale)), f"{stale} should have been moved"
