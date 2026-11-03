"""The learner's progress command must fail honestly and respect marker filters."""

from __future__ import annotations

import subprocess

import pytest

from scripts import grade


@pytest.mark.parametrize("include_slow", [False, True])
def test_grade_uses_one_combined_marker(monkeypatch, include_slow):
    seen = []
    def run(cmd, **kwargs):
        seen.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")
    monkeypatch.setattr(grade.subprocess, "run", run)
    grade.run_files(["tests/example.py"], "solutions", include_slow)
    pytest_args = seen[3:]  # exclude Python's own '-m pytest'
    assert pytest_args.count("-m") == 1
    expression = pytest_args[pytest_args.index("-m") + 1]
    assert expression == ("not gpu" if include_slow else "not slow and not gpu")


@pytest.mark.parametrize("status,expected", [("FAIL", 1), ("PART", 1), ("PASS", 0)])
def test_grade_exit_status(monkeypatch, status, expected):
    monkeypatch.setattr(grade.sys, "argv", ["grade.py", "--stage", "0"])
    monkeypatch.setattr(grade, "run_files", lambda *a: (status, "failure details", 0, 1, 0))
    assert grade.main() == expected
