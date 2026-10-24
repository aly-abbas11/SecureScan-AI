"""Tests for the CWE remediation lookup in :mod:`src.models.fix_suggester`.

The module used to open ``cwe_database.json`` through a bare relative path, so
importing it from anywhere but ``src/models`` raised ``FileNotFoundError``.
``test_database_path_is_independent_of_cwd`` covers that.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from src.models import fix_suggester
from src.models.fix_suggester import (
    CWE_DATABASE,
    format_fix_report,
    get_fix_suggestion,
    list_cwes,
    main,
    normalise_cwe_id,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("119", "CWE-119"),
        ("cwe-119", "CWE-119"),
        ("CWE-119", "CWE-119"),
        (" Cwe-119 ", "CWE-119"),
    ],
)
def test_normalise_cwe_id(raw, expected):
    assert normalise_cwe_id(raw) == expected


def test_database_is_loaded():
    """A missing or empty file would silently turn every lookup into a miss."""
    assert CWE_DATABASE
    assert fix_suggester.DATABASE_PATH.is_file()


def test_known_cwe_returns_remediation():
    result = get_fix_suggestion("119")
    assert result["found"] is True
    assert result["cwe_id"] == "CWE-119"
    for field in (
        "name",
        "severity",
        "description",
        "why_dangerous",
        "bad_code",
        "good_code",
        "fix_steps",
    ):
        assert result[field], field


def test_unknown_cwe_reports_a_miss():
    result = get_fix_suggestion("CWE-999999")
    assert result["found"] is False
    assert "not in the database" in result["message"]


def test_list_cwes_is_sorted_by_severity():
    records = list_cwes()
    assert records

    order = {name: i for i, name in enumerate(fix_suggester.SEVERITY_ORDER)}
    severities = [order[record["severity"]] for record in records]
    assert severities == sorted(severities)


def test_format_report_contains_every_section():
    report = format_fix_report("119")
    for section in (
        "VULNERABILITY REPORT",
        "WHAT IS IT?",
        "WHY IS IT DANGEROUS?",
        "VULNERABLE CODE:",
        "FIXED CODE:",
        "HOW TO FIX IT:",
    ):
        assert section in report


def test_format_report_handles_an_unknown_id():
    assert format_fix_report("CWE-999999").startswith("Not found:")


def test_main_exit_codes(capsys):
    assert main(["119"]) == 0
    assert main(["CWE-999999"]) == 1
    assert main([]) == 2
    assert main(["--list"]) == 0

    captured = capsys.readouterr()
    assert "CWE-119" in captured.out
    assert "No CWE id given" in captured.err


def test_database_path_is_independent_of_cwd(tmp_path, monkeypatch):
    """Importing must work from any directory, not just ``src/models``."""
    module_name = "src.models.fix_suggester"
    original = sys.modules[module_name]

    monkeypatch.chdir(tmp_path)
    sys.modules.pop(module_name)
    try:
        reloaded = importlib.import_module(module_name)
        assert reloaded.DATABASE_PATH.is_file()
        assert reloaded.CWE_DATABASE
    finally:
        sys.modules[module_name] = original