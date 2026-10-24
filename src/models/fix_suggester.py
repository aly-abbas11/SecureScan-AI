"""
CWE remediation lookup for SecureScan AI.

Maps a CWE identifier to a human-readable explanation and a concrete fix.

The database path is resolved relative to this file rather than the current
working directory, so importing or running this module works from anywhere::

    python -m src.models.fix_suggester CWE-119
    python -m src.models.fix_suggester 119 89 79
    python -m src.models.fix_suggester --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Resolve the database next to this module. The original implementation used a
# bare relative path, so importing from any other directory raised
# FileNotFoundError.
DATABASE_PATH = Path(__file__).resolve().parent / "cwe_database.json"

SEVERITY_EMOJI = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}

# Order used when rendering "list" output.
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]


def load_cwe_database(path: str | Path = DATABASE_PATH) -> Dict[str, Dict[str, Any]]:
    """Load the CWE remediation database.

    Args:
        path: Location of the JSON database.

    Returns:
        Mapping of CWE id to its remediation record.

    Raises:
        FileNotFoundError: If the database is missing.
        ValueError: If the file is not a JSON object.
    """
    database_path = Path(path)
    if not database_path.is_file():
        raise FileNotFoundError(
            f"CWE database not found at {database_path}. "
            "It ships with the repository at src/models/cwe_database.json."
        )

    with database_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{database_path} should contain a JSON object of CWE records.")

    return data


# Loaded once at import. A failure here is a genuine packaging error, so it is
# allowed to propagate rather than being swallowed into an empty dict.
CWE_DATABASE = load_cwe_database()


def normalise_cwe_id(cwe_id: str) -> str:
    """Coerce a user-supplied identifier into ``CWE-<number>`` form.

    Args:
        cwe_id: e.g. ``"119"``, ``"cwe-119"``, ``" CWE-119 "``.

    Returns:
        The canonical identifier, e.g. ``"CWE-119"``.
    """
    cleaned = str(cwe_id).strip().upper()
    if not cleaned.startswith("CWE-"):
        cleaned = f"CWE-{cleaned}"
    return cleaned


def get_fix_suggestion(cwe_id: str) -> Dict[str, Any]:
    """Look up remediation guidance for a CWE identifier.

    Args:
        cwe_id: CWE identifier in any of the accepted forms.

    Returns:
        A record with ``found=True`` and the database fields, or ``found=False``
        plus a ``message`` explaining the miss.
    """
    canonical = normalise_cwe_id(cwe_id)

    if canonical not in CWE_DATABASE:
        return {
            "found": False,
            "cwe_id": canonical,
            "message": (
                f"{canonical} is not in the database "
                f"({len(CWE_DATABASE)} entries available; "
                "run with --list to see them)."
            ),
        }

    data = CWE_DATABASE[canonical]
    return {
        "found": True,
        "cwe_id": canonical,
        "name": data["name"],
        "severity": data["severity"],
        "description": data["description"],
        "why_dangerous": data["why_dangerous"],
        "bad_code": data["bad_code"],
        "good_code": data["good_code"],
        "fix_steps": data["fix_steps"],
    }


def format_fix_report(cwe_id: str) -> str:
    """Render remediation guidance as a printable report.

    Args:
        cwe_id: CWE identifier in any of the accepted forms.

    Returns:
        The formatted report, or a short message if the CWE is unknown.
    """
    result = get_fix_suggestion(cwe_id)
    if not result["found"]:
        return f"Not found: {result['message']}"

    emoji = SEVERITY_EMOJI.get(result["severity"], "⚪")
    lines = [
        "=" * 60,
        "VULNERABILITY REPORT",
        "=" * 60,
        f"CWE ID:    {result['cwe_id']}",
        f"Name:      {result['name']}",
        f"Severity:  {emoji} {result['severity']}",
        "-" * 60,
        "",
        "WHAT IS IT?",
        result["description"],
        "",
        "WHY IS IT DANGEROUS?",
        result["why_dangerous"],
        "",
        "VULNERABLE CODE:",
        result["bad_code"],
        "",
        "FIXED CODE:",
        result["good_code"],
        "",
        "HOW TO FIX IT:",
    ]
    lines += [f"  {i}. {step}" for i, step in enumerate(result["fix_steps"], 1)]
    lines.append("=" * 60)
    return "\n".join(lines)


def print_fix_report(cwe_id: str) -> None:
    """Print :func:`format_fix_report` to stdout.

    Args:
        cwe_id: CWE identifier in any of the accepted forms.
    """
    print(format_fix_report(cwe_id))


def list_cwes() -> List[Dict[str, str]]:
    """Summarise the database, ordered by severity then identifier.

    Returns:
        One ``{"cwe_id", "name", "severity"}`` record per entry.
    """
    order = {name: i for i, name in enumerate(SEVERITY_ORDER)}
    records = [
        {
            "cwe_id": cwe_id,
            "name": record["name"],
            "severity": record["severity"],
        }
        for cwe_id, record in CWE_DATABASE.items()
    ]
    return sorted(
        records,
        key=lambda r: (order.get(r["severity"], len(order)), r["cwe_id"]),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.models.fix_suggester",
        description="Look up CWE remediation guidance.",
    )
    parser.add_argument(
        "cwe_ids",
        nargs="*",
        help="One or more CWE identifiers, e.g. CWE-119 or 119.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List every CWE in the database and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    if args.list:
        for record in list_cwes():
            emoji = SEVERITY_EMOJI.get(record["severity"], "⚪")
            print(f"{record['cwe_id']:<10} {emoji} {record['severity']:<9} {record['name']}")
        return 0

    if not args.cwe_ids:
        print(
            "No CWE id given. Try: python -m src.models.fix_suggester --list",
            file=sys.stderr,
        )
        return 2

    exit_code = 0
    for index, cwe_id in enumerate(args.cwe_ids):
        if index:
            print()
        print(format_fix_report(cwe_id))
        if not get_fix_suggestion(cwe_id)["found"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
