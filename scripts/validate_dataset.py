#!/usr/bin/env python3
"""
validate_dataset.py

Validate the raw accelerometer gesture dataset for Lab 2.

Expected default repository layout:

    data/
      raw/
        <gesture_name>/
          p01_s01_<gesture_name>_001.csv
          ...

Expected default CSV columns:

    timestamp,accel_x,accel_y,accel_z

The parent directory under data/raw is treated as the authoritative class
label. The gesture name in the filename must match that directory.

This script uses only the Python standard library.

Examples:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --raw-dir data/raw
    python scripts/validate_dataset.py --balance-threshold 0.25
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_REQUIRED_COLUMNS = ("timestamp", "accel_x", "accel_y", "accel_z")

# Examples:
#   p01_s02_circle_003.csv
#   p01_s02_left_right_003.csv
FILENAME_RE = re.compile(
    r"^(?P<participant>p\d{2})_"
    r"(?P<session>s\d{2})_"
    r"(?P<gesture>[A-Za-z0-9][A-Za-z0-9_-]*)_"
    r"(?P<repetition>\d{3})\.csv$"
)


@dataclass(frozen=True)
class Issue:
    severity: str  # ERROR or WARNING
    path: Path
    message: str


@dataclass
class FileResult:
    path: Path
    class_name: str
    participant: str | None = None
    session: str | None = None
    repetition: str | None = None
    row_count: int = 0
    valid: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Lab 2 raw accelerometer gesture dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset directory (default: data/raw).",
    )
    parser.add_argument(
        "--required-columns",
        nargs="+",
        default=list(DEFAULT_REQUIRED_COLUMNS),
        metavar="COLUMN",
        help=(
            "Required CSV columns "
            "(default: timestamp accel_x accel_y accel_z)."
        ),
    )
    parser.add_argument(
        "--balance-threshold",
        type=float,
        default=0.20,
        help=(
            "Warn when a class count differs from the mean class count by more "
            "than this fraction (default: 0.20 = 20%%)."
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum number of data rows required in each CSV (default: 1).",
    )
    parser.add_argument(
        "--allow-nonstrict-filenames",
        action="store_true",
        help=(
            "Treat filenames that do not match "
            "p##_s##_<gesture>_###.csv as warnings instead of errors."
        ),
    )
    return parser.parse_args()


def add_issue(
    issues: list[Issue],
    severity: str,
    path: Path,
    message: str,
) -> None:
    issues.append(Issue(severity=severity, path=path, message=message))


def visible_csv_files(raw_dir: Path) -> list[Path]:
    """Return CSV files while ignoring hidden files/directories."""
    files: list[Path] = []

    for path in raw_dir.rglob("*.csv"):
        try:
            relative = path.relative_to(raw_dir)
        except ValueError:
            continue

        if any(part.startswith(".") for part in relative.parts):
            continue

        files.append(path)

    return sorted(files)


def validate_csv(
    path: Path,
    required_columns: list[str],
    min_rows: int,
    issues: list[Issue],
) -> tuple[int, bool]:
    """Validate CSV structure and numeric data."""
    had_error = False

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                add_issue(issues, "ERROR", path, "CSV has no header row.")
                return 0, False

            original_fieldnames = list(reader.fieldnames)
            normalized_fieldnames = [name.strip() for name in original_fieldnames]

            missing_columns = [
                col for col in required_columns if col not in normalized_fieldnames
            ]
            if missing_columns:
                add_issue(
                    issues,
                    "ERROR",
                    path,
                    "Missing required column(s): " + ", ".join(missing_columns),
                )
                had_error = True

            if len(set(normalized_fieldnames)) != len(normalized_fieldnames):
                add_issue(
                    issues,
                    "ERROR",
                    path,
                    "CSV contains duplicate column names.",
                )
                had_error = True

            # DictReader uses original field names as dictionary keys. This
            # mapping lets us tolerate surrounding whitespace in the header.
            column_key = {
                normalized: original
                for normalized, original in zip(
                    normalized_fieldnames, original_fieldnames
                )
            }

            row_count = 0
            previous_timestamp: float | None = None
            timestamp_warning_reported = False

            for line_number, row in enumerate(reader, start=2):
                row_count += 1

                for column in required_columns:
                    if column not in column_key:
                        continue

                    raw_value = row.get(column_key[column])
                    value = "" if raw_value is None else raw_value.strip()

                    if value == "":
                        add_issue(
                            issues,
                            "ERROR",
                            path,
                            f"Missing value in column '{column}' at line {line_number}.",
                        )
                        had_error = True
                        continue

                    try:
                        number = float(value)
                    except ValueError:
                        add_issue(
                            issues,
                            "ERROR",
                            path,
                            (
                                f"Non-numeric value '{value}' in column "
                                f"'{column}' at line {line_number}."
                            ),
                        )
                        had_error = True
                        continue

                    if not math.isfinite(number):
                        add_issue(
                            issues,
                            "ERROR",
                            path,
                            (
                                f"Non-finite value '{value}' in column "
                                f"'{column}' at line {line_number}."
                            ),
                        )
                        had_error = True
                        continue

                    if column == "timestamp":
                        if (
                            previous_timestamp is not None
                            and number < previous_timestamp
                            and not timestamp_warning_reported
                        ):
                            add_issue(
                                issues,
                                "WARNING",
                                path,
                                (
                                    "Timestamp decreases within the file "
                                    f"(first observed at line {line_number})."
                                ),
                            )
                            timestamp_warning_reported = True
                        previous_timestamp = number

            if row_count < min_rows:
                add_issue(
                    issues,
                    "ERROR",
                    path,
                    (
                        f"CSV contains {row_count} data row(s); "
                        f"minimum required is {min_rows}."
                    ),
                )
                had_error = True

            return row_count, not had_error

    except UnicodeDecodeError:
        add_issue(
            issues,
            "ERROR",
            path,
            "File is not valid UTF-8/UTF-8-BOM text.",
        )
        return 0, False
    except csv.Error as exc:
        add_issue(issues, "ERROR", path, f"CSV parsing error: {exc}")
        return 0, False
    except OSError as exc:
        add_issue(issues, "ERROR", path, f"Could not read file: {exc}")
        return 0, False


def validate_file(
    path: Path,
    raw_dir: Path,
    required_columns: list[str],
    min_rows: int,
    strict_filenames: bool,
    issues: list[Issue],
) -> FileResult:
    relative = path.relative_to(raw_dir)

    if len(relative.parts) < 2:
        add_issue(
            issues,
            "ERROR",
            path,
            "CSV must be inside a class directory under data/raw.",
        )
        class_name = "<unknown>"
    else:
        class_name = relative.parts[0]

    result = FileResult(path=path, class_name=class_name)

    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        severity = "ERROR" if strict_filenames else "WARNING"
        add_issue(
            issues,
            severity,
            path,
            (
                "Filename does not match required format "
                "p##_s##_<gesture>_###.csv "
                "(example: p01_s02_circle_003.csv)."
            ),
        )
        if severity == "ERROR":
            result.valid = False
    else:
        result.participant = match.group("participant")
        result.session = match.group("session")
        filename_gesture = match.group("gesture")
        result.repetition = match.group("repetition")

        if filename_gesture != class_name:
            add_issue(
                issues,
                "ERROR",
                path,
                (
                    f"Filename gesture '{filename_gesture}' does not match "
                    f"parent directory label '{class_name}'."
                ),
            )
            result.valid = False

    row_count, csv_valid = validate_csv(
        path=path,
        required_columns=required_columns,
        min_rows=min_rows,
        issues=issues,
    )
    result.row_count = row_count
    result.valid = result.valid and csv_valid

    return result


def print_counter(title: str, counts: Counter[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not counts:
        print("None")
        return

    width = max(len(str(key)) for key in counts)
    for key in sorted(counts):
        print(f"{key:<{width}} : {counts[key]}")


def print_issues(title: str, issues: Iterable[Issue]) -> None:
    selected = list(issues)
    if not selected:
        return

    print(f"\n{title}")
    print("-" * len(title))
    for issue in selected:
        print(f"[{issue.severity}] {issue.path}: {issue.message}")


def main() -> int:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    required_columns = [column.strip() for column in args.required_columns]

    if args.balance_threshold < 0:
        print("ERROR: --balance-threshold must be >= 0.", file=sys.stderr)
        return 2

    if args.min_rows < 1:
        print("ERROR: --min-rows must be >= 1.", file=sys.stderr)
        return 2

    print("Lab 2 Dataset Validation")
    print("=" * 24)
    print(f"Raw directory       : {raw_dir}")
    print(f"Required columns    : {', '.join(required_columns)}")
    print("Filename rule       : p##_s##_<gesture>_###.csv")
    print(
        "Balance warning     : "
        f">{args.balance_threshold * 100:.0f}% from mean class count"
    )

    if not raw_dir.exists():
        print(f"\nERROR: Raw dataset directory does not exist: {raw_dir}")
        return 1

    if not raw_dir.is_dir():
        print(f"\nERROR: Raw dataset path is not a directory: {raw_dir}")
        return 1

    csv_files = visible_csv_files(raw_dir)
    if not csv_files:
        print("\nERROR: No CSV files were found under the raw dataset directory.")
        return 1

    issues: list[Issue] = []
    results: list[FileResult] = []
    seen_capture_ids: dict[tuple[str, str, str, str], Path] = {}

    for path in csv_files:
        result = validate_file(
            path=path,
            raw_dir=raw_dir,
            required_columns=required_columns,
            min_rows=args.min_rows,
            strict_filenames=not args.allow_nonstrict_filenames,
            issues=issues,
        )
        results.append(result)

        if (
            result.participant is not None
            and result.session is not None
            and result.repetition is not None
            and result.class_name != "<unknown>"
        ):
            capture_id = (
                result.participant,
                result.session,
                result.class_name,
                result.repetition,
            )
            previous_path = seen_capture_ids.get(capture_id)
            if previous_path is not None:
                add_issue(
                    issues,
                    "ERROR",
                    path,
                    (
                        "Duplicate capture identifier "
                        f"{capture_id}; first seen at {previous_path}."
                    ),
                )
                result.valid = False
            else:
                seen_capture_ids[capture_id] = path

    class_counts = Counter(
        result.class_name
        for result in results
        if result.class_name != "<unknown>"
    )
    participant_counts = Counter(
        result.participant
        for result in results
        if result.participant is not None
    )
    session_counts = Counter(
        result.session
        for result in results
        if result.session is not None
    )

    # Balance is a warning rather than a hard validation failure.
    if len(class_counts) >= 2:
        avg_count = mean(class_counts.values())
        allowed_difference = avg_count * args.balance_threshold

        for class_name, count in sorted(class_counts.items()):
            if abs(count - avg_count) > allowed_difference:
                add_issue(
                    issues,
                    "WARNING",
                    raw_dir / class_name,
                    (
                        f"Class has {count} capture(s), which differs from the "
                        f"mean class count ({avg_count:.1f}) by more than "
                        f"{args.balance_threshold * 100:.0f}%."
                    ),
                )

    errors = [issue for issue in issues if issue.severity == "ERROR"]
    warnings = [issue for issue in issues if issue.severity == "WARNING"]

    paths_with_errors = {issue.path for issue in errors}
    valid_files = sum(
        1
        for result in results
        if result.valid and result.path not in paths_with_errors
    )

    print("\nSummary")
    print("-------")
    print(f"CSV files checked   : {len(results)}")
    print(f"Valid files         : {valid_files}")
    print(f"Files with errors   : {len(results) - valid_files}")
    print(f"Errors              : {len(errors)}")
    print(f"Warnings            : {len(warnings)}")

    row_counts = [result.row_count for result in results if result.row_count > 0]
    if row_counts:
        print(f"Shortest capture    : {min(row_counts)} rows")
        print(f"Longest capture     : {max(row_counts)} rows")
        print(f"Average capture     : {mean(row_counts):.1f} rows")

    print_counter("Class counts", class_counts)
    print_counter("Participant counts", participant_counts)
    print_counter("Session counts", session_counts)

    print_issues("Errors", errors)
    print_issues("Warnings", warnings)

    if errors:
        print("\nRESULT: FAIL")
        print("Fix the errors above before using or submitting the dataset.")
        return 1

    print("\nRESULT: PASS")
    if warnings:
        print("The dataset passed structural validation, but review the warnings.")
    else:
        print("No structural errors or warnings were found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
