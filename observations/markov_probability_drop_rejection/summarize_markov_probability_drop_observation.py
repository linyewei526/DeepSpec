#!/usr/bin/env python3
"""Read-only summary of a selected-token Markov probability-drop rejection-prediction run."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", default=None, help="Show only one dataset.")
    parser.add_argument(
        "--family",
        choices=("both", "absolute", "percentage"),
        default="both",
        help="Select the threshold family shown with --show-thresholds/--threshold.",
    )
    parser.add_argument(
        "--show-thresholds",
        action="store_true",
        help="Print every threshold row after the denominator summary.",
    )
    parser.add_argument(
        "--threshold",
        default=None,
        help="Print only an exact decimal threshold, for example 0.100.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def print_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    if not rows:
        print("No matching rows.")
        return
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(len(header))
    ]
    print("  ".join(header[index].ljust(widths[index]) for index in range(len(header))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(header))))


def normalize_threshold_label(value: str) -> str:
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid --threshold value: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError("--threshold must be finite")
    text = format(decimal_value, "f")
    integer, separator, fraction = text.partition(".")
    if not separator:
        fraction = ""
    fraction = fraction.rstrip("0").ljust(3, "0")
    return f"{integer}.{fraction}"


def main() -> None:
    cli = parse_args()
    run_dir = cli.run_dir.expanduser().resolve()
    settings_path = run_dir / "settings.json"
    manifest_path = run_dir / "experiment_manifest.json"
    results_path = run_dir / "dataset_results.jsonl"
    if not settings_path.is_file():
        raise FileNotFoundError(f"settings.json not found: {settings_path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"status": "manifest_missing"}
    )
    result_rows = load_jsonl(results_path)
    if cli.dataset is not None:
        result_rows = [row for row in result_rows if row.get("dataset") == cli.dataset]

    print(f"run_dir: {run_dir}")
    print(f"experiment: {settings.get('experiment')}")
    print(f"status: {manifest.get('status')}")
    print(f"completed_datasets: {manifest.get('completed_dataset_count', 0)}")
    if not result_rows:
        print("No matching completed dataset result is available.")
        return

    denominator_rows = []
    for result in result_rows:
        summary = result.get("markov_probability_drop_observation_summary") or {}
        counts = summary.get("counts") or {}
        denominator_rows.append(
            (
                str(result.get("dataset")),
                str(summary.get("sample_count", "-")),
                str(counts.get("verification_rounds", "-")),
                str(counts.get("eligible_accepted_token_count", "-")),
                str(counts.get("eligible_rejected_token_count", "-")),
                str(counts.get("unscorable_first_position_rejections", "-")),
                str(counts.get("ignored_after_first_rejection_token_count", "-")),
                str(counts.get("ignored_after_accepted_eos_token_count", "-")),
            )
        )
    print_table(
        (
            "dataset",
            "samples",
            "rounds",
            "eligible_accept",
            "eligible_reject",
            "reject@pos0",
            "ignored_after_reject",
            "ignored_after_eos",
        ),
        denominator_rows,
    )

    if not cli.show_thresholds and cli.threshold is None:
        print("Use --show-thresholds for all rows or --threshold 0.100 for one threshold.")
        return

    requested_label = (
        normalize_threshold_label(cli.threshold)
        if cli.threshold is not None
        else None
    )
    family_specs = []
    if cli.family in ("both", "absolute"):
        family_specs.append(("absolute", "absolute_drop_thresholds"))
    if cli.family in ("both", "percentage"):
        family_specs.append(("percentage", "percentage_drop_thresholds"))

    threshold_rows = []
    for result in result_rows:
        summary = result.get("markov_probability_drop_observation_summary") or {}
        for family_name, key in family_specs:
            for row in summary.get(key) or []:
                if requested_label is not None and row.get("threshold_label") != requested_label:
                    continue
                threshold_rows.append(
                    (
                        str(result.get("dataset")),
                        family_name,
                        str(row.get("threshold_label", row.get("threshold"))),
                        str(row.get("accepted_count", "-")),
                        str(row.get("rejected_count", "-")),
                        str(row.get("flagged_evaluable_count", "-")),
                        fmt(row.get("accepted_share_among_flagged")),
                        fmt(row.get("rejected_share_among_flagged")),
                        fmt(row.get("accepted_flag_rate")),
                        fmt(row.get("rejected_capture_rate")),
                    )
                )
    print_table(
        (
            "dataset",
            "family",
            "threshold",
            "accepted",
            "rejected",
            "flagged",
            "accept_share",
            "reject_precision",
            "accepted_FPR",
            "reject_recall",
        ),
        threshold_rows,
    )


if __name__ == "__main__":
    main()
