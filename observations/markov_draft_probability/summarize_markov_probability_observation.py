#!/usr/bin/env python3
"""Read-only summary of a completed/in-progress selected-token Markov draft-probability observation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", default=None, help="Show only one dataset.")
    parser.add_argument(
        "--show-cdf",
        action="store_true",
        help="Also print all detailed width-0.05 CDF rows as JSON.",
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
    rows = load_jsonl(results_path)
    if cli.dataset is not None:
        rows = [row for row in rows if row.get("dataset") == cli.dataset]

    print(f"run_dir: {run_dir}")
    print(f"experiment: {settings.get('experiment')}")
    print(f"status: {manifest.get('status')}")
    print(f"completed_datasets: {manifest.get('completed_dataset_count', 0)}")
    if not rows:
        print("No matching completed dataset result is available.")
        return

    header = (
        "dataset",
        "samples",
        "accepted_n",
        "accepted_q_mean",
        "corrections",
        "rejected_q_mean",
        "paired_n",
        "gap_mean",
        "gap_mean_%",
        *[f"rank{index}" for index in range(1, 11)],
        "rank_other",
    )
    table_rows = []
    for row in rows:
        summary = row.get("markov_draft_probability_observation_summary") or {}
        ranks = summary.get("true_draft_rank_probabilities") or {}
        table_rows.append(
            (
                str(row.get("dataset")),
                str(summary.get("sample_count", "-")),
                str(summary.get("accepted_token_count", "-")),
                fmt(summary.get("accepted_probability_mean")),
                str(summary.get("correction_events", "-")),
                fmt(summary.get("rejected_probability_mean")),
                str(summary.get("paired_gap_events", "-")),
                fmt(summary.get("signed_absolute_gap_mean")),
                fmt(summary.get("signed_relative_gap_mean_percent")),
                *[fmt(ranks.get(str(index))) for index in range(1, 11)],
                fmt(ranks.get("other")),
            )
        )
    widths = [max(len(header[index]), *(len(row[index]) for row in table_rows)) for index in range(len(header))]
    print("  ".join(header[index].ljust(widths[index]) for index in range(len(header))))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(header))))

    if cli.show_cdf:
        artifact_root = Path(settings["outputs"]["markov_draft_probability_artifact_root"])
        for result_row in rows:
            dataset = str(result_row["dataset"])
            metrics_path = artifact_root / dataset / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(json.dumps(payload["observation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
