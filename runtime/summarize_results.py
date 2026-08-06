#!/usr/bin/env python3
"""Print a compact summary for one timestamped DSpark experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {run_dir}")
    manifest_path = run_dir / "experiment_manifest.json"
    incremental_path = run_dir / "dataset_results.jsonl"
    artifact_root = run_dir / "tensorboard" / "artifacts" / f"step_{args.step}"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_names = ", ".join(item["name"] for item in manifest.get("datasets", []))
        print(f"run_dir: {run_dir}")
        print(f"status: {manifest.get('status', 'unknown')}")
        print(f"start_time: {manifest.get('start_time', 'unknown')}")
        print(f"elapsed_seconds: {manifest.get('elapsed_seconds', 'N/A')}")
        print(f"datasets: {dataset_names or 'unknown'}")
        print(
            "completed_datasets: "
            f"{manifest.get('completed_dataset_count', 'unknown')}/"
            f"{len(manifest.get('datasets', []))}"
        )
        if manifest.get("error"):
            print(f"error: {manifest['error'].get('type')}: {manifest['error'].get('message')}")
        print()

    rows = []
    if incremental_path.is_file():
        for line_number, line in enumerate(
            incremental_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {incremental_path}:{line_number}"
                ) from exc
            rows.append(
                (
                    str(payload.get("dataset", "unknown")),
                    payload.get("spec", {}),
                    payload.get("confidence_summary") or {},
                )
            )

    if not rows:
        for path in sorted(artifact_root.glob("*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                (
                    path.parent.name,
                    payload.get("spec", {}),
                    payload.get("confidence_summary") or {},
                )
            )
    if not rows:
        if manifest_path.is_file() or incremental_path.is_file():
            print("No dataset has completed yet.")
            return
        raise FileNotFoundError(f"No result files found under {run_dir}")

    header = (
        f"{'dataset':<18} {'samples':>8} {'propose':>10} {'accept_len':>11} "
        f"{'verify':>10} {'ECE':>8} {'AUC':>8} {'Brier':>8}"
    )
    print(header)
    print("-" * len(header))
    for dataset_name, spec, confidence in rows:
        print(
            f"{dataset_name:<18} "
            f"{fmt(spec.get('num_samples')):>8} "
            f"{fmt(spec.get('draft_tokens_per_proposal')):>10} "
            f"{fmt(spec.get('acceptance_length')):>11} "
            f"{fmt(spec.get('verify_rate')):>10} "
            f"{fmt(confidence.get('ece_mean')):>8} "
            f"{fmt(confidence.get('auc_mean')):>8} "
            f"{fmt(confidence.get('brier_mean')):>8}"
        )


if __name__ == "__main__":
    main()
