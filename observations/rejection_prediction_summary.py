"""Shared reporting helpers for rejection-prediction observation experiments."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MACRO_START = "<!-- rejection-prediction-macro-average:start -->"
MACRO_END = "<!-- rejection-prediction-macro-average:end -->"
COUNT_FIELDS = (
    "accepted_count",
    "rejected_count",
    "flagged_evaluable_count",
)
RATIO_FIELDS = (
    "accepted_share_among_flagged",
    "rejected_share_among_flagged",
    "accepted_flag_rate",
    "rejected_capture_rate",
    "flag_rate_among_evaluable",
)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_prediction_threshold_rows(
    *,
    thresholds: Sequence[float],
    threshold_labels: Sequence[str],
    accepted_counts: Sequence[int],
    rejected_counts: Sequence[int],
    eligible_accepted: int,
    eligible_rejected: int,
    monotonic: str,
) -> list[dict]:
    """Build common rows and validate cumulative threshold-count invariants."""
    lengths = {
        len(thresholds),
        len(threshold_labels),
        len(accepted_counts),
        len(rejected_counts),
    }
    if len(lengths) != 1:
        raise ValueError("Threshold values, labels, and count vectors must have equal length")
    if monotonic not in ("nondecreasing", "nonincreasing"):
        raise ValueError(f"Unsupported monotonic direction: {monotonic!r}")

    rows = []
    eligible_evaluable = int(eligible_accepted) + int(eligible_rejected)
    previous_accepted: int | None = None
    previous_rejected: int | None = None
    for threshold, label, accepted_value, rejected_value in zip(
        thresholds,
        threshold_labels,
        accepted_counts,
        rejected_counts,
        strict=True,
    ):
        accepted_count = int(accepted_value)
        rejected_count = int(rejected_value)
        if not 0 <= accepted_count <= eligible_accepted:
            raise RuntimeError(
                f"Accepted threshold count {accepted_count} is outside denominator "
                f"{eligible_accepted}"
            )
        if not 0 <= rejected_count <= eligible_rejected:
            raise RuntimeError(
                f"Rejected threshold count {rejected_count} is outside denominator "
                f"{eligible_rejected}"
            )
        if previous_accepted is not None:
            if monotonic == "nondecreasing" and accepted_count < previous_accepted:
                raise RuntimeError("Accepted threshold counts are not monotone non-decreasing")
            if monotonic == "nonincreasing" and accepted_count > previous_accepted:
                raise RuntimeError("Accepted threshold counts are not monotone non-increasing")
        if previous_rejected is not None:
            if monotonic == "nondecreasing" and rejected_count < previous_rejected:
                raise RuntimeError("Rejected threshold counts are not monotone non-decreasing")
            if monotonic == "nonincreasing" and rejected_count > previous_rejected:
                raise RuntimeError("Rejected threshold counts are not monotone non-increasing")
        previous_accepted = accepted_count
        previous_rejected = rejected_count
        flagged = accepted_count + rejected_count
        rows.append(
            {
                "threshold": float(threshold),
                "threshold_label": str(label),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "flagged_evaluable_count": flagged,
                "accepted_share_among_flagged": ratio(accepted_count, flagged),
                "rejected_share_among_flagged": ratio(rejected_count, flagged),
                "accepted_flag_rate": ratio(accepted_count, eligible_accepted),
                "rejected_capture_rate": ratio(rejected_count, eligible_rejected),
                "flag_rate_among_evaluable": ratio(flagged, eligible_evaluable),
            }
        )
    return rows


def markdown_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.4f}%"


def markdown_threshold_table(
    *,
    title: str,
    threshold_name: str,
    rows: Sequence[dict],
    macro: bool = False,
    dataset_count: int | None = None,
) -> list[str]:
    count_suffix = "_mean" if macro else ""
    ratio_suffix = "_macro_mean" if macro else ""
    lines = [
        f"### {title}",
        "",
        (
            f"| {threshold_name} | accepted_count{count_suffix} | "
            f"rejected_count{count_suffix} | flagged_evaluable_count{count_suffix} | "
            f"accepted_share{ratio_suffix} | rejected_share / precision{ratio_suffix} | "
            f"accepted_FPR{ratio_suffix} | rejection_recall{ratio_suffix} | "
            f"flag_rate{ratio_suffix} |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if macro:
            if dataset_count is None:
                raise ValueError("dataset_count is required for a macro-average table")
            counts = [f"{float(row[field]):.4f}" for field in COUNT_FIELDS]
            ratios = []
            for field in RATIO_FIELDS:
                value = row[field]
                defined = int(row[f"{field}_defined_dataset_count"])
                rendered = markdown_ratio(value)
                ratios.append(f"{rendered} ({defined}/{dataset_count})")
        else:
            counts = [str(int(row[field])) for field in COUNT_FIELDS]
            ratios = [markdown_ratio(row[field]) for field in RATIO_FIELDS]
        lines.append(
            "| {threshold} | {accepted} | {rejected} | {flagged} | "
            "{accepted_share} | {rejected_share} | {accepted_fpr} | "
            "{rejection_recall} | {flag_rate} |".format(
                threshold=row["threshold_label"],
                accepted=counts[0],
                rejected=counts[1],
                flagged=counts[2],
                accepted_share=ratios[0],
                rejected_share=ratios[1],
                accepted_fpr=ratios[2],
                rejection_recall=ratios[3],
                flag_rate=ratios[4],
            )
        )
    lines.append("")
    return lines


def build_macro_rows(dataset_rows: Sequence[Sequence[dict]]) -> list[dict]:
    """Arithmetic per-dataset mean at each threshold (never pooled/micro)."""
    if not dataset_rows:
        return []
    row_count = len(dataset_rows[0])
    if any(len(rows) != row_count for rows in dataset_rows):
        raise ValueError("Datasets do not share the same number of thresholds")
    result = []
    dataset_count = len(dataset_rows)
    for index in range(row_count):
        source_rows = [rows[index] for rows in dataset_rows]
        labels = {str(row["threshold_label"]) for row in source_rows}
        values = [float(row["threshold"]) for row in source_rows]
        if len(labels) != 1 or max(values) - min(values) > 1e-12:
            raise ValueError(f"Datasets have mismatched thresholds at row {index}")
        row = {
            "threshold": values[0],
            "threshold_label": source_rows[0]["threshold_label"],
        }
        for field in COUNT_FIELDS:
            row[field] = sum(float(source[field]) for source in source_rows) / dataset_count
        for field in RATIO_FIELDS:
            defined_values = [
                float(source[field])
                for source in source_rows
                if source.get(field) is not None
            ]
            row[field] = (
                sum(defined_values) / len(defined_values) if defined_values else None
            )
            row[f"{field}_defined_dataset_count"] = len(defined_values)
        result.append(row)
    return result


def build_macro_section(
    *,
    summaries: Mapping[str, dict],
    family_specs: Sequence[tuple[str, str, str]],
) -> str:
    """Render the final cross-dataset macro-average tables."""
    ordered = list(summaries.items())
    dataset_count = len(ordered)
    if not ordered:
        return ""
    lines = [
        MACRO_START,
        "## All-dataset macro average",
        "",
        (
            f"- Dataset count: `{dataset_count}`; datasets: "
            + ", ".join(f"`{name}`" for name, _ in ordered)
        ),
        (
            "- Averaging rule: first compute every metric independently within each "
            "dataset, then take the arithmetic mean across datasets. This is a macro "
            "average, not a pooled-token (micro) average."
        ),
        (
            "- Count columns are also per-dataset arithmetic means and may be fractional. "
            "For ratio columns, `(m/n)` gives the number of datasets with a defined "
            "denominator over the total dataset count; undefined 0/0 values are excluded "
            "from that ratio's mean."
        ),
        "",
    ]
    for title, threshold_name, summary_key in family_specs:
        rows = build_macro_rows([summary[summary_key] for _, summary in ordered])
        lines.extend(
            markdown_threshold_table(
                title=title,
                threshold_name=threshold_name,
                rows=rows,
                macro=True,
                dataset_count=dataset_count,
            )
        )
    lines.append(MACRO_END)
    return "\n".join(lines) + "\n"


def strip_macro_section(text: str) -> str:
    start = text.find(MACRO_START)
    end = text.find(MACRO_END)
    if start < 0 and end < 0:
        return text.rstrip() + "\n"
    if start < 0 or end < start:
        raise ValueError("Malformed macro-average marker block in Markdown")
    end += len(MACRO_END)
    if text[end:].strip():
        raise ValueError("Macro-average block must be the final Markdown section")
    return text[:start].rstrip() + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_dataset_and_refresh_macro(
    *,
    path: Path,
    dataset_section: str,
    summaries: Mapping[str, dict],
    family_specs: Sequence[tuple[str, str, str]],
) -> None:
    base = strip_macro_section(path.read_text(encoding="utf-8")) if path.exists() else ""
    text = base.rstrip() + "\n\n" + dataset_section.strip() + "\n\n"
    text += build_macro_section(summaries=summaries, family_specs=family_specs)
    _write_text_atomic(path, text)


def refresh_macro_only(
    *,
    path: Path,
    summaries: Mapping[str, dict],
    family_specs: Sequence[tuple[str, str, str]],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Markdown results file not found: {path}")
    base = strip_macro_section(path.read_text(encoding="utf-8"))
    text = base.rstrip() + "\n\n"
    text += build_macro_section(summaries=summaries, family_specs=family_specs)
    _write_text_atomic(path, text)


def build_markdown_header(
    *,
    title: str,
    score_definition: str,
    comparison: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Score: {score_definition}",
        f"- Flag condition: `{comparison}`.",
        (
            "- Outcomes: accepted positions and the first rejected/replaced position are "
            "evaluable; positions after that rejection and positions discarded after an "
            "accepted EOS are excluded."
        ),
        "- Dataset position 0 is included because this direct-score rule needs no prior-token mean.",
        "- This file is updated after every completed dataset.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_header_exclusive(
    *,
    path: Path,
    title: str,
    score_definition: str,
    comparison: str,
) -> None:
    text = build_markdown_header(
        title=title,
        score_definition=score_definition,
        comparison=comparison,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def rebuild_direct_markdown(
    *,
    path: Path,
    title: str,
    score_definition: str,
    comparison: str,
    dataset_records: Sequence[tuple[str, str, dict]],
    table_title: str,
) -> None:
    """Idempotently rebuild a direct-threshold Markdown file from exact summaries."""
    if not dataset_records:
        raise ValueError("No dataset records are available for Markdown rebuilding")
    summaries = {name: summary for name, _, summary in dataset_records}
    parts = [
        build_markdown_header(
            title=title,
            score_definition=score_definition,
            comparison=comparison,
        ).rstrip()
    ]
    for dataset_name, completed_at, summary in dataset_records:
        parts.append(
            build_direct_dataset_section(
                dataset_name=dataset_name,
                completed_at=completed_at,
                summary=summary,
                table_title=table_title,
            ).strip()
        )
    parts.append(
        build_macro_section(
            summaries=summaries,
            family_specs=((table_title, "threshold", "thresholds"),),
        ).strip()
    )
    _write_text_atomic(path, "\n\n".join(parts) + "\n")


def build_direct_dataset_section(
    *,
    dataset_name: str,
    completed_at: str,
    summary: dict,
    table_title: str,
    threshold_name: str = "threshold",
) -> str:
    counts = summary["counts"]
    lines = [
        f"## Dataset: {dataset_name}",
        "",
        f"- Completed at: `{completed_at}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Verification rounds: `{counts['verification_rounds']}`",
        (
            "- Eligible accepted/rejected denominators: "
            f"`{counts['eligible_accepted_token_count']}` / "
            f"`{counts['eligible_rejected_token_count']}`"
        ),
        (
            "- Ignored after first rejection / accepted EOS: "
            f"`{counts['ignored_after_first_rejection_token_count']}` / "
            f"`{counts['ignored_after_accepted_eos_token_count']}`"
        ),
        "",
    ]
    lines.extend(
        markdown_threshold_table(
            title=table_title,
            threshold_name=threshold_name,
            rows=summary["thresholds"],
        )
    )
    return "\n".join(lines) + "\n"


def validate_score_thresholds(values: Iterable[str | float]) -> tuple[tuple[float, ...], tuple[str, ...]]:
    thresholds: list[float] = []
    labels: list[str] = []
    previous: float | None = None
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Score threshold must be finite and in [0, 1], got {raw!r}")
        if previous is not None and value <= previous:
            raise ValueError("Score thresholds must be strictly increasing")
        label = str(raw)
        thresholds.append(value)
        labels.append(label)
        previous = value
    if not thresholds:
        raise ValueError("At least one score threshold is required")
    return tuple(thresholds), tuple(labels)
