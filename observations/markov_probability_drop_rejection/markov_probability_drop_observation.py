"""Evaluate whether a selected-token Markov probability drop predicts rejection.

For draft position i >= 1, this experiment compares ``P_i = q_i[z_i]`` with
the mean selected-token draft probability of positions 0..i-1 from the same
draft forward pass.  Here ``q_i = softmax(markov_corrected_logits_i)`` without
temperature scaling and z_i is the submitted token.  This diagnostic
distribution is separate from the operational distribution used by
speculative verification.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

from deepspec.eval.base_evaluator import VerificationResult
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.utils import CustomJSONEncoder
from observations.markov_diagnostic_draft import (
    DiagnosticMarkovDraftProposal,
    DiagnosticMarkovProposalMixin,
)
from observations.rejection_prediction_summary import (
    append_dataset_and_refresh_macro,
)


SCHEMA_VERSION = 2
ABSOLUTE_FAMILY = "absolute_drop"
PERCENTAGE_FAMILY = "percentage_drop"
SCALAR_COUNT_KEYS = (
    "verification_rounds",
    "proposal_rounds",
    "zero_draft_proposal_rounds",
    "generated_draft_token_count",
    "total_accepted_draft_token_count",
    "fully_accepted_proposal_rounds",
    "accepted_eos_rounds",
    "correction_events",
    "eligible_accepted_token_count",
    "eligible_rejected_token_count",
    "unscorable_position0_accepted_count",
    "unscorable_first_position_rejections",
    "ignored_after_first_rejection_token_count",
    "ignored_after_accepted_eos_token_count",
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return int(numerator) / int(denominator)


def _threshold_label(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    integer, separator, fraction = text.partition(".")
    if not separator:
        fraction = ""
    return f"{integer}.{fraction.ljust(3, '0')}"


def _validate_thresholds(values: Iterable[float], family: str) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError(f"{family} thresholds must not be empty")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError(f"{family} thresholds must be finite and in [0, 1]")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError(f"{family} thresholds must be strictly increasing")
    return thresholds


def _validate_count_invariants(counts: dict[str, int]) -> None:
    expected_generated = (
        counts["total_accepted_draft_token_count"]
        + counts["correction_events"]
        + counts["ignored_after_first_rejection_token_count"]
        + counts["ignored_after_accepted_eos_token_count"]
    )
    if counts["generated_draft_token_count"] != expected_generated:
        raise RuntimeError(
            "Generated draft-token accounting mismatch: "
            f"{counts['generated_draft_token_count']} != {expected_generated}"
        )
    expected_eligible_accepted = (
        counts["total_accepted_draft_token_count"]
        - counts["unscorable_position0_accepted_count"]
    )
    if counts["eligible_accepted_token_count"] != expected_eligible_accepted:
        raise RuntimeError(
            "Eligible accepted-token accounting mismatch: "
            f"{counts['eligible_accepted_token_count']} != {expected_eligible_accepted}"
        )
    expected_eligible_rejected = (
        counts["correction_events"]
        - counts["unscorable_first_position_rejections"]
    )
    if counts["eligible_rejected_token_count"] != expected_eligible_rejected:
        raise RuntimeError(
            "Eligible rejected-token accounting mismatch: "
            f"{counts['eligible_rejected_token_count']} != {expected_eligible_rejected}"
        )


class MarkovProbabilityDropMetrics:
    """Rank-local counters, with threshold comparisons retained on device."""

    def __init__(
        self,
        *,
        device: torch.device,
        absolute_thresholds: Iterable[float],
        percentage_thresholds: Iterable[float],
    ) -> None:
        self.device = device
        self.absolute_threshold_values = _validate_thresholds(
            absolute_thresholds,
            ABSOLUTE_FAMILY,
        )
        self.percentage_threshold_values = _validate_thresholds(
            percentage_thresholds,
            PERCENTAGE_FAMILY,
        )
        self.absolute_thresholds = torch.tensor(
            self.absolute_threshold_values,
            dtype=torch.float32,
            device=device,
        )
        self.percentage_thresholds = torch.tensor(
            self.percentage_threshold_values,
            dtype=torch.float32,
            device=device,
        )
        self.absolute_accepted = torch.zeros(
            len(self.absolute_threshold_values),
            dtype=torch.int64,
            device=device,
        )
        self.absolute_rejected = torch.zeros_like(self.absolute_accepted)
        self.percentage_accepted = torch.zeros(
            len(self.percentage_threshold_values),
            dtype=torch.int64,
            device=device,
        )
        self.percentage_rejected = torch.zeros_like(self.percentage_accepted)
        # [accepted, rejected] counts for which P_i_mean is numerically zero.
        self.percentage_undefined = torch.zeros(2, dtype=torch.int64, device=device)
        self.counts = Counter({key: 0 for key in SCALAR_COUNT_KEYS})

    def observe(
        self,
        *,
        proposal: DiagnosticMarkovDraftProposal,
        verification: VerificationResult,
    ) -> None:
        self.counts["verification_rounds"] += 1
        proposal_length = int(proposal.draft_token_count)
        if proposal_length <= 0:
            self.counts["zero_draft_proposal_rounds"] += 1
            return

        self.counts["proposal_rounds"] += 1
        self.counts["generated_draft_token_count"] += proposal_length
        diagnostic_logits = proposal.diagnostic_markov_logits
        if diagnostic_logits is None:
            raise RuntimeError(
                "The Markov probability-drop experiment requires diagnostic Markov logits"
            )
        if diagnostic_logits.ndim != 3 or diagnostic_logits.size(0) != 1:
            raise ValueError(
                "diagnostic_markov_logits must have shape "
                "[1, proposal_length, vocab_size], got "
                f"{tuple(diagnostic_logits.shape)}"
            )
        if diagnostic_logits.size(1) < proposal_length:
            raise ValueError(
                "diagnostic_markov_logits is shorter than draft_token_count"
            )
        diagnostic_logits = diagnostic_logits[:, :proposal_length, :].detach().float()
        if not bool(torch.isfinite(diagnostic_logits).all().item()):
            raise ValueError("Diagnostic Markov logits must be finite")
        diagnostic_probabilities = torch.softmax(diagnostic_logits, dim=-1)
        if proposal.verify_input_ids.ndim != 2 or proposal.verify_input_ids.size(0) != 1:
            raise ValueError(
                "verify_input_ids must have shape [1, proposal_length + 1], got "
                f"{tuple(proposal.verify_input_ids.shape)}"
            )
        if proposal.verify_input_ids.size(1) < proposal_length + 1:
            raise ValueError("verify_input_ids is shorter than draft_token_count + 1")

        accepted_count = int(verification.accepted_draft_tokens)
        effective_length = int(verification.effective_proposal_length)
        if accepted_count < 0 or accepted_count > proposal_length:
            raise ValueError(
                f"accepted_draft_tokens={accepted_count} is incompatible with "
                f"proposal_length={proposal_length}"
            )
        if effective_length < 0 or effective_length > proposal_length:
            raise ValueError(
                f"effective_proposal_length={effective_length} is incompatible "
                f"with proposal_length={proposal_length}"
            )

        self.counts["total_accepted_draft_token_count"] += accepted_count
        eligible_accepted = max(accepted_count - 1, 0)
        self.counts["eligible_accepted_token_count"] += eligible_accepted
        if accepted_count > 0:
            self.counts["unscorable_position0_accepted_count"] += 1

        correction_event = (
            not bool(verification.terminated_by_stop_token)
            and accepted_count < proposal_length
        )
        eligible_rejected = 0
        if verification.terminated_by_stop_token:
            if effective_length != accepted_count:
                raise ValueError(
                    "An accepted-EOS round must have effective_proposal_length "
                    "equal to accepted_draft_tokens"
                )
            self.counts["accepted_eos_rounds"] += 1
            self.counts["ignored_after_accepted_eos_token_count"] += (
                proposal_length - effective_length
            )
        elif correction_event:
            self.counts["correction_events"] += 1
            self.counts["ignored_after_first_rejection_token_count"] += (
                proposal_length - accepted_count - 1
            )
            if accepted_count == 0:
                self.counts["unscorable_first_position_rejections"] += 1
            else:
                eligible_rejected = 1
                self.counts["eligible_rejected_token_count"] += 1
        else:
            if accepted_count != proposal_length:
                raise ValueError(
                    "A non-EOS, non-correction round must accept the full proposal"
                )
            self.counts["fully_accepted_proposal_rounds"] += 1

        if proposal_length <= 1:
            return

        submitted_token_ids = proposal.verify_input_ids[
            0, 1 : proposal_length + 1
        ].long()
        selected_probabilities = diagnostic_probabilities[
            0, :proposal_length, :
        ].gather(
            dim=-1,
            index=submitted_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.isfinite(selected_probabilities).all().item()):
            raise ValueError("Selected Markov draft probabilities must be finite")
        if bool(
            ((selected_probabilities < 0.0) | (selected_probabilities > 1.0))
            .any()
            .item()
        ):
            raise ValueError("Selected Markov draft probabilities must be in [0, 1]")
        prior_means = selected_probabilities.cumsum(dim=0)[:-1] / torch.arange(
            1,
            proposal_length,
            dtype=selected_probabilities.dtype,
            device=selected_probabilities.device,
        )
        current_probabilities = selected_probabilities[1:]
        absolute_drops = prior_means - current_probabilities
        percentage_valid = prior_means > 0.0
        safe_prior_means = prior_means.clamp_min(torch.finfo(prior_means.dtype).tiny)
        percentage_drops = (
            1.0 - current_probabilities / safe_prior_means
        ).clamp_(min=0.0, max=1.0)

        if eligible_accepted > 0:
            accepted_absolute = absolute_drops[:eligible_accepted]
            self.absolute_accepted.add_(
                (
                    accepted_absolute[:, None]
                    >= self.absolute_thresholds[None, :]
                ).sum(dim=0, dtype=torch.int64)
            )
            accepted_valid = percentage_valid[:eligible_accepted]
            accepted_percentage = percentage_drops[:eligible_accepted]
            self.percentage_accepted.add_(
                (
                    (accepted_percentage[:, None] >= self.percentage_thresholds[None, :])
                    & accepted_valid[:, None]
                ).sum(dim=0, dtype=torch.int64)
            )
            self.percentage_undefined[0].add_(
                (~accepted_valid).sum(dtype=torch.int64)
            )

        if eligible_rejected:
            score_index = accepted_count - 1
            rejected_absolute = absolute_drops[score_index]
            self.absolute_rejected.add_(
                (rejected_absolute >= self.absolute_thresholds).to(torch.int64)
            )
            rejected_valid = percentage_valid[score_index]
            rejected_percentage = percentage_drops[score_index]
            self.percentage_rejected.add_(
                (
                    (rejected_percentage >= self.percentage_thresholds)
                    & rejected_valid
                ).to(torch.int64)
            )
            self.percentage_undefined[1].add_(
                (~rejected_valid).to(torch.int64)
            )

    def local_payload(self) -> dict:
        absolute_accepted = self.absolute_accepted.detach().cpu().tolist()
        absolute_rejected = self.absolute_rejected.detach().cpu().tolist()
        percentage_accepted = self.percentage_accepted.detach().cpu().tolist()
        percentage_rejected = self.percentage_rejected.detach().cpu().tolist()
        percentage_undefined = self.percentage_undefined.detach().cpu().tolist()
        return {
            "counts": {key: int(self.counts[key]) for key in SCALAR_COUNT_KEYS},
            "percentage_undefined": {
                "accepted": int(percentage_undefined[0]),
                "rejected": int(percentage_undefined[1]),
            },
            "absolute_thresholds": [
                {
                    "threshold": value,
                    "threshold_label": _threshold_label(value),
                    "accepted_count": int(absolute_accepted[index]),
                    "rejected_count": int(absolute_rejected[index]),
                }
                for index, value in enumerate(self.absolute_threshold_values)
            ],
            "percentage_thresholds": [
                {
                    "threshold": value,
                    "threshold_label": _threshold_label(value),
                    "accepted_count": int(percentage_accepted[index]),
                    "rejected_count": int(percentage_rejected[index]),
                }
                for index, value in enumerate(self.percentage_threshold_values)
            ],
        }

    def reduce(self) -> dict:
        use_cpu = str(dist.get_backend()).lower() == "gloo"
        reduction_device = torch.device("cpu") if use_cpu else self.device
        scalar_counts = torch.tensor(
            [int(self.counts[key]) for key in SCALAR_COUNT_KEYS],
            dtype=torch.int64,
            device=reduction_device,
        )
        tensors = {
            "scalar_counts": scalar_counts,
            "absolute_accepted": self.absolute_accepted.detach().to(reduction_device),
            "absolute_rejected": self.absolute_rejected.detach().to(reduction_device),
            "percentage_accepted": self.percentage_accepted.detach().to(reduction_device),
            "percentage_rejected": self.percentage_rejected.detach().to(reduction_device),
            "percentage_undefined": self.percentage_undefined.detach().to(reduction_device),
        }
        for tensor in tensors.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {
            "counts": {
                key: int(tensors["scalar_counts"][index].item())
                for index, key in enumerate(SCALAR_COUNT_KEYS)
            },
            "absolute_accepted": tensors["absolute_accepted"].cpu().tolist(),
            "absolute_rejected": tensors["absolute_rejected"].cpu().tolist(),
            "percentage_accepted": tensors["percentage_accepted"].cpu().tolist(),
            "percentage_rejected": tensors["percentage_rejected"].cpu().tolist(),
            "percentage_undefined": tensors["percentage_undefined"].cpu().tolist(),
        }

    def build_report(
        self,
        *,
        dataset_name: str,
        sample_count: int,
        reduced: dict,
    ) -> dict:
        counts = {key: int(reduced["counts"][key]) for key in SCALAR_COUNT_KEYS}
        _validate_count_invariants(counts)
        absolute_accepted_denominator = counts["eligible_accepted_token_count"]
        absolute_rejected_denominator = counts["eligible_rejected_token_count"]
        percentage_undefined_accepted = int(reduced["percentage_undefined"][0])
        percentage_undefined_rejected = int(reduced["percentage_undefined"][1])
        percentage_accepted_denominator = (
            absolute_accepted_denominator - percentage_undefined_accepted
        )
        percentage_rejected_denominator = (
            absolute_rejected_denominator - percentage_undefined_rejected
        )
        if percentage_accepted_denominator < 0 or percentage_rejected_denominator < 0:
            raise RuntimeError("Percentage undefined counts exceed eligible denominators")

        absolute_rows = build_threshold_rows(
            thresholds=self.absolute_threshold_values,
            accepted_counts=reduced["absolute_accepted"],
            rejected_counts=reduced["absolute_rejected"],
            eligible_accepted=absolute_accepted_denominator,
            eligible_rejected=absolute_rejected_denominator,
        )
        percentage_rows = build_threshold_rows(
            thresholds=self.percentage_threshold_values,
            accepted_counts=reduced["percentage_accepted"],
            rejected_counts=reduced["percentage_rejected"],
            eligible_accepted=percentage_accepted_denominator,
            eligible_rejected=percentage_rejected_denominator,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_name,
            "sample_count": int(sample_count),
            "created_at": _now_iso(),
            "definitions": {
                "selected_token_markov_draft_probability": (
                    "P_i = q_obs_i[z_i], where q_obs_i = "
                    "softmax(markov_corrected_logits_i) without temperature scaling "
                    "and z_i is the submitted draft token"
                ),
                "operational_distribution_separation": (
                    "speculative verification continues to use the temperature-dependent "
                    "operational proposal.draft_probs; q_obs_i is diagnostic only"
                ),
                "semantic_note": (
                    "P_i is draft-model probability mass for the submitted token, "
                    "not confidence-head acceptance probability"
                ),
                "prior_mean": (
                    "P_i_mean = mean(P_0, ..., P_{i-1}); only defined for i >= 1"
                ),
                "absolute_drop": "P_i_mean - P_i",
                "percentage_drop": (
                    "max(0, 1 - P_i / P_i_mean); undefined when P_i_mean == 0"
                ),
                "threshold_comparison": "drop >= threshold (inclusive)",
                "accepted_label": "draft position i < accepted_draft_tokens",
                "rejected_label": (
                    "the first failed/replaced position i == accepted_draft_tokens "
                    "in a non-EOS correction round"
                ),
                "censoring": (
                    "position 0, positions after the first rejection, and positions "
                    "after an accepted EOS are excluded from threshold outcomes"
                ),
                "accepted_share_among_flagged": "flagged accepted / all flagged evaluable",
                "rejected_share_among_flagged": "flagged rejected / all flagged evaluable (prediction precision)",
                "accepted_flag_rate": "flagged accepted / all eligible accepted (false-positive rate)",
                "rejected_capture_rate": "flagged rejected / all eligible rejected (recall)",
            },
            "counts": {
                **counts,
                "eligible_evaluable_token_count": (
                    absolute_accepted_denominator + absolute_rejected_denominator
                ),
                "percentage_undefined_accepted_count": percentage_undefined_accepted,
                "percentage_undefined_rejected_count": percentage_undefined_rejected,
                "percentage_eligible_accepted_token_count": percentage_accepted_denominator,
                "percentage_eligible_rejected_token_count": percentage_rejected_denominator,
            },
            "absolute_drop": {
                "eligible_accepted_denominator": absolute_accepted_denominator,
                "eligible_rejected_denominator": absolute_rejected_denominator,
                "thresholds": absolute_rows,
            },
            "percentage_drop": {
                "eligible_accepted_denominator": percentage_accepted_denominator,
                "eligible_rejected_denominator": percentage_rejected_denominator,
                "thresholds": percentage_rows,
            },
        }


def build_threshold_rows(
    *,
    thresholds: Iterable[float],
    accepted_counts: Iterable[int],
    rejected_counts: Iterable[int],
    eligible_accepted: int,
    eligible_rejected: int,
) -> list[dict]:
    rows = []
    eligible_evaluable = int(eligible_accepted) + int(eligible_rejected)
    previous_accepted = None
    previous_rejected = None
    for threshold, accepted_value, rejected_value in zip(
        thresholds,
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
        if previous_accepted is not None and accepted_count > previous_accepted:
            raise RuntimeError("Accepted threshold counts are not monotone non-increasing")
        if previous_rejected is not None and rejected_count > previous_rejected:
            raise RuntimeError("Rejected threshold counts are not monotone non-increasing")
        previous_accepted = accepted_count
        previous_rejected = rejected_count
        flagged_evaluable = accepted_count + rejected_count
        rows.append(
            {
                "threshold": float(threshold),
                "threshold_label": _threshold_label(float(threshold)),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "flagged_evaluable_count": flagged_evaluable,
                "accepted_share_among_flagged": _ratio(
                    accepted_count,
                    flagged_evaluable,
                ),
                "rejected_share_among_flagged": _ratio(
                    rejected_count,
                    flagged_evaluable,
                ),
                "accepted_flag_rate": _ratio(
                    accepted_count,
                    eligible_accepted,
                ),
                "rejected_capture_rate": _ratio(
                    rejected_count,
                    eligible_rejected,
                ),
                "flag_rate_among_evaluable": _ratio(
                    flagged_evaluable,
                    eligible_evaluable,
                ),
            }
        )
    return rows


def summarize_drop_report(report: dict) -> dict:
    return {
        "sample_count": report["sample_count"],
        "counts": report["counts"],
        "absolute_drop_thresholds": report["absolute_drop"]["thresholds"],
        "percentage_drop_thresholds": report["percentage_drop"]["thresholds"],
    }


def _markdown_ratio(value) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.4f}%"


def _markdown_threshold_table(
    *,
    title: str,
    threshold_name: str,
    rows: list[dict],
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        (
            f"| {threshold_name} | accepted_count | rejected_count | "
            "flagged_evaluable_count | accepted_share | rejected_share / precision | "
            "accepted_FPR | rejection_recall | flag_rate |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {threshold} | {accepted} | {rejected} | {flagged} | "
            "{accepted_share} | {rejected_share} | {accepted_fpr} | "
            "{rejected_recall} | {flag_rate} |".format(
                threshold=row["threshold_label"],
                accepted=row["accepted_count"],
                rejected=row["rejected_count"],
                flagged=row["flagged_evaluable_count"],
                accepted_share=_markdown_ratio(
                    row["accepted_share_among_flagged"]
                ),
                rejected_share=_markdown_ratio(
                    row["rejected_share_among_flagged"]
                ),
                accepted_fpr=_markdown_ratio(row["accepted_flag_rate"]),
                rejected_recall=_markdown_ratio(row["rejected_capture_rate"]),
                flag_rate=_markdown_ratio(row["flag_rate_among_evaluable"]),
            )
        )
    lines.append("")
    return lines


def build_markdown_dataset_section(
    *,
    dataset_name: str,
    completed_at: str,
    summary: dict,
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
            "- Unscorable first-position rejections: "
            f"`{counts['unscorable_first_position_rejections']}`"
        ),
        (
            "- Ignored after first rejection / accepted EOS: "
            f"`{counts['ignored_after_first_rejection_token_count']}` / "
            f"`{counts['ignored_after_accepted_eos_token_count']}`"
        ),
        "",
    ]
    lines.extend(
        _markdown_threshold_table(
            title="token_x_drop_abs",
            threshold_name="x",
            rows=summary["absolute_drop_thresholds"],
        )
    )
    lines.extend(
        _markdown_threshold_table(
            title="token_y_drop_pct",
            threshold_name="y",
            rows=summary["percentage_drop_thresholds"],
        )
    )
    return "\n".join(lines) + "\n"


def append_markdown_dataset_result(
    *,
    path: Path,
    dataset_name: str,
    completed_at: str,
    summary: dict,
    all_summaries: dict[str, dict],
) -> None:
    section = build_markdown_dataset_section(
        dataset_name=dataset_name,
        completed_at=completed_at,
        summary=summary,
    )
    append_dataset_and_refresh_macro(
        path=path,
        dataset_section=section,
        summaries=all_summaries,
        family_specs=(
            ("token_x_drop_abs", "x", "absolute_drop_thresholds"),
            ("token_y_drop_pct", "y", "percentage_drop_thresholds"),
        ),
    )


def _write_threshold_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = (
        "threshold",
        "threshold_label",
        "accepted_count",
        "rejected_count",
        "flagged_evaluable_count",
        "accepted_share_among_flagged",
        "rejected_share_among_flagged",
        "accepted_flag_rate",
        "rejected_capture_rate",
        "flag_rate_among_evaluable",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_threshold_curves(dataset_dir: Path, report: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, family_key, title in (
        (axes[0], "absolute_drop", "absolute drop"),
        (axes[1], "percentage_drop", "percentage drop"),
    ):
        rows = report[family_key]["thresholds"]
        x_values = [row["threshold"] for row in rows]
        for metric, label in (
            ("rejected_share_among_flagged", "rejection precision"),
            ("accepted_share_among_flagged", "accepted share / false discovery"),
            ("rejected_capture_rate", "rejection recall"),
            ("accepted_flag_rate", "accepted flag rate / FPR"),
        ):
            y_values = [
                float("nan") if row[metric] is None else row[metric]
                for row in rows
            ]
            axis.plot(x_values, y_values, marker=".", linewidth=1.4, label=label)
        axis.set_title(title)
        axis.set_xlabel("drop threshold")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("ratio")
    fig.suptitle(f"{report['dataset']} Markov probability-drop rejection prediction")
    fig.tight_layout()
    output_path = dataset_dir / "threshold_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


class MarkovProbabilityDropRecorder:
    def __init__(
        self,
        *,
        device: torch.device,
        absolute_thresholds: Iterable[float],
        percentage_thresholds: Iterable[float],
        artifact_root: Path,
        tensorboard_dir: str | None,
        step: int | None,
    ) -> None:
        self.device = device
        self.absolute_thresholds = _validate_thresholds(
            absolute_thresholds,
            ABSOLUTE_FAMILY,
        )
        self.percentage_thresholds = _validate_thresholds(
            percentage_thresholds,
            PERCENTAGE_FAMILY,
        )
        self.artifact_root = Path(artifact_root)
        self.tensorboard_dir = tensorboard_dir
        self.step = step
        self.current: MarkovProbabilityDropMetrics | None = None
        self.rows: list[dict] = []

    def start(self) -> None:
        if self.current is not None:
            raise RuntimeError("Previous Markov probability-drop dataset was not finished")
        self.current = MarkovProbabilityDropMetrics(
            device=self.device,
            absolute_thresholds=self.absolute_thresholds,
            percentage_thresholds=self.percentage_thresholds,
        )

    def observe(
        self,
        *,
        proposal: DiagnosticMarkovDraftProposal,
        verification: VerificationResult,
    ) -> None:
        if self.current is None:
            raise RuntimeError("MarkovProbabilityDropRecorder.start() was not called")
        self.current.observe(proposal=proposal, verification=verification)

    def finish(
        self,
        *,
        dataset_name: str,
        metric_summary: dict,
        args_payload: dict,
        tasks: list,
    ) -> dict | None:
        if self.current is None:
            raise RuntimeError("MarkovProbabilityDropRecorder.start() was not called")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dataset_dir = self.artifact_root / dataset_name
        local_payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_name,
            "rank": rank,
            "world_size": world_size,
            "statistics": self.current.local_payload(),
        }
        _write_json_atomic(
            dataset_dir / "rank_stats" / f"rank_{rank}.json",
            local_payload,
        )
        dist.barrier()
        reduced = self.current.reduce()
        report = self.current.build_report(
            dataset_name=dataset_name,
            sample_count=int(metric_summary["sample_count"]),
            reduced=reduced,
        )
        self.current = None

        summary = None
        if rank == 0:
            summary = summarize_drop_report(report)
            output_payload = {
                "config": {"args": args_payload, "tasks": tasks},
                "spec_metric_summary": metric_summary,
                "markov_probability_drop_observation": report,
                "markov_probability_drop_observation_summary": summary,
            }
            _write_json_atomic(dataset_dir / "metrics.json", output_payload)
            _write_threshold_csv(
                dataset_dir / "absolute_drop_thresholds.csv",
                report["absolute_drop"]["thresholds"],
            )
            _write_threshold_csv(
                dataset_dir / "percentage_drop_thresholds.csv",
                report["percentage_drop"]["thresholds"],
            )
            _plot_threshold_curves(dataset_dir, report)
            self.rows.append(report)
            print(
                "Markov probability-drop observation counts: "
                + json.dumps(
                    {"dataset": dataset_name, **report["counts"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            print(f"Wrote Markov probability-drop artifacts to {dataset_dir}", flush=True)
        dist.barrier()
        return summary

    def log_tensorboard(self) -> None:
        if not self.rows or self.tensorboard_dir is None or self.step is None:
            return
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for report in self.rows:
            dataset = report["dataset"]
            for family_key in ("absolute_drop", "percentage_drop"):
                for row in report[family_key]["thresholds"]:
                    threshold_label = row["threshold_label"].replace(".", "p")
                    for metric in (
                        "accepted_share_among_flagged",
                        "rejected_share_among_flagged",
                        "accepted_flag_rate",
                        "rejected_capture_rate",
                    ):
                        value = row[metric]
                        if value is not None:
                            writer.add_scalar(
                                f"markov_probability_drop/{dataset}/{family_key}/{metric}_{threshold_label}",
                                float(value),
                                self.step,
                            )
        writer.close()

    def print_results(self) -> None:
        if dist.get_rank() != 0 or not self.rows:
            return
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = (
            "dataset",
            "samples",
            "rounds",
            "eligible_accept",
            "eligible_reject",
            "reject@pos0",
            "ignored_after_reject",
            "ignored_after_eos",
        )
        for report in self.rows:
            counts = report["counts"]
            table.add_row(
                (
                    report["dataset"],
                    report["sample_count"],
                    counts["verification_rounds"],
                    counts["eligible_accepted_token_count"],
                    counts["eligible_rejected_token_count"],
                    counts["unscorable_first_position_rejections"],
                    counts["ignored_after_first_rejection_token_count"],
                    counts["ignored_after_accepted_eos_token_count"],
                )
            )
        print("Markov probability-drop rejection-prediction denominators:", flush=True)
        print(table.get_string(), flush=True)


class MarkovProbabilityDropEvaluator(
    DiagnosticMarkovProposalMixin,
    Qwen3DSparkEvaluator,
):
    """Qwen3 DSpark evaluator with isolated Markov probability-drop counters."""

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        if self.draft_model.markov_head is None:
            raise RuntimeError("Draft checkpoint has no Markov correction head")
        self.markov_probability_drop_recorder = MarkovProbabilityDropRecorder(
            device=self.device,
            absolute_thresholds=args.absolute_drop_thresholds,
            percentage_thresholds=args.percentage_drop_thresholds,
            artifact_root=Path(args.observation_artifact_root),
            tensorboard_dir=args.tensorboard_dir,
            step=args.step,
        )
        self.markov_probability_drop_summaries: dict[str, dict] = {}

    def mark_dataset_started(self, dataset_name: str) -> None:
        super().mark_dataset_started(dataset_name)
        self.markov_probability_drop_recorder.start()

    def _post_verify(self, proposal, verification) -> None:
        # Preserve the parent recorder and add only deterministic counters.
        super()._post_verify(proposal, verification)
        if not isinstance(proposal, DiagnosticMarkovDraftProposal):
            raise TypeError(
                "Expected DiagnosticMarkovDraftProposal, got "
                f"{type(proposal)!r}"
            )
        self.markov_probability_drop_recorder.observe(
            proposal=proposal,
            verification=verification,
        )

    def record_dataset_metrics(self, *, dataset_name: str, metric_summary: dict):
        self.mark_dataset_phase(
            dataset_name,
            "reducing_and_writing_markov_probability_drop_observations",
        )
        summary = self.markov_probability_drop_recorder.finish(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
            args_payload=json.loads(
                json.dumps(vars(self.args), cls=CustomJSONEncoder)
            ),
            tasks=[list(task) for task in self.tasks],
        )
        if summary is not None:
            self.markov_probability_drop_summaries[dataset_name] = summary
        return super().record_dataset_metrics(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )

    def record_incremental_dataset_result(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_summary: dict | None = None,
    ) -> None:
        if dist.get_rank() != 0:
            return
        completed_at = _now_iso()
        dataset_name = str(metrics_row["dataset"])
        result = {
            "dataset": dataset_name,
            "completed_at": completed_at,
            "spec": metrics_row,
            "confidence_summary": confidence_summary,
            "markov_probability_drop_observation_summary": (
                self.markov_probability_drop_summaries.get(dataset_name)
            ),
        }
        markdown_value = getattr(self.args, "markdown_results_path", None)
        markdown_summary = self.markov_probability_drop_summaries.get(dataset_name)
        if markdown_value is not None and markdown_summary is not None:
            markdown_path = Path(markdown_value)
            append_markdown_dataset_result(
                path=markdown_path,
                dataset_name=dataset_name,
                completed_at=completed_at,
                summary=markdown_summary,
                all_summaries=self.markov_probability_drop_summaries,
            )
            print(
                f"Appended Markdown result and refreshed macro averages in {markdown_path}",
                flush=True,
            )
        results_value = getattr(self.args, "dataset_results_path", None)
        if results_value is not None:
            results_path = Path(results_value)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"Appended dataset result to {results_path}", flush=True)
        self._update_manifest_dataset(
            dataset_name,
            status="completed",
            phase="completed",
            completed_at=completed_at,
            result=result,
        )

    def log_tensorboard(self) -> None:
        super().log_tensorboard()
        self.markov_probability_drop_recorder.log_tensorboard()

    def print_results(self) -> None:
        super().print_results()
        self.markov_probability_drop_recorder.print_results()


def evaluation_worker(local_rank: int, args) -> None:
    if local_rank == 0:
        print(
            json.dumps(
                vars(args),
                indent=2,
                ensure_ascii=False,
                cls=CustomJSONEncoder,
            ),
            flush=True,
        )
    evaluator = MarkovProbabilityDropEvaluator(local_rank, args)
    try:
        evaluator.evaluate()
    finally:
        evaluator.clean_up()
