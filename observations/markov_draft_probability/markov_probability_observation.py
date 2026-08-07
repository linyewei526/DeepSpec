"""Observe selected-token probabilities from Markov-corrected draft distributions.

For draft position ``k``, the recorded scalar is ``q_k[z_k]``, where ``q_k`` is
the complete operational draft distribution obtained by applying softmax (and
the configured sampling temperature) to the Markov-corrected logits, and
``z_k`` is the token actually proposed at that position.  This is a draft-model
token probability, not a confidence-head acceptance prediction and not a
cumulative product over positions.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

from deepspec.eval.base_evaluator import VerificationResult
from deepspec.eval.dspark.draft_ops import DSparkDraftProposal
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.utils import CustomJSONEncoder


SCHEMA_VERSION = 2
BIN_WIDTH = 0.05
PROBABILITY_MIN_BIN = 0
PROBABILITY_MAX_BIN = 19
GAP_MIN_BIN = 0
GAP_MAX_BIN = 19
RANK_CATEGORIES = tuple([str(index) for index in range(1, 11)] + ["other"])


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


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


@dataclass
class HistogramAccumulator:
    """Exact count/sum plus a width-0.05 histogram keyed by integer bin."""

    count: int = 0
    value_sum: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    bins: Counter[int] = field(default_factory=Counter)

    def add(self, value: float, *, fixed_min: float | None = None, fixed_max: float | None = None) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Observation value must be finite, got {value!r}")
        if fixed_min is not None and value < fixed_min - 1e-7:
            raise ValueError(f"Observation value {value} is below {fixed_min}")
        if fixed_max is not None and value > fixed_max + 1e-7:
            raise ValueError(f"Observation value {value} is above {fixed_max}")
        value = max(value, fixed_min) if fixed_min is not None else value
        value = min(value, fixed_max) if fixed_max is not None else value
        index = math.floor(value / BIN_WIDTH)
        if fixed_max is not None and math.isclose(value, fixed_max, abs_tol=1e-12):
            index = math.ceil(fixed_max / BIN_WIDTH) - 1
        self.count += 1
        self.value_sum += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.bins[int(index)] += 1

    def merge(self, other: "HistogramAccumulator") -> None:
        self.count += int(other.count)
        self.value_sum += float(other.value_sum)
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        if other.maximum is not None:
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
        self.bins.update(other.bins)

    def to_payload(self) -> dict:
        return {
            "count": self.count,
            "value_sum": self.value_sum,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "bins": {str(index): count for index, count in sorted(self.bins.items())},
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "HistogramAccumulator":
        return cls(
            count=int(payload.get("count", 0)),
            value_sum=float(payload.get("value_sum", 0.0)),
            minimum=_finite_or_none(payload.get("minimum")),
            maximum=_finite_or_none(payload.get("maximum")),
            bins=Counter({int(key): int(value) for key, value in payload.get("bins", {}).items()}),
        )

    def report(
        self,
        *,
        min_bin: int | None = None,
        max_bin: int | None = None,
        include_zero: bool = False,
    ) -> dict:
        if min_bin is None or max_bin is None:
            if self.bins:
                observed_min = min(self.bins)
                observed_max = max(self.bins)
                min_bin = observed_min if min_bin is None else min_bin
                max_bin = observed_max if max_bin is None else max_bin
            else:
                min_bin = 0 if min_bin is None else min_bin
                max_bin = -1 if max_bin is None else max_bin
        if include_zero and max_bin >= min_bin:
            min_bin = min(min_bin, 0)
            max_bin = max(max_bin, 0)

        cumulative = 0
        rows = []
        for index in range(min_bin, max_bin + 1):
            bin_count = int(self.bins.get(index, 0))
            cumulative += bin_count
            lower = round(index * BIN_WIDTH, 10)
            upper = round((index + 1) * BIN_WIDTH, 10)
            rows.append(
                {
                    "bin_index": index,
                    "lower": lower,
                    "upper": upper,
                    "interval": f"[{lower:.2f}, {upper:.2f}{']' if index == max_bin else ')'}",
                    "count": bin_count,
                    "probability": bin_count / self.count if self.count else None,
                    "cumulative_count": cumulative,
                    "cdf": cumulative / self.count if self.count else None,
                }
            )
        return {
            "count": self.count,
            "mean": self.value_sum / self.count if self.count else None,
            "sum": self.value_sum,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "bin_width": BIN_WIDTH,
            "bins": rows,
        }


@dataclass
class RankAccumulator:
    count: int = 0
    correction_probability_sum: float = 0.0
    correction_probability_min: float | None = None
    correction_probability_max: float | None = None

    def add(self, correction_probability: float) -> None:
        value = float(correction_probability)
        if not math.isfinite(value) or value < 0.0 or value > 1.0 + 1e-6:
            raise ValueError(f"Invalid correction-token q_k probability: {value}")
        self.count += 1
        self.correction_probability_sum += value
        self.correction_probability_min = (
            value if self.correction_probability_min is None else min(self.correction_probability_min, value)
        )
        self.correction_probability_max = (
            value if self.correction_probability_max is None else max(self.correction_probability_max, value)
        )

    def merge(self, other: "RankAccumulator") -> None:
        self.count += other.count
        self.correction_probability_sum += other.correction_probability_sum
        if other.correction_probability_min is not None:
            self.correction_probability_min = (
                other.correction_probability_min
                if self.correction_probability_min is None
                else min(self.correction_probability_min, other.correction_probability_min)
            )
        if other.correction_probability_max is not None:
            self.correction_probability_max = (
                other.correction_probability_max
                if self.correction_probability_max is None
                else max(self.correction_probability_max, other.correction_probability_max)
            )

    def to_payload(self) -> dict:
        return {
            "count": self.count,
            "correction_probability_sum": self.correction_probability_sum,
            "correction_probability_min": self.correction_probability_min,
            "correction_probability_max": self.correction_probability_max,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "RankAccumulator":
        return cls(
            count=int(payload.get("count", 0)),
            correction_probability_sum=float(payload.get("correction_probability_sum", 0.0)),
            correction_probability_min=_finite_or_none(payload.get("correction_probability_min")),
            correction_probability_max=_finite_or_none(payload.get("correction_probability_max")),
        )


class DatasetObservation:
    """Rank-local sufficient statistics for one dataset."""

    COUNT_KEYS = (
        "verification_rounds",
        "proposal_rounds",
        "zero_draft_proposal_rounds",
        "fully_accepted_proposal_rounds",
        "accepted_eos_rounds",
        "correction_events",
        "first_position_correction_events",
        "gap_candidate_events",
        "paired_gap_events",
        "negative_gap_excluded_events",
        "undefined_relative_gap_events",
    )

    def __init__(self) -> None:
        self.counts = Counter({key: 0 for key in self.COUNT_KEYS})
        self.accepted_probability = HistogramAccumulator()
        self.rejected_probability = HistogramAccumulator()
        self.signed_gap = HistogramAccumulator()
        self.signed_relative_gap = HistogramAccumulator()
        self.true_draft_rank = {category: RankAccumulator() for category in RANK_CATEGORIES}

    def observe(self, proposal: DSparkDraftProposal, verification: VerificationResult) -> None:
        self.counts["verification_rounds"] += 1
        proposal_length = int(proposal.draft_token_count)
        if proposal_length <= 0:
            self.counts["zero_draft_proposal_rounds"] += 1
            return
        self.counts["proposal_rounds"] += 1

        draft_probs = proposal.draft_probs
        if draft_probs is None:
            raise RuntimeError(
                "The Markov draft-probability experiment requires full draft_probs"
            )
        if draft_probs.ndim != 3 or draft_probs.size(0) != 1:
            raise ValueError(
                "draft_probs must have shape [1, proposal_length, vocab_size], got "
                f"{tuple(draft_probs.shape)}"
            )
        if draft_probs.size(1) < proposal_length:
            raise ValueError("draft_probs is shorter than draft_token_count")
        if proposal.verify_input_ids.ndim != 2 or proposal.verify_input_ids.size(0) != 1:
            raise ValueError(
                "verify_input_ids must have shape [1, proposal_length + 1], got "
                f"{tuple(proposal.verify_input_ids.shape)}"
            )
        if proposal.verify_input_ids.size(1) < proposal_length + 1:
            raise ValueError("verify_input_ids is shorter than draft_token_count + 1")

        accepted_count = int(verification.accepted_draft_tokens)
        if accepted_count < 0 or accepted_count > proposal_length:
            raise ValueError(
                f"accepted_draft_tokens={accepted_count} is incompatible with proposal_length={proposal_length}"
            )
        submitted_token_ids = proposal.verify_input_ids[
            0, 1 : proposal_length + 1
        ].long()
        selected_probabilities = draft_probs[
            0, :proposal_length, :
        ].detach().float().gather(
            dim=-1,
            index=submitted_token_ids.unsqueeze(-1),
        ).squeeze(-1).cpu()
        if not bool(torch.isfinite(selected_probabilities).all().item()):
            raise ValueError("Selected Markov draft probabilities must be finite")
        if bool(
            ((selected_probabilities < 0.0) | (selected_probabilities > 1.0))
            .any()
            .item()
        ):
            raise ValueError("Selected Markov draft probabilities must be in [0, 1]")
        accepted_values = [
            float(value)
            for value in selected_probabilities[:accepted_count].tolist()
        ]
        for value in accepted_values:
            self.accepted_probability.add(value, fixed_min=0.0, fixed_max=1.0)

        if verification.terminated_by_stop_token:
            self.counts["accepted_eos_rounds"] += 1
            return
        if accepted_count >= proposal_length:
            self.counts["fully_accepted_proposal_rounds"] += 1
            return

        # Exactly the first rejected position is replaced by verification's
        # residual/AR token.  Later draft positions are never verified.
        self.counts["correction_events"] += 1
        rejected_value = float(selected_probabilities[accepted_count].item())
        self.rejected_probability.add(rejected_value, fixed_min=0.0, fixed_max=1.0)

        if accepted_count == 0:
            self.counts["first_position_correction_events"] += 1
        else:
            self.counts["gap_candidate_events"] += 1
            accepted_mean = sum(accepted_values) / len(accepted_values)
            signed_gap = accepted_mean - rejected_value
            if signed_gap < 0.0:
                # The experiment intentionally conditions every gap statistic on
                # accepted_mean >= rejected_probability.  Keep an explicit audit
                # count, but do not add this event to either gap distribution.
                self.counts["negative_gap_excluded_events"] += 1
            else:
                self.signed_gap.add(signed_gap, fixed_min=0.0, fixed_max=1.0)
                self.counts["paired_gap_events"] += 1
                if accepted_mean > 0.0:
                    self.signed_relative_gap.add(
                        signed_gap / accepted_mean,
                        fixed_min=0.0,
                        fixed_max=1.0,
                    )
                else:  # A selected softmax probability should be positive; retain an audit count.
                    self.counts["undefined_relative_gap_events"] += 1

        draft_probs = proposal.draft_probs
        if draft_probs is None or draft_probs.ndim != 3 or draft_probs.size(0) != 1:
            raise RuntimeError("Full Markov-corrected q_k is required for true_draft_rank")
        q_k = draft_probs[0, accepted_count, :].detach().float()
        correction_token_id = int(verification.next_token.reshape(-1)[0].item())
        if correction_token_id < 0 or correction_token_id >= q_k.numel():
            raise ValueError(
                f"Correction token id {correction_token_id} is outside q_k vocab {q_k.numel()}"
            )
        correction_probability = float(q_k[correction_token_id].item())
        # Competition rank: ties share a rank, determined against the complete q_k.
        true_rank = 1 + int(torch.count_nonzero(q_k > correction_probability).item())
        category = str(true_rank) if true_rank <= 10 else "other"
        self.true_draft_rank[category].add(correction_probability)

    def merge(self, other: "DatasetObservation") -> None:
        self.counts.update(other.counts)
        self.accepted_probability.merge(other.accepted_probability)
        self.rejected_probability.merge(other.rejected_probability)
        self.signed_gap.merge(other.signed_gap)
        self.signed_relative_gap.merge(other.signed_relative_gap)
        for category in RANK_CATEGORIES:
            self.true_draft_rank[category].merge(other.true_draft_rank[category])

    def to_payload(self) -> dict:
        return {
            "counts": dict(self.counts),
            "distributions": {
                "accepted_selected_draft_probability": self.accepted_probability.to_payload(),
                "rejected_selected_draft_probability": self.rejected_probability.to_payload(),
                "signed_absolute_gap": self.signed_gap.to_payload(),
                "signed_relative_gap": self.signed_relative_gap.to_payload(),
            },
            "true_draft_rank": {
                category: self.true_draft_rank[category].to_payload()
                for category in RANK_CATEGORIES
            },
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DatasetObservation":
        observation = cls()
        observation.counts = Counter(
            {key: int(payload.get("counts", {}).get(key, 0)) for key in cls.COUNT_KEYS}
        )
        distributions = payload.get("distributions", {})
        observation.accepted_probability = HistogramAccumulator.from_payload(
            distributions.get("accepted_selected_draft_probability", {})
        )
        observation.rejected_probability = HistogramAccumulator.from_payload(
            distributions.get("rejected_selected_draft_probability", {})
        )
        observation.signed_gap = HistogramAccumulator.from_payload(
            distributions.get("signed_absolute_gap", {})
        )
        observation.signed_relative_gap = HistogramAccumulator.from_payload(
            distributions.get("signed_relative_gap", {})
        )
        rank_payload = payload.get("true_draft_rank", {})
        observation.true_draft_rank = {
            category: RankAccumulator.from_payload(rank_payload.get(category, {}))
            for category in RANK_CATEGORIES
        }
        return observation

    def build_report(self, *, dataset_name: str, sample_count: int) -> dict:
        correction_events = int(self.counts["correction_events"])
        first_position_events = int(self.counts["first_position_correction_events"])
        gap_candidate_events = int(self.counts["gap_candidate_events"])
        paired_gap_events = int(self.counts["paired_gap_events"])
        negative_gap_excluded_events = int(self.counts["negative_gap_excluded_events"])
        undefined_relative_gap_events = int(self.counts["undefined_relative_gap_events"])
        if correction_events != first_position_events + gap_candidate_events:
            raise RuntimeError(
                "Gap count invariant failed: correction_events must equal "
                "first_position_correction_events + gap_candidate_events"
            )
        if gap_candidate_events != paired_gap_events + negative_gap_excluded_events:
            raise RuntimeError(
                "Gap count invariant failed: gap_candidate_events must equal "
                "paired_gap_events + negative_gap_excluded_events"
            )
        if self.signed_gap.count != paired_gap_events:
            raise RuntimeError(
                "Gap count invariant failed: signed_absolute_gap count must equal paired_gap_events"
            )
        if self.signed_relative_gap.count + undefined_relative_gap_events != paired_gap_events:
            raise RuntimeError(
                "Gap count invariant failed: signed_relative_gap count plus "
                "undefined_relative_gap_events must equal paired_gap_events"
            )
        probability_accepted = self.accepted_probability.report(
            min_bin=PROBABILITY_MIN_BIN,
            max_bin=PROBABILITY_MAX_BIN,
        )
        probability_rejected = self.rejected_probability.report(
            min_bin=PROBABILITY_MIN_BIN,
            max_bin=PROBABILITY_MAX_BIN,
        )
        signed_gap = self.signed_gap.report(
            min_bin=GAP_MIN_BIN,
            max_bin=GAP_MAX_BIN,
        )
        signed_relative_gap = self.signed_relative_gap.report(
            min_bin=GAP_MIN_BIN,
            max_bin=GAP_MAX_BIN,
        )
        if signed_relative_gap["mean"] is not None:
            signed_relative_gap["mean_percent"] = 100.0 * signed_relative_gap["mean"]
        else:
            signed_relative_gap["mean_percent"] = None

        rank_rows = []
        for category in RANK_CATEGORIES:
            rank = self.true_draft_rank[category]
            rank_rows.append(
                {
                    "category": category,
                    "count": rank.count,
                    "probability": rank.count / correction_events if correction_events else None,
                    "correction_q_probability_mean": (
                        rank.correction_probability_sum / rank.count if rank.count else None
                    ),
                    "correction_q_probability_min": rank.correction_probability_min,
                    "correction_q_probability_max": rank.correction_probability_max,
                }
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_name,
            "sample_count": int(sample_count),
            "created_at": _now_iso(),
            "definitions": {
                "markov_draft_probability": (
                    "q_k[z_k], where q_k = softmax(markov_corrected_logits_k / "
                    "temperature) and z_k is the submitted draft token; at the "
                    "baseline temperature 1.0 this is softmax(markov_corrected_logits_k)"
                ),
                "rejected_position": (
                    "the first failed draft position in a non-EOS verification round; "
                    "this is the position replaced by verification.next_token"
                ),
                "signed_absolute_gap": (
                    "mean(selected-token draft probabilities at accepted positions "
                    "in the round) - selected-token draft probability at the rejected position; "
                    "reported only when this value is nonnegative"
                ),
                "signed_relative_gap": (
                    "signed_absolute_gap / mean(selected-token draft probabilities "
                    "at accepted positions in the round); reported only for the same "
                    "nonnegative-gap events"
                ),
                "gap_candidate_event": (
                    "a correction event with at least one accepted draft position, "
                    "so the same-round accepted-position mean is defined"
                ),
                "negative_gap_exclusion": (
                    "when signed_absolute_gap < 0, increment negative_gap_excluded_events "
                    "and exclude the event from paired_gap_events, both gap distributions, "
                    "gap means, CDF/CSV/plots, and probability-derived TensorBoard gap "
                    "summaries; the exclusion audit count itself remains visible"
                ),
                "paired_gap_event": (
                    "a gap candidate with signed_absolute_gap >= 0 that enters the "
                    "signed_absolute_gap distribution; it also enters signed_relative_gap "
                    "unless the accepted-position mean is zero"
                ),
                "true_draft_rank": (
                    "1 + count_v(q_k[v] > q_k[correction_token]) over the complete "
                    "Markov-corrected draft distribution; competition ranking for ties"
                ),
                "cdf": "cumulative probability through each width-0.05 histogram interval",
            },
            "counts": {key: int(self.counts[key]) for key in self.COUNT_KEYS},
            "distributions": {
                "accepted_selected_draft_probability": probability_accepted,
                "rejected_selected_draft_probability": probability_rejected,
                "signed_absolute_gap": signed_gap,
                "signed_relative_gap": signed_relative_gap,
            },
            "true_draft_rank": {
                "denominator": correction_events,
                "categories": rank_rows,
            },
        }


def merge_observation_payloads(payloads: Iterable[dict]) -> DatasetObservation:
    merged = DatasetObservation()
    for payload in payloads:
        merged.merge(DatasetObservation.from_payload(payload))
    return merged


def summarize_observation_report(report: dict) -> dict:
    distributions = report["distributions"]
    rank_probabilities = {
        row["category"]: row["probability"]
        for row in report["true_draft_rank"]["categories"]
    }
    return {
        "sample_count": report["sample_count"],
        "verification_rounds": report["counts"]["verification_rounds"],
        "accepted_token_count": distributions["accepted_selected_draft_probability"]["count"],
        "accepted_probability_mean": distributions["accepted_selected_draft_probability"]["mean"],
        "correction_events": report["counts"]["correction_events"],
        "rejected_probability_mean": distributions["rejected_selected_draft_probability"]["mean"],
        "first_position_correction_events": report["counts"]["first_position_correction_events"],
        "gap_candidate_events": report["counts"]["gap_candidate_events"],
        "paired_gap_events": report["counts"]["paired_gap_events"],
        "negative_gap_excluded_events": report["counts"]["negative_gap_excluded_events"],
        "undefined_relative_gap_events": report["counts"]["undefined_relative_gap_events"],
        "signed_absolute_gap_mean": distributions["signed_absolute_gap"]["mean"],
        "signed_relative_gap_mean": distributions["signed_relative_gap"]["mean"],
        "signed_relative_gap_mean_percent": distributions["signed_relative_gap"]["mean_percent"],
        "true_draft_rank_probabilities": rank_probabilities,
    }


def _write_distribution_csv(path: Path, named_reports: Iterable[tuple[str, dict]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "bin_index",
                "lower",
                "upper",
                "interval",
                "count",
                "probability",
                "cumulative_count",
                "cdf",
            ),
        )
        writer.writeheader()
        for metric_name, distribution in named_reports:
            for row in distribution["bins"]:
                writer.writerow({"metric": metric_name, **row})


def _write_rank_csv(path: Path, rank_report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "category",
            "count",
            "probability",
            "correction_q_probability_mean",
            "correction_q_probability_min",
            "correction_q_probability_max",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rank_report["categories"])


def _plot_report(dataset_dir: Path, report: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions = report["distributions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for key, label in (
        ("accepted_selected_draft_probability", "accepted positions"),
        ("rejected_selected_draft_probability", "replaced position"),
    ):
        bins = distributions[key]["bins"]
        axes[0].plot([row["upper"] for row in bins], [row["cdf"] or 0.0 for row in bins], label=label)
    axes[0].set(
        xlabel="selected-token Markov draft probability",
        ylabel="CDF",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for key, label in (
        ("signed_absolute_gap", "absolute gap"),
        ("signed_relative_gap", "relative gap"),
    ):
        bins = distributions[key]["bins"]
        if bins:
            axes[1].plot([row["upper"] for row in bins], [row["cdf"] or 0.0 for row in bins], label=label)
    axes[1].set(
        xlabel="gap (accepted mean - rejected), conditioned on gap >= 0",
        ylabel="CDF",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    rank_rows = report["true_draft_rank"]["categories"]
    axes[2].bar(
        [row["category"] for row in rank_rows],
        [row["probability"] or 0.0 for row in rank_rows],
    )
    axes[2].set(xlabel="true_draft_rank category", ylabel="probability", ylim=(0, 1))
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(f"{report['dataset']} Markov draft-probability observations")
    fig.tight_layout()
    output_path = dataset_dir / "observation_plots.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


class MarkovDraftProbabilityRecorder:
    def __init__(self, *, artifact_root: Path, tensorboard_dir: str | None, step: int | None):
        self.artifact_root = Path(artifact_root)
        self.tensorboard_dir = tensorboard_dir
        self.step = step
        self.current: DatasetObservation | None = None
        self.rows: list[dict] = []

    def start(self) -> None:
        if self.current is not None:
            raise RuntimeError("Previous dataset observation was not finished")
        self.current = DatasetObservation()

    def observe(self, *, proposal: DSparkDraftProposal, verification: VerificationResult) -> None:
        if self.current is None:
            raise RuntimeError("MarkovDraftProbabilityRecorder.start() was not called")
        self.current.observe(proposal, verification)

    def finish(self, *, dataset_name: str, metric_summary: dict, args_payload: dict, tasks: list) -> dict | None:
        if self.current is None:
            raise RuntimeError("MarkovDraftProbabilityRecorder.start() was not called")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dataset_dir = self.artifact_root / dataset_name
        rank_dir = dataset_dir / "rank_stats"
        rank_payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_name,
            "rank": rank,
            "world_size": world_size,
            "statistics": self.current.to_payload(),
        }
        _write_json_atomic(rank_dir / f"rank_{rank}.json", rank_payload)
        self.current = None
        dist.barrier()

        summary = None
        if rank == 0:
            payloads = []
            for source_rank in range(world_size):
                source_path = rank_dir / f"rank_{source_rank}.json"
                source_payload = json.loads(source_path.read_text(encoding="utf-8"))
                if source_payload.get("dataset") != dataset_name:
                    raise RuntimeError(f"Mismatched rank observation file: {source_path}")
                payloads.append(source_payload["statistics"])
            merged = merge_observation_payloads(payloads)
            report = merged.build_report(
                dataset_name=dataset_name,
                sample_count=int(metric_summary["sample_count"]),
            )
            output_payload = {
                "config": {"args": args_payload, "tasks": tasks},
                "spec_metric_summary": metric_summary,
                "observation": report,
                "observation_summary": summarize_observation_report(report),
            }
            metrics_path = dataset_dir / "metrics.json"
            _write_json_atomic(metrics_path, output_payload)
            _write_distribution_csv(
                dataset_dir / "markov_draft_probability_cdf.csv",
                (
                    ("accepted_selected_draft_probability", report["distributions"]["accepted_selected_draft_probability"]),
                    ("rejected_selected_draft_probability", report["distributions"]["rejected_selected_draft_probability"]),
                ),
            )
            _write_distribution_csv(
                dataset_dir / "signed_gap_cdf.csv",
                (
                    ("signed_absolute_gap", report["distributions"]["signed_absolute_gap"]),
                    ("signed_relative_gap", report["distributions"]["signed_relative_gap"]),
                ),
            )
            _write_rank_csv(dataset_dir / "true_draft_rank.csv", report["true_draft_rank"])
            _plot_report(dataset_dir, report)
            summary = output_payload["observation_summary"]
            self.rows.append({"dataset": dataset_name, **summary})
            print(
                "Markov draft-probability observation: "
                + json.dumps({"dataset": dataset_name, **summary}, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            print(f"Wrote Markov draft-probability artifacts to {dataset_dir}", flush=True)
        dist.barrier()
        return summary

    def log_tensorboard(self) -> None:
        if not self.rows or self.tensorboard_dir is None or self.step is None:
            return
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        scalar_keys = (
            "accepted_probability_mean",
            "rejected_probability_mean",
            "gap_candidate_events",
            "paired_gap_events",
            "negative_gap_excluded_events",
            "undefined_relative_gap_events",
            "signed_absolute_gap_mean",
            "signed_relative_gap_mean",
            "signed_relative_gap_mean_percent",
        )
        for row in self.rows:
            dataset = row["dataset"]
            for key in scalar_keys:
                value = row.get(key)
                if value is not None and math.isfinite(float(value)):
                    writer.add_scalar(f"markov_draft_probability/{dataset}/{key}", float(value), self.step)
            for category, value in row.get("true_draft_rank_probabilities", {}).items():
                if value is not None:
                    writer.add_scalar(
                        f"markov_draft_probability/{dataset}/true_draft_rank_{category}",
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
            "accepted_n",
            "accepted_mean",
            "corrections",
            "rejected_mean",
            "gap_candidates",
            "negative_excluded",
            "included_gap_n",
            "signed_gap_mean",
            "signed_gap_%",
            *[f"rank{index}" for index in range(1, 11)],
            "rank_other",
        )
        for row in self.rows:
            ranks = row["true_draft_rank_probabilities"]
            fmt = lambda value: "-" if value is None else f"{float(value):.4f}"
            table.add_row(
                (
                    row["dataset"],
                    row["accepted_token_count"],
                    fmt(row["accepted_probability_mean"]),
                    row["correction_events"],
                    fmt(row["rejected_probability_mean"]),
                    row["gap_candidate_events"],
                    row["negative_gap_excluded_events"],
                    row["paired_gap_events"],
                    fmt(row["signed_absolute_gap_mean"]),
                    fmt(row["signed_relative_gap_mean_percent"]),
                    *[fmt(ranks[str(index)]) for index in range(1, 11)],
                    fmt(ranks["other"]),
                )
            )
        print("Raw Markov draft-probability and correction-rank observations:", flush=True)
        print(table.get_string(), flush=True)


class MarkovDraftProbabilityEvaluator(Qwen3DSparkEvaluator):
    """Qwen3 DSpark evaluator with an additional read-only observation hook."""

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        if self.draft_model.markov_head is None:
            raise RuntimeError("Draft checkpoint has no Markov correction head")
        self.markov_probability_recorder = MarkovDraftProbabilityRecorder(
            artifact_root=Path(args.observation_artifact_root),
            tensorboard_dir=args.tensorboard_dir,
            step=args.step,
        )
        self.observation_summaries: dict[str, dict] = {}

    def mark_dataset_started(self, dataset_name: str) -> None:
        super().mark_dataset_started(dataset_name)
        self.markov_probability_recorder.start()

    def _post_verify(self, proposal, verification) -> None:
        # Preserve the baseline confidence calibration recorder, then add this
        # isolated selected-token Markov draft-probability observation.
        super()._post_verify(proposal, verification)
        if not isinstance(proposal, DSparkDraftProposal):
            raise TypeError(f"Expected DSparkDraftProposal, got {type(proposal)!r}")
        self.markov_probability_recorder.observe(
            proposal=proposal,
            verification=verification,
        )

    def record_dataset_metrics(self, *, dataset_name: str, metric_summary: dict):
        self.mark_dataset_phase(dataset_name, "reducing_and_writing_markov_draft_probability_observations")
        summary = self.markov_probability_recorder.finish(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
            args_payload=json.loads(json.dumps(vars(self.args), cls=CustomJSONEncoder)),
            tasks=[list(task) for task in self.tasks],
        )
        if summary is not None:
            self.observation_summaries[dataset_name] = summary
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
            "markov_draft_probability_observation_summary": self.observation_summaries.get(dataset_name),
        }
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
        self.markov_probability_recorder.log_tensorboard()

    def print_results(self) -> None:
        super().print_results()
        self.markov_probability_recorder.print_results()


def evaluation_worker(local_rank: int, args) -> None:
    if local_rank == 0:
        print(json.dumps(vars(args), indent=2, ensure_ascii=False, cls=CustomJSONEncoder), flush=True)
    evaluator = MarkovDraftProbabilityEvaluator(local_rank, args)
    try:
        evaluator.evaluate()
    finally:
        evaluator.clean_up()
