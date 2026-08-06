from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from transformers import DynamicCache
from tqdm import tqdm

from deepspec.data.parser import encode_chat_messages
from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)
from deepspec.utils import (
    init_dist,
    seed_all,
)


DEFAULT_DATASET_ROOT = "./eval_datasets"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


class DatasetProgress:
    """Show one rank-0 tqdm bar aggregated through lightweight progress files."""

    def __init__(
        self,
        *,
        dataset_name: str,
        total_samples: int,
        local_samples: int,
        global_rank: int,
        world_size: int,
        progress_dir: str | None,
    ):
        self.dataset_name = dataset_name
        self.total_samples = int(total_samples)
        self.local_samples = int(local_samples)
        self.global_rank = int(global_rank)
        self.world_size = int(world_size)
        self.progress_root = Path(progress_dir) if progress_dir is not None else None
        self.dataset_dir = (
            self.progress_root / dataset_name
            if self.progress_root is not None
            else None
        )
        self.rank_path = (
            self.dataset_dir / f"rank_{self.global_rank}.json"
            if self.dataset_dir is not None
            else None
        )
        self.completed = 0
        self.progress_bar = None
        self.stop_event = threading.Event()
        self.monitor_thread = None

    def _write_rank_state(self, status: str) -> None:
        if self.rank_path is None:
            return
        _write_json_atomic(
            self.rank_path,
            {
                "dataset": self.dataset_name,
                "rank": self.global_rank,
                "world_size": self.world_size,
                "completed_samples": self.completed,
                "assigned_samples": self.local_samples,
                "status": status,
                "updated_at": _now_iso(),
            },
        )

    def _read_aggregate(self) -> tuple[int, int]:
        assert self.dataset_dir is not None
        completed = 0
        ready_ranks = 0
        for rank in range(self.world_size):
            path = self.dataset_dir / f"rank_{rank}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            completed += int(payload.get("completed_samples", 0))
            ready_ranks += 1
        return min(completed, self.total_samples), ready_ranks

    def _refresh_aggregate(self, *, force: bool = False) -> None:
        assert self.progress_bar is not None
        assert self.dataset_dir is not None
        completed, ready_ranks = self._read_aggregate()
        delta = completed - int(self.progress_bar.n)
        if delta > 0:
            self.progress_bar.update(delta)
        if delta > 0 or force:
            self.progress_bar.set_postfix_str(
                f"ranks={ready_ranks}/{self.world_size}",
                refresh=True,
            )
            _write_json_atomic(
                self.dataset_dir / "progress.json",
                {
                    "dataset": self.dataset_name,
                    "completed_samples": completed,
                    "total_samples": self.total_samples,
                    "ready_ranks": ready_ranks,
                    "world_size": self.world_size,
                    "status": (
                        "completed" if completed >= self.total_samples else "running"
                    ),
                    "updated_at": _now_iso(),
                },
            )

    def _monitor(self) -> None:
        while not self.stop_event.wait(1.0):
            self._refresh_aggregate()

    def __enter__(self):
        if self.dataset_dir is not None:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._write_rank_state("running")
        if self.global_rank == 0:
            aggregate_enabled = self.dataset_dir is not None
            bar_total = self.total_samples if aggregate_enabled else self.local_samples
            description = (
                f"{self.dataset_name} samples"
                if aggregate_enabled
                else f"{self.dataset_name} rank0 samples"
            )
            self.progress_bar = tqdm(
                total=bar_total,
                desc=description,
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stderr,
                leave=True,
            )
            if aggregate_enabled:
                self._refresh_aggregate(force=True)
                self.monitor_thread = threading.Thread(
                    target=self._monitor,
                    name=f"{self.dataset_name}-progress-monitor",
                    daemon=True,
                )
                self.monitor_thread.start()
        return self

    def advance(self) -> None:
        self.completed += 1
        self._write_rank_state("running")
        if (
            self.global_rank == 0
            and self.progress_root is None
            and self.progress_bar is not None
        ):
            self.progress_bar.update(1)

    def __exit__(self, exc_type, exc_value, traceback_obj):
        success = exc_type is None
        self._write_rank_state("completed" if success else "failed")

        if self.global_rank == 0 and self.progress_bar is not None:
            if self.monitor_thread is not None:
                if success:
                    while self._read_aggregate()[0] < self.total_samples:
                        time.sleep(1.0)
                self.stop_event.set()
                self.monitor_thread.join(timeout=5.0)
                self._refresh_aggregate(force=True)
            self.progress_bar.close()
        return False


def load_and_process_dataset(
    data_name: str,
    dataset_root: str = DEFAULT_DATASET_ROOT,
):
    dataset_path = Path(dataset_root) / f"{data_name}.jsonl"
    assert dataset_path.exists()

    rows = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            assert (
                isinstance(turns, list)
                and len(turns) > 0
                and all(isinstance(turn, str) for turn in turns)
            ), (
                f"{dataset_path}:{line_number} must contain a non-empty string list field `turns`."
            )
            row["turns"] = turns[:1]
            rows.append(row)
    return rows


def trim_output_ids(
    output_ids: torch.Tensor,
    num_input_tokens: int,
    stop_token_ids: list[int] | None,
) -> torch.Tensor:
    if stop_token_ids is None:
        return output_ids
    stop_token_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
    stop_token_indices = torch.isin(
        output_ids[0][num_input_tokens:],
        stop_token_tensor,
    ).nonzero(as_tuple=True)[0]
    if stop_token_indices.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]


def has_stop_token(token_ids: torch.Tensor, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_token_tensor = torch.tensor(stop_token_ids, device=token_ids.device)
    return bool(torch.isin(token_ids, stop_token_tensor).any().item())


def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    stop_token_ids = []
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    return stop_token_ids


def assert_no_final_target_layer(target_model, target_layer_ids) -> None:
    target_config = target_model.config
    if hasattr(target_config, "text_config"):
        target_config = target_config.text_config
    last_layer_id = int(target_config.num_hidden_layers) - 1
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    assert last_layer_id not in target_layer_ids, (
        "target_layer_ids must not include the final target decoder layer "
        f"{last_layer_id}. Transformers output_hidden_states stores the final "
        "normalized hidden state at that position, while target cache generation "
        "stores raw decoder-layer outputs. Use an earlier layer and regenerate "
        "the target cache and draft checkpoint."
    )


def build_results_table(
    *,
    rows: list[dict[str, object]],
    model_name_or_path: str,
    draft_name_or_path: str,
    header: bool = True,
) -> str:
    from prettytable import PrettyTable

    table = PrettyTable()
    max_positions = max(
        (len(metrics["accept_rates_by_position"]) for metrics in rows),
        default=0,
    )
    field_names = [
        "dataset",
        "target_model",
        "draft_model",
        "#propose",
        "accept_len",
        "verify_rate",
    ]
    field_names.extend(f"accept_rate@{pos_idx}" for pos_idx in range(max_positions))
    table.field_names = field_names
    table.header = header

    normalized_target_model_name = model_name_or_path.rstrip("/")
    target_model_name = (
        os.path.basename(normalized_target_model_name) or normalized_target_model_name
    )
    normalized_draft_model_name = draft_name_or_path.rstrip("/")
    draft_model_name = (
        os.path.basename(normalized_draft_model_name) or normalized_draft_model_name
    )
    for metrics in rows:
        row = [
            metrics["dataset"],
            target_model_name,
            draft_model_name,
            f"{metrics['draft_tokens_per_proposal']:.2f}+1",
            f"{metrics['acceptance_length']:.2f}",
            f"{metrics['verify_rate']:.4f}",
        ]
        row.extend(
            f"{accept_rate:.4f}" if accept_rate is not None else "-"
            for accept_rate in metrics["accept_rates_by_position"]
        )
        row.extend(["-"] * (max_positions - len(metrics["accept_rates_by_position"])))
        table.add_row(row)
    return table.get_string()


@dataclass
class DraftProposal:
    draft_token_count: int
    verify_input_ids: torch.Tensor
    draft_probs: torch.Tensor | None


@dataclass
class VerificationResult:
    target_output: Any
    target_probs: torch.Tensor
    accept_prefix_mask: torch.Tensor | None
    accepted_draft_tokens: int
    next_token: torch.Tensor
    effective_proposal_length: int
    terminated_by_stop_token: bool = False
    committed_tokens: torch.Tensor | None = None


def verify_draft_tokens(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids: torch.Tensor,
    start: int,
    past_key_values_target: DynamicCache,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids: torch.Tensor | None = None,
    stop_token_ids: list[int] | None = None,
) -> VerificationResult:
    """Verify draft tokens with the target model and rejection sampling."""
    if proposal.draft_token_count > max_proposal_tokens:
        raise ValueError(
            "DraftProposal.draft_token_count must not exceed "
            f"max_proposal_tokens={max_proposal_tokens}, "
            f"got {proposal.draft_token_count}."
        )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with the current "
            "accepted token."
        )

    draft_token_count = int(proposal.draft_token_count)
    verify_length = draft_token_count + 1
    verify_position_ids = position_ids[:, start : start + verify_length]
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=verify_position_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    if target_output.logits.ndim != 3:
        raise ValueError(
            "target model must return rank-3 logits [B, S, V], "
            f"got ndim={target_output.logits.ndim}."
        )
    target_probs = logits_to_probs(target_output.logits, float(temperature))
    if (
        draft_token_count > 0
        and proposal.draft_probs is not None
        and proposal.draft_probs.size(-1) != target_probs.size(-1)
    ):
        raise ValueError(
            "DraftProposal.draft_probs vocab size must match target logits, "
            f"got {proposal.draft_probs.size(-1)} and {target_probs.size(-1)}."
        )

    accept_prefix_mask = None
    if draft_token_count > 0:
        assert proposal.draft_probs is not None
        proposed_tokens = proposal.verify_input_ids[:, 1:]
        selected_target_probs = gather_token_probs(
            target_probs[:, :-1, :],
            proposed_tokens,
        )
        selected_draft_probs = gather_token_probs(
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)
        accept_prob = torch.clamp(
            selected_target_probs / selected_draft_probs,
            max=1.0,
        )
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
        accept_prefix_mask = accept_mask.cumprod(dim=1)
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    else:
        accepted_draft_tokens = 0

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            eos_pos = int(eos_hits[0].item())
            accepted_draft_tokens = eos_pos + 1
            effective_proposal_length = eos_pos + 1
            terminated_by_stop_token = True

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:
        assert proposal.draft_probs is not None
        next_token = sample_residual(
            target_probs[:, accepted_draft_tokens, :],
            proposal.draft_probs[:, accepted_draft_tokens, :],
        )
    else:
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

    committed_tokens = torch.cat(
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )

    return VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


@torch.inference_mode()
def generate_decoding_sample(
    *,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_proposal_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    init_context: Callable[..., Any],
    propose: Callable[..., DraftProposal],
    update: Callable[[Any, VerificationResult], None],
    post_verify: Callable[[DraftProposal, VerificationResult], None] | None = None,
) -> SimpleNamespace:
    """Speculative-decoding loop.

    `init_context(initial_output, output_ids, position_ids, num_input_tokens)`
    builds the algorithm-specific state once after prefill. `propose(context,
    output_ids, position_ids, start, stop_token_ids)` returns the next
    DraftProposal. `update(context, verification)` advances the state when the
    loop continues. `post_verify` is an optional diagnostic hook called after
    every verification (used for confidence calibration).
    """
    assert max_proposal_tokens >= 1
    assert input_ids.size(0) == 1, "only bsz=1 is supported"

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)

    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    output = target_model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(
        logits_to_probs(output.logits, float(temperature))
    )

    start = input_ids.shape[1]
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []

    initial_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
    if has_stop_token(initial_token, stop_token_ids):
        output_ids = output_ids[:, : num_input_tokens + 1]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            verify_count=0,
        )

    context = init_context(
        initial_output=output,
        output_ids=output_ids,
        position_ids=position_ids,
        num_input_tokens=num_input_tokens,
    )

    while start < max_length:
        proposal = propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        verification = verify_draft_tokens(
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],
            stop_token_ids=stop_token_ids,
        )
        if post_verify is not None:
            post_verify(proposal, verification)

        proposal_lengths.append(int(verification.effective_proposal_length))
        accepted_draft_tokens = int(verification.accepted_draft_tokens)
        accepted_draft_lengths.append(accepted_draft_tokens)

        output_ids[:, start : start + accepted_draft_tokens + 1] = (
            proposal.verify_input_ids[:, : accepted_draft_tokens + 1]
        )

        if verification.terminated_by_stop_token:
            acceptance_lengths.append(accepted_draft_tokens)
            start += accepted_draft_tokens
            past_key_values_target.crop(start)
            break

        output_ids[:, start + accepted_draft_tokens + 1] = verification.next_token
        new_token_ids = output_ids[:, start + 1 : start + accepted_draft_tokens + 2]
        acceptance_lengths.append(accepted_draft_tokens + 1)
        start += accepted_draft_tokens + 1
        past_key_values_target.crop(start)
        update(context, verification)

        if has_stop_token(new_token_ids, stop_token_ids):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        verify_count=len(proposal_lengths),
    )


class BaseEvaluator:

    def __init__(self, local_rank: int, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(
            local_rank,
            timeout_minutes=int(getattr(args, "dist_timeout_minutes", 24 * 60)),
            backend=str(getattr(args, "dist_backend", "gloo")),
        )
        self.tasks = args.tasks

        self.target_model, self.draft_model, self.tokenizer = self.build_models()
        self.metrics_rows: list[dict[str, object]] = []

    @property
    def max_proposal_tokens(self) -> int:
        raise NotImplementedError

    def build_models(self):
        raise NotImplementedError

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        raise NotImplementedError

    def build_metrics_row(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
    ) -> dict[str, object]:
        proposal_count = int(metric_summary["proposal_count"])
        if proposal_count == 0:
            acceptance_length = 0.0
            draft_tokens_per_proposal = 0.0
            verify_rate = 0.0
            accept_rates_by_position = [None] * self.max_proposal_tokens
        else:
            acceptance_length = (
                int(metric_summary["acceptance_length_sum"]) / proposal_count
            )
            draft_tokens_per_proposal = (
                int(metric_summary["proposal_length_sum"]) / proposal_count
            )
            verify_rate = int(metric_summary["acceptance_length_sum"]) / (
                int(metric_summary["proposal_length_sum"]) + proposal_count
            )
            proposals_at_pos = metric_summary["proposals_at_pos"]
            accepted_at_pos = metric_summary["accepted_at_pos"]
            assert isinstance(proposals_at_pos, list)
            assert isinstance(accepted_at_pos, list)
            accept_rates_by_position = []
            for pos_idx in range(self.max_proposal_tokens):
                position_proposal_count = proposals_at_pos[pos_idx]
                if position_proposal_count == 0:
                    accept_rates_by_position.append(None)
                    continue
                accept_rates_by_position.append(
                    accepted_at_pos[pos_idx] / position_proposal_count
                )
        return {
            "dataset": dataset_name,
            "num_samples": int(metric_summary["sample_count"]),
            "draft_tokens_per_proposal": draft_tokens_per_proposal,
            "acceptance_length": acceptance_length,
            "verify_rate": verify_rate,
            "accept_rates_by_position": accept_rates_by_position,
        }

    def run_dataset(
        self,
        *,
        dataset_name: str,
        max_samples: int | None,
    ) -> list[SimpleNamespace]:
        seed_all(int(self.args.seed))
        dataset = load_and_process_dataset(dataset_name)

        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(int(self.args.seed))
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        responses = []
        local_indices = range(self.global_rank, len(dataset), self.world_size)
        if self.global_rank == 0:
            print(
                f"Dataset {dataset_name}: {len(dataset)} samples across "
                f"{self.world_size} rank(s)",
                flush=True,
            )
        with DatasetProgress(
            dataset_name=dataset_name,
            total_samples=len(dataset),
            local_samples=len(local_indices),
            global_rank=self.global_rank,
            world_size=self.world_size,
            progress_dir=getattr(self.args, "progress_dir", None),
        ) as progress:
            for idx in local_indices:
                seed_all(int(self.args.seed) + idx)
                instance = dataset[idx]
                messages = [{"role": "user", "content": instance["turns"][0]}]
                input_ids = encode_chat_messages(
                    self.tokenizer,
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    # enable_thinking=True,
                ).to(self.device)
                responses.append(
                    self.generate_one_sample(
                        input_ids=input_ids,
                        stop_token_ids=stop_token_ids,
                    )
                )
                progress.advance()

        return responses

    def _update_manifest_dataset(self, dataset_name: str, **updates) -> None:
        if dist.get_rank() != 0:
            return
        manifest_value = getattr(self.args, "experiment_manifest_path", None)
        if manifest_value is None:
            return
        manifest_path = Path(manifest_value)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        for dataset_record in manifest.get("datasets", []):
            if dataset_record.get("name") == dataset_name:
                dataset_record.update(updates)
                break
        manifest["last_update_time"] = _now_iso()
        manifest["completed_dataset_count"] = sum(
            record.get("status") == "completed"
            for record in manifest.get("datasets", [])
        )
        _write_json_atomic(manifest_path, manifest)

    def mark_dataset_started(self, dataset_name: str) -> None:
        if dist.get_rank() != 0:
            return
        started_at = _now_iso()
        self._update_manifest_dataset(
            dataset_name,
            status="running",
            phase="sampling",
            started_at=started_at,
            completed_at=None,
        )
        print(f"Starting dataset: {dataset_name}", flush=True)

    def mark_dataset_phase(self, dataset_name: str, phase: str) -> None:
        if dist.get_rank() != 0:
            return
        self._update_manifest_dataset(
            dataset_name,
            status="running",
            phase=phase,
        )
        print(f"Dataset {dataset_name}: {phase}", flush=True)

    def record_incremental_dataset_result(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_summary: dict | None = None,
    ) -> None:
        if dist.get_rank() != 0:
            return
        completed_at = _now_iso()
        result = {
            "dataset": metrics_row["dataset"],
            "completed_at": completed_at,
            "spec": metrics_row,
            "confidence_summary": confidence_summary,
        }
        results_value = getattr(self.args, "dataset_results_path", None)
        if results_value is not None:
            results_path = Path(results_value)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"Appended dataset result to {results_path}", flush=True)

        self._update_manifest_dataset(
            str(metrics_row["dataset"]),
            status="completed",
            phase="completed",
            completed_at=completed_at,
            result=result,
        )

    def allreduce_response_metrics(
        self,
        responses: list[SimpleNamespace],
    ) -> dict[str, int | list[int]]:
        metric_summary: dict[str, int | list[int]] = {
            "sample_count": len(responses),
            "proposal_count": 0,
            "acceptance_length_sum": 0,
            "proposal_length_sum": 0,
            "proposals_at_pos": [0] * self.max_proposal_tokens,
            "accepted_at_pos": [0] * self.max_proposal_tokens,
        }
        proposals_at_pos = metric_summary["proposals_at_pos"]
        accepted_at_pos = metric_summary["accepted_at_pos"]
        assert isinstance(proposals_at_pos, list)
        assert isinstance(accepted_at_pos, list)

        for response in responses:
            acceptance_lengths = getattr(response, "acceptance_lengths", None)
            proposal_lengths = getattr(response, "proposal_lengths", None)
            accepted_draft_lengths = getattr(response, "accepted_draft_lengths", None)
            if (
                acceptance_lengths is None
                or proposal_lengths is None
                or accepted_draft_lengths is None
            ):
                continue
            assert (
                len(acceptance_lengths)
                == len(proposal_lengths)
                == len(accepted_draft_lengths)
            ), (
                "acceptance_lengths, proposal_lengths and accepted_draft_lengths "
                "must have the same length."
            )
            for acceptance_length, proposal_length, accepted_draft_length in zip(
                acceptance_lengths,
                proposal_lengths,
                accepted_draft_lengths,
            ):
                metric_summary["proposal_count"] += 1
                metric_summary["acceptance_length_sum"] += int(acceptance_length)
                metric_summary["proposal_length_sum"] += int(proposal_length)
                accepted_draft_length = int(accepted_draft_length)
                for pos_idx in range(self.max_proposal_tokens):
                    if proposal_length > pos_idx:
                        proposals_at_pos[pos_idx] += 1
                    if accepted_draft_length > pos_idx:
                        accepted_at_pos[pos_idx] += 1

        reduction_device = (
            torch.device("cpu")
            if str(dist.get_backend()).lower() == "gloo"
            else self.device
        )
        scalar_tensor = torch.tensor(
            [
                int(metric_summary["sample_count"]),
                int(metric_summary["proposal_count"]),
                int(metric_summary["acceptance_length_sum"]),
                int(metric_summary["proposal_length_sum"]),
            ],
            device=reduction_device,
            dtype=torch.int64,
        )
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

        position_tensor = torch.tensor(
            proposals_at_pos + accepted_at_pos,
            device=reduction_device,
            dtype=torch.int64,
        )
        if position_tensor.numel() > 0:
            dist.all_reduce(position_tensor, op=dist.ReduceOp.SUM)
        return {
            "sample_count": int(scalar_tensor[0].item()),
            "proposal_count": int(scalar_tensor[1].item()),
            "acceptance_length_sum": int(scalar_tensor[2].item()),
            "proposal_length_sum": int(scalar_tensor[3].item()),
            "proposals_at_pos": position_tensor[
                : self.max_proposal_tokens
            ].tolist(),
            "accepted_at_pos": position_tensor[
                self.max_proposal_tokens :
            ].tolist(),
        }

    def log_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        assert self.args.tensorboard_dir is not None
        assert self.args.step is not None
        Path(self.args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        for metrics in self.metrics_rows:
            dataset_name = metrics["dataset"]
            writer.add_scalar(
                f"eval/{dataset_name}/draft_tokens_per_proposal",
                float(metrics["draft_tokens_per_proposal"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/accept_length",
                float(metrics["acceptance_length"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/verify_rate",
                float(metrics["verify_rate"]),
                global_step=self.args.step,
            )
            for pos_idx, accept_rate in enumerate(metrics["accept_rates_by_position"]):
                if accept_rate is None:
                    continue
                writer.add_scalar(
                    f"eval/{dataset_name}/accept_rate@{pos_idx}",
                    float(accept_rate),
                    global_step=self.args.step,
                )
        writer.close()

    def print_results(self) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=self.metrics_rows,
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
            ),
            flush=True,
        )

    def print_dataset_result(self, metrics_row: dict[str, object]) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=[metrics_row],
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
                header=False,
            ),
            flush=True,
        )

    def record_dataset_metrics(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
    ) -> dict[str, object] | None:
        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) <= 0:
            return None
        metrics_row = self.build_metrics_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        self.metrics_rows.append(metrics_row)
        self.print_dataset_result(metrics_row)
        return metrics_row

    def report_results(self) -> None:
        if dist.get_rank() == 0 and self.metrics_rows:
            if self.args.tensorboard_dir is not None:
                self.log_tensorboard()
        self.print_results()

    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            self.mark_dataset_started(dataset_name)
            responses = self.run_dataset(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            self.mark_dataset_phase(dataset_name, "reducing_spec_metrics")
            metric_summary = self.allreduce_response_metrics(responses)
            metrics_row = self.record_dataset_metrics(
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )
            if metrics_row is not None:
                self.mark_dataset_phase(dataset_name, "writing_incremental_result")
                self.record_incremental_dataset_result(metrics_row=metrics_row)

        self.report_results()

    def clean_up(self) -> None:
        dist.destroy_process_group()
