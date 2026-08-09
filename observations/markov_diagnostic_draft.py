"""Isolated DSpark proposals retaining temperature-independent Markov logits.

The verifier must receive the operational proposal distribution associated
with the decoding temperature.  Markov observation experiments additionally
need the already-computed corrected logits so they can define a separate
diagnostic distribution ``softmax(markov_corrected_logits)`` without changing
sampling or speculative verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from deepspec.eval.base_evaluator import DraftProposal
from deepspec.eval.dspark.draft_ops import (
    DSparkDraftProposal,
    forward_dspark_draft_block,
)
from deepspec.utils.sampling import logits_to_probs


@dataclass
class DiagnosticMarkovDraftProposal(DSparkDraftProposal):
    """DSpark proposal plus corrected logits used only by observation hooks."""

    diagnostic_markov_logits: torch.Tensor | None = None


def _empty_proposal(draft_input_ids: torch.Tensor) -> DiagnosticMarkovDraftProposal:
    return DiagnosticMarkovDraftProposal(
        draft_token_count=0,
        verify_input_ids=draft_input_ids[:, :1],
        draft_probs=None,
        confidence_logits=None,
        diagnostic_markov_logits=None,
    )


def _predict_confidence_logits(
    model,
    *,
    proposal_hidden_states: torch.Tensor,
    draft_input_ids: torch.Tensor,
    sampled_tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    prev_token_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :-1]],
        dim=1,
    )
    confidence_pred = model.predict_confidence_step(
        proposal_hidden_states,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is None:
        return None
    return confidence_pred.float().reshape(
        confidence_pred.shape[0],
        block_size,
        -1,
    )[:, :, 0]


def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    *,
    block_size: int,
    threshold: float,
) -> int:
    if threshold <= 0.0:
        return int(block_size)
    below_threshold = confidence_logits.sigmoid() < threshold
    if not bool(below_threshold[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())


def build_diagnostic_markov_proposal(
    model,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
) -> DiagnosticMarkovDraftProposal:
    """Build the baseline proposal while retaining its corrected logits.

    ``draft_probs`` remains the operational temperature-dependent distribution
    consumed by speculative verification. ``diagnostic_markov_logits`` is
    never consumed by decoding and is retained only until the post-verify
    observation hook.
    """

    assert draft_input_ids.size(0) == 1, (
        "build_diagnostic_markov_proposal requires batch_size=1"
    )
    proposal_hidden_states = block_hidden[:, :block_size, :]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    sampled_tokens, corrected_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)
    confidence_logits = None
    if model.confidence_head is not None:
        confidence_logits = _predict_confidence_logits(
            model,
            proposal_hidden_states=proposal_hidden_states,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
            block_size=block_size,
        )
        if confidence_logits is None:
            return _empty_proposal(draft_input_ids)
        proposal_draft_tokens = _confident_prefix_length(
            confidence_logits,
            block_size=block_size,
            threshold=float(confidence_threshold),
        )

    if proposal_draft_tokens == 0:
        return _empty_proposal(draft_input_ids)

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    retained_logits = corrected_logits[:, :proposal_draft_tokens, :]
    operational_draft_probs = logits_to_probs(retained_logits, temperature)
    return DiagnosticMarkovDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=operational_draft_probs,
        confidence_logits=(
            confidence_logits[:, :proposal_draft_tokens]
            if confidence_logits is not None
            else None
        ),
        diagnostic_markov_logits=retained_logits,
    )


class DiagnosticMarkovProposalMixin:
    """Override only Markov observation evaluators' proposal construction."""

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        del stop_token_ids
        model = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), self.max_proposal_tokens),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=self.max_proposal_tokens,
        )
        return build_diagnostic_markov_proposal(
            model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
        )


__all__ = [
    "DiagnosticMarkovDraftProposal",
    "DiagnosticMarkovProposalMixin",
    "build_diagnostic_markov_proposal",
]
