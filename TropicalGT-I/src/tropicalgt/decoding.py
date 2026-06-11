from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .data import PAD_ID, encode_byte_ids, encode_record_bytes
from .records import GraphRecord
from .tokenizer import TokenGTTokenizer


@dataclass(frozen=True)
class MeetInMiddleConfig:
    """Configuration for the TropicalGT-I meet-in-the-middle adaptation.

    The implementation follows the MIM paper's operational idea of scoring a
    sequence from both sides, while staying honest about the current model.  By
    default TropicalGT-I uses one shared-weight model for the reverse pass.  A
    future separately trained right-to-left checkpoint can replace this path
    without changing the config surface.
    """

    enabled: bool = False
    agreement_weight: float = 0.0
    reverse_nll_weight: float = 0.0
    split_ratio: float = 0.5
    max_records: int = 0
    reverse_model_path: str = ""
    mode: str = "shared_weight_reverse_pass"


def meet_in_middle_config(raw: dict[str, Any] | None) -> MeetInMiddleConfig:
    raw = raw if isinstance(raw, dict) else {}
    return MeetInMiddleConfig(
        enabled=bool(raw.get("enabled", False)),
        agreement_weight=max(float(raw.get("agreement_weight", 0.0)), 0.0),
        reverse_nll_weight=max(float(raw.get("reverse_nll_weight", 0.0)), 0.0),
        split_ratio=min(max(float(raw.get("split_ratio", 0.5)), 0.05), 0.95),
        max_records=max(int(raw.get("max_records", 0)), 0),
        reverse_model_path=str(raw.get("reverse_model_path", "") or ""),
        mode=str(raw.get("mode", "shared_weight_reverse_pass") or "shared_weight_reverse_pass"),
    )


def encode_record_bytes_reverse(
    record: GraphRecord,
    seq_len: int,
    graph_autoregressive: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = record.autoregressive_text(seed=seed) if graph_autoregressive else record.text
    ids = [b + 1 for b in text.encode("utf-8", "ignore")]
    ids.reverse()
    return encode_byte_ids(ids, seq_len)


def meet_in_middle_batch(
    model,
    records: list[GraphRecord],
    tokenizer: TokenGTTokenizer,
    seq_len: int,
    device: torch.device,
    graph_autoregressive: bool = False,
    seed: int = 0,
    config: MeetInMiddleConfig | dict[str, Any] | None = None,
    forward_logits: torch.Tensor | None = None,
    forward_nll: torch.Tensor | None = None,
    require_grad: bool = False,
) -> dict[str, Any]:
    """Run a graph-aware forward/reverse meet-in-the-middle pass.

    The forward and reverse contexts use the same graph tokenization.  The byte
    stream is ordered by the configured graph autoregressive policy; the reverse
    pass then scores the same ordered bytes from right to left.  Agreement is
    measured at the configured split point by comparing the two distributions
    for the same original byte token.
    """

    cfg = config if isinstance(config, MeetInMiddleConfig) else meet_in_middle_config(config)
    if not cfg.enabled or not records:
        return {
            "enabled": False,
            "mode": cfg.mode,
            "reason": "disabled" if not cfg.enabled else "empty batch",
            "loss": torch.zeros((), device=device) if require_grad else None,
            "metrics": {"mim_enabled": 0.0},
            "records": [],
        }

    selected_records = records[: cfg.max_records] if cfg.max_records > 0 else records
    if forward_logits is not None:
        forward_logits = forward_logits[: len(selected_records)]
    xs_rev, ys_rev = zip(
        *(
            encode_record_bytes_reverse(
                record,
                seq_len,
                graph_autoregressive=graph_autoregressive,
                seed=seed,
            )
            for record in selected_records
        )
    )
    x_rev = torch.stack(xs_rev).to(device)
    y_rev = torch.stack(ys_rev).to(device)
    graph_batch = tokenizer.batch_encode(selected_records)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        reverse_out = model(x_rev, graph_batch, None)
        reverse_logits = reverse_out["logits"]
        reverse_nll = F.cross_entropy(reverse_logits.reshape(-1, reverse_logits.shape[-1]), y_rev.reshape(-1), ignore_index=PAD_ID)
        agreement = _agreement_terms(
            selected_records,
            forward_logits,
            reverse_logits,
            seq_len=seq_len,
            graph_autoregressive=graph_autoregressive,
            seed=seed,
            split_ratio=cfg.split_ratio,
        )
        agreement_loss = agreement["agreement_loss"]
        loss = cfg.agreement_weight * agreement_loss + cfg.reverse_nll_weight * reverse_nll

    metrics = {
        "mim_enabled": 1.0,
        "mim_shared_weight_reverse_pass": 1.0 if not cfg.reverse_model_path else 0.0,
        "mim_split_ratio": float(cfg.split_ratio),
        "mim_agreement_weight": float(cfg.agreement_weight),
        "mim_reverse_nll_weight": float(cfg.reverse_nll_weight),
        "mim_reverse_nll": _float_detached(reverse_nll),
        "mim_agreement_loss": _float_detached(agreement_loss),
        "mim_join_token_match_rate": agreement["match_rate"],
        "mim_true_meet_logprob_mean": agreement["true_logprob_mean"],
        "mim_candidate_count": float(len(selected_records)),
        "mim_loss": _float_detached(loss),
    }
    if forward_nll is not None and torch.is_tensor(forward_nll):
        metrics["mim_bidirectional_nll"] = _float_detached(0.5 * (forward_nll.detach() + reverse_nll.detach()))
    else:
        metrics["mim_bidirectional_nll"] = metrics["mim_reverse_nll"]
    return {
        "enabled": True,
        "mode": cfg.mode if cfg.reverse_model_path else "shared_weight_reverse_pass",
        "reverse_model_path": cfg.reverse_model_path,
        "loss": loss,
        "metrics": metrics,
        "records": agreement["records"],
    }


def _agreement_terms(
    records: Iterable[GraphRecord],
    forward_logits: torch.Tensor | None,
    reverse_logits: torch.Tensor,
    seq_len: int,
    graph_autoregressive: bool,
    seed: int,
    split_ratio: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    losses: list[torch.Tensor] = []
    true_log_probs: list[torch.Tensor] = []
    matches = 0
    valid = 0
    for batch_idx, record in enumerate(records):
        text = record.autoregressive_text(seed=seed) if graph_autoregressive else record.text
        ids = [b + 1 for b in text.encode("utf-8", "ignore")]
        n = min(len(ids), seq_len + 1)
        if n < 3:
            continue
        meet_index = min(max(int(round((n - 1) * float(split_ratio))), 1), n - 2)
        f_pos = meet_index - 1
        r_token_index = n - 1 - meet_index
        r_pos = r_token_index - 1
        if r_pos < 0 or f_pos < 0 or r_pos >= reverse_logits.shape[1]:
            continue
        true_token = int(ids[meet_index])
        rev_dist = reverse_logits[batch_idx, r_pos].float()
        rev_log_probs = F.log_softmax(rev_dist, dim=-1)
        rev_pred = int(rev_dist.argmax().detach().cpu())
        fwd_pred = None
        if forward_logits is not None and f_pos < forward_logits.shape[1]:
            fwd_dist = forward_logits[batch_idx, f_pos].float()
            fwd_log_probs = F.log_softmax(fwd_dist, dim=-1)
            fwd_pred = int(fwd_dist.argmax().detach().cpu())
            losses.append(_symmetric_kl(fwd_log_probs, rev_log_probs))
            true_log_probs.append(0.5 * (fwd_log_probs[true_token] + rev_log_probs[true_token]))
            if fwd_pred == rev_pred:
                matches += 1
        else:
            true_log_probs.append(rev_log_probs[true_token])
        valid += 1
        rows.append(
            {
                "record_id": record.record_id,
                "meet_index": int(meet_index),
                "forward_position": int(f_pos),
                "reverse_position": int(r_pos),
                "true_token": int(true_token),
                "true_byte": int(true_token - 1),
                "forward_argmax": fwd_pred,
                "reverse_argmax": rev_pred,
                "argmax_match": bool(fwd_pred == rev_pred) if fwd_pred is not None else False,
                "context_mode": "graph_random_order" if (record.metadata or {}).get("decoding_order_kind") == "random_autoregressive" else str((record.metadata or {}).get("decoding_order_kind", "")),
            }
        )
    if losses:
        agreement_loss = torch.stack(losses).mean()
    else:
        agreement_loss = reverse_logits.sum() * 0.0
    if true_log_probs:
        true_logprob_mean = _float_detached(torch.stack(true_log_probs).mean())
    else:
        true_logprob_mean = 0.0
    return {
        "agreement_loss": agreement_loss,
        "match_rate": float(matches / max(valid, 1)),
        "true_logprob_mean": true_logprob_mean,
        "records": rows,
    }


def _symmetric_kl(left_log_probs: torch.Tensor, right_log_probs: torch.Tensor) -> torch.Tensor:
    left_probs = left_log_probs.exp()
    right_probs = right_log_probs.exp()
    return 0.5 * (
        F.kl_div(left_log_probs, right_probs, reduction="sum", log_target=False)
        + F.kl_div(right_log_probs, left_probs, reduction="sum", log_target=False)
    )


def _float_detached(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)
