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
    agreement_kind: str = "total_variation"
    max_meet_points: int = 8
    ngram_size: int = 4
    verification_window: int = 4


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
        agreement_kind=str(raw.get("agreement_kind", "total_variation") or "total_variation"),
        max_meet_points=max(int(raw.get("max_meet_points", raw.get("num_meet_points", 8))), 0),
        ngram_size=max(int(raw.get("ngram_size", 4)), 1),
        verification_window=max(int(raw.get("verification_window", 4)), 1),
    )


def encode_record_bytes_reverse(
    record: GraphRecord,
    seq_len: int,
    graph_autoregressive: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if graph_autoregressive:
        text = record.autoregressive_text(seed=seed, direction="reverse")
        ids = [b + 1 for b in text.encode("utf-8", "ignore")]
    else:
        text = record.text
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
            config=cfg,
        )
        agreement_loss = agreement["agreement_loss"]
        loss = cfg.agreement_weight * agreement_loss + cfg.reverse_nll_weight * reverse_nll

    metrics = {
        "mim_enabled": 1.0,
        "mim_shared_weight_reverse_pass": 1.0 if not cfg.reverse_model_path else 0.0,
        "mim_split_ratio": float(cfg.split_ratio),
        "mim_agreement_weight": float(cfg.agreement_weight),
        "mim_reverse_nll_weight": float(cfg.reverse_nll_weight),
        "mim_max_meet_points": float(cfg.max_meet_points),
        "mim_ngram_size": float(cfg.ngram_size),
        "mim_verification_window": float(cfg.verification_window),
        "mim_reverse_nll": _float_detached(reverse_nll),
        "mim_agreement_loss": _float_detached(agreement_loss),
        "mim_join_token_match_rate": agreement["match_rate"],
        "mim_selected_meet_points_mean": agreement["selected_meet_points_mean"],
        "mim_candidate_meet_count_mean": agreement["candidate_meet_count_mean"],
        "mim_verified_meet_count_mean": agreement["verified_meet_count_mean"],
        "mim_ngram_candidate_rate": agreement["ngram_candidate_rate"],
        "mim_truth_verification_rate": agreement["truth_verification_rate"],
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
    config: MeetInMiddleConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    losses: list[torch.Tensor] = []
    true_log_probs: list[torch.Tensor] = []
    matches = 0
    valid = 0
    selected_counts: list[int] = []
    candidate_counts: list[int] = []
    verified_counts: list[int] = []
    for batch_idx, record in enumerate(records):
        forward_text = record.autoregressive_text(seed=seed, direction="forward") if graph_autoregressive else record.text
        reverse_text = record.autoregressive_text(seed=seed, direction="reverse") if graph_autoregressive else record.text
        ids = [b + 1 for b in forward_text.encode("utf-8", "ignore")]
        reverse_ids = [b + 1 for b in reverse_text.encode("utf-8", "ignore")]
        if not graph_autoregressive:
            reverse_ids.reverse()
        n = min(len(ids), len(reverse_ids), seq_len + 1)
        if n < 3:
            continue
        meet_indices = _selected_meet_indices(n, config.split_ratio, config.max_meet_points)
        selected_counts.append(len(meet_indices))
        fwd_argmax_by_index: dict[int, int] = {}
        rev_argmax_by_index: dict[int, int] = {}
        point_rows = []
        for meet_index in meet_indices:
            f_pos = meet_index - 1
            r_pos = meet_index - 1 if graph_autoregressive else n - 2 - meet_index
            if r_pos < 0 or f_pos < 0 or r_pos >= reverse_logits.shape[1] or meet_index >= len(ids) or r_pos + 1 >= len(reverse_ids):
                continue
            true_token = int(ids[meet_index])
            reverse_true_token = int(reverse_ids[r_pos + 1])
            rev_dist = reverse_logits[batch_idx, r_pos].float()
            rev_log_probs = F.log_softmax(rev_dist, dim=-1)
            rev_pred = int(rev_dist.argmax().detach().cpu())
            rev_argmax_by_index[meet_index] = rev_pred
            fwd_pred = None
            agreement_value = None
            if forward_logits is not None and f_pos < forward_logits.shape[1]:
                fwd_dist = forward_logits[batch_idx, f_pos].float()
                fwd_log_probs = F.log_softmax(fwd_dist, dim=-1)
                fwd_pred = int(fwd_dist.argmax().detach().cpu())
                fwd_argmax_by_index[meet_index] = fwd_pred
                loss_i = _distribution_agreement(fwd_log_probs, rev_log_probs, config.agreement_kind)
                losses.append(loss_i)
                agreement_value = _float_detached(loss_i)
                true_log_probs.append(0.5 * (fwd_log_probs[true_token] + rev_log_probs[reverse_true_token]))
                if fwd_pred == rev_pred:
                    matches += 1
            else:
                true_log_probs.append(rev_log_probs[reverse_true_token])
            valid += 1
            point_rows.append(
                {
                    "meet_index": int(meet_index),
                    "forward_position": int(f_pos),
                    "reverse_position": int(r_pos),
                    "true_token": int(true_token),
                    "true_byte": int(true_token - 1),
                    "reverse_true_token": int(reverse_true_token),
                    "reverse_true_byte": int(reverse_true_token - 1),
                    "same_true_token_alignment": bool(true_token == reverse_true_token),
                    "forward_argmax": fwd_pred,
                    "reverse_argmax": rev_pred,
                    "argmax_match": bool(fwd_pred == rev_pred) if fwd_pred is not None else False,
                    "agreement": agreement_value,
                }
            )
        candidate_count, verified_count = _ngram_verification_counts(
            ids[:n],
            fwd_argmax_by_index,
            rev_argmax_by_index,
            ngram_size=config.ngram_size,
            verification_window=config.verification_window,
        )
        candidate_counts.append(candidate_count)
        verified_counts.append(verified_count)
        rows.append(
            {
                "record_id": record.record_id,
                "context_mode": str((record.metadata or {}).get("decoding_order_kind", "")),
                "reverse_context_mode": str((record.metadata or {}).get("decoding_reverse_order_kind", "")),
                "graph_autoregressive": bool(graph_autoregressive),
                "selected_meet_points": len(point_rows),
                "candidate_meet_count": int(candidate_count),
                "verified_meet_count": int(verified_count),
                "ngram_size": int(config.ngram_size),
                "verification_window": int(config.verification_window),
                "meet_points": point_rows,
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
        "selected_meet_points_mean": float(sum(selected_counts) / max(len(selected_counts), 1)),
        "candidate_meet_count_mean": float(sum(candidate_counts) / max(len(candidate_counts), 1)),
        "verified_meet_count_mean": float(sum(verified_counts) / max(len(verified_counts), 1)),
        "ngram_candidate_rate": float(sum(candidate_counts) / max(sum(selected_counts), 1)),
        "truth_verification_rate": float(sum(verified_counts) / max(sum(selected_counts), 1)),
        "records": rows,
    }


def _selected_meet_indices(n: int, split_ratio: float, max_meet_points: int) -> list[int]:
    candidates = list(range(1, max(n - 1, 1)))
    if not candidates:
        return []
    center = min(max(int(round((n - 1) * float(split_ratio))), 1), n - 2)
    if max_meet_points <= 0 or max_meet_points >= len(candidates):
        selected = candidates
    else:
        if max_meet_points == 1:
            selected = [center]
        else:
            grid = torch.linspace(0, len(candidates) - 1, steps=max_meet_points).round().to(torch.long).tolist()
            selected = [candidates[int(idx)] for idx in grid]
            selected.append(center)
    return sorted(set(int(idx) for idx in selected if 1 <= int(idx) <= n - 2))


def _distribution_agreement(left_log_probs: torch.Tensor, right_log_probs: torch.Tensor, kind: str) -> torch.Tensor:
    normalized = str(kind).lower().replace("-", "_")
    if normalized in {"tv", "total_variation", "total_variation_distance"}:
        return 0.5 * (left_log_probs.exp() - right_log_probs.exp()).abs().sum()
    if normalized in {"symmetric_kl", "skl", "kl"}:
        return _symmetric_kl(left_log_probs, right_log_probs)
    raise ValueError(f"Unsupported meet-in-the-middle agreement_kind={kind!r}")


def _ngram_verification_counts(
    ids: list[int],
    forward_argmax: dict[int, int],
    reverse_argmax: dict[int, int],
    ngram_size: int,
    verification_window: int,
) -> tuple[int, int]:
    if not ids:
        return 0, 0
    ngram_size = max(int(ngram_size), 1)
    verification_window = max(int(verification_window), ngram_size)
    candidate_count = 0
    verified_count = 0
    max_start = max(len(ids) - ngram_size, 0)
    for start in range(1, max_start + 1):
        ngram_positions = list(range(start, start + ngram_size))
        if not all(pos in forward_argmax and pos in reverse_argmax for pos in ngram_positions):
            continue
        if all(forward_argmax[pos] == reverse_argmax[pos] for pos in ngram_positions):
            candidate_count += 1
            verify_positions = list(range(start, min(start + verification_window, len(ids) - 1)))
            if verify_positions and all(
                forward_argmax.get(pos) == reverse_argmax.get(pos) == int(ids[pos])
                for pos in verify_positions
            ):
                verified_count += 1
    return candidate_count, verified_count


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
